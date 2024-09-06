import os
import sys
import time
import torch
import numpy as np
import sounddevice as sd
import librosa
import torch.nn.functional as F
import torchaudio.transforms as tat
from infer.lib import rtrvc as rvc_for_realtime
from configs.config import Config
from tools.torchgate import TorchGate
import wave
import soundfile as sf

INPUT_WAV = "path/to/your/input.wav"
MODEL_PATH = "path/to/your/model.pth"
INDEX_PATH = "path/to/your/index.index"

def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result

class FakeStream:
    def __init__(self, wav_file, callback, samplerate, blocksize, channels):
        self.callback = callback
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.wav_file = wave.open(wav_file, 'rb')
        self.is_running = False
        self.output_data = []  # 用于存储输出音频数据

    def start(self):
        self.is_running = True
        while self.is_running:
            audio_data = self.wav_file.readframes(self.blocksize)
            if not audio_data:
                break  # 文件结束时退出循环
            
            indata = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            indata = indata.reshape(-1, self.channels)
            
            outdata = np.zeros_like(indata)
            self.callback(indata, outdata, self.blocksize, time.time(), None)
            
            # 收集输出数据
            self.output_data.append(outdata)
            
            # 模拟实时处理的延迟
            time.sleep(self.blocksize / self.samplerate)
        
        self.stop()

    def stop(self):
        self.is_running = False
        self.wav_file.close()
        
        # 保存收集的输出数据为WAV文件
        if self.output_data:
            output_array = np.concatenate(self.output_data, axis=0)
            output_filename = "output_converted.wav"
            sf.write(output_filename, output_array, self.samplerate)
            print(f"转换后的音频已保存为: {output_filename}")

class RealtimeVC:
    def __init__(self):
        self.config = Config()
        self.setup_audio_parameters()
        self.load_model()
        self.initialize_buffers()
        self.wav_file = INPUT_WAV  # 设置您的WAV文件路径
        self.setup_audio_stream()

    def setup_audio_parameters(self):
        self.samplerate = 44100  # 可以根据需要调整
        self.channels = 1
        self.zc = self.samplerate // 100
        self.block_time = 0.25
        self.block_frame = int(np.round(self.block_time * self.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_time = 0.05
        self.crossfade_frame = int(np.round(self.crossfade_time * self.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_time = 2.5
        self.extra_frame = int(np.round(self.extra_time * self.samplerate / self.zc)) * self.zc
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc

        # 原gui_config参数
        self.threhold = -60
        self.pitch = 0
        self.formant = 0.0
        self.I_noise_reduce = False
        self.O_noise_reduce = False
        self.use_pv = False
        self.rms_mix_rate = 0.0
        self.index_rate = 0.0
        self.f0method = "rmvpe"  # 默认使用fcpe方法

    def load_model(self):
        # 这里需要设置正确的模型路径
        pth_path = MODEL_PATH
        index_path = INDEX_PATH
        self.rvc = rvc_for_realtime.RVC(
            self.pitch,  # pitch
            self.formant,  # formant
            pth_path,
            index_path,
            self.index_rate,  # index_rate
            1,  # n_cpu
            None,  # inp_q
            None,  # opt_q
            self.config
        )

    def initialize_buffers(self):
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.config.device,
            dtype=torch.float32
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame,
            device=self.config.device,
            dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.fade_in_window = torch.sin(
            0.5 * np.pi * torch.linspace(
                0.0,
                1.0,
                steps=self.sola_buffer_frame,
                device=self.config.device,
                dtype=torch.float32
            )
        ) ** 2
        self.fade_out_window = 1 - self.fade_in_window

        self.resampler = tat.Resample(
            orig_freq=self.samplerate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(self.config.device)

        if self.rvc.tgt_sr != self.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr,
                new_freq=self.samplerate,
                dtype=torch.float32,
            ).to(self.config.device)
        else:
            self.resampler2 = None

        self.tg = TorchGate(
            sr=self.samplerate,
            n_fft=4 * self.zc,
            prop_decrease=0.9
        ).to(self.config.device)

    def setup_audio_stream(self):
        self.stream = FakeStream(
            wav_file=self.wav_file,
            callback=self.audio_callback,
            samplerate=self.samplerate,
            blocksize=self.block_frame,
            channels=self.channels
        )

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        
        # 处理输入音频
        if self.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()

        # 输入降噪和重采样
        if self.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = self.resampler(
                self.input_wav[-indata.shape[0] - 2 * self.zc :]
            )[160:]
        
        # 执行语音转换
        infer_wav = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            self.f0method,
        )
        
        if self.resampler2 is not None:
            infer_wav = self.resampler2(infer_wav)
        
        # 输出降噪
        if self.O_noise_reduce:
            self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
        
        # volume envelop mixing
        if self.rms_mix_rate < 1:
            if self.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)
            )

        # SOLA算法
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        
        if not self.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        else:
            infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        
        outdata[:] = infer_wav[: self.block_frame].repeat(self.channels, 1).t().cpu().numpy()
        
        total_time = time.perf_counter() - start_time
        print(f"处理时间: {total_time*1000:.2f} ms")

    def start(self):
        print("开始实时语音转换...")
        self.stream.start()
        print("实时语音转换已完成。")

    def stop(self):
        print("实时语音转换已停止。")
        self.stream.stop()

if __name__ == "__main__":
    vc = RealtimeVC()
    try:
        vc.start()
    except KeyboardInterrupt:
        vc.stop()