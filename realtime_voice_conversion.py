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

INPUT_WAV = "path/to/your/input.wav"
MODEL_PATH = "path/to/your/model.pth"
INDEX_PATH = "path/to/your/index.index"

class FakeStream:
    def __init__(self, wav_file, callback, samplerate, blocksize, channels):
        self.callback = callback
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.wav_file = wave.open(wav_file, 'rb')
        self.is_running = False

    def start(self):
        self.is_running = True
        while self.is_running:
            audio_data = self.wav_file.readframes(self.blocksize)
            if not audio_data:
                self.wav_file.rewind()
                continue
            
            indata = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            indata = indata.reshape(-1, self.channels)
            
            outdata = np.zeros_like(indata)
            self.callback(indata, outdata, self.blocksize, time.time(), None)
            
            # 模拟实时处理的延迟
            time.sleep(self.blocksize / self.samplerate)

    def stop(self):
        self.is_running = False
        self.wav_file.close()

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
        self.block_time = 0.25
        self.block_frame = int(self.samplerate * self.block_time)
        self.extra_time = 2.5
        self.extra_frame = int(self.samplerate * self.extra_time)
        self.crossfade_time = 0.05
        self.crossfade_frame = int(self.samplerate * self.crossfade_time)
        self.zc = self.samplerate // 100

    def load_model(self):
        # 这里需要设置正确的模型路径
        pth_path = MODEL_PATH
        index_path = INDEX_PATH
        self.rvc = rvc_for_realtime.RVC(
            0,  # pitch
            0,  # formant
            pth_path,
            index_path,
            0.0,  # index_rate
            1,  # n_cpu
            None,  # inp_q
            None,  # opt_q
            self.config
        )

    def initialize_buffers(self):
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.block_frame,
            device=self.config.device,
            dtype=torch.float32
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32
        )
        self.sola_buffer = torch.zeros(
            self.crossfade_frame,
            device=self.config.device,
            dtype=torch.float32
        )
        self.fade_in_window = torch.sin(
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=self.config.device, dtype=torch.float32)
        ) ** 2
        self.fade_out_window = 1 - self.fade_in_window

    def setup_audio_stream(self):
        self.stream = FakeStream(
            wav_file=self.wav_file,
            callback=self.audio_callback,
            samplerate=self.samplerate,
            blocksize=self.block_frame,
            channels=self.channels
        )

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)

        start_time = time.perf_counter()

        # 处理输入音频
        indata = librosa.to_mono(indata.T)
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = torch.from_numpy(indata).to(self.config.device)

        # 重采样到16kHz
        self.input_wav_res[:-160 * (indata.shape[0] // self.zc + 1)] = self.input_wav_res[160 * (indata.shape[0] // self.zc + 1):].clone()
        self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1):] = F.interpolate(
            self.input_wav[-indata.shape[0] - 2 * self.zc:].unsqueeze(0).unsqueeze(0),
            scale_factor=160 / self.zc,
            mode='linear',
            align_corners=False
        ).squeeze()

        # 执行语音转换
        infer_wav = self.rvc.infer(
            self.input_wav_res,
            160 * self.block_frame // self.zc,
            self.extra_frame // self.zc,
            (self.block_frame + self.crossfade_frame) // self.zc,
            "fcpe"  # 可以根据需要更改音高提取方法
        )

        # 重采样回原始采样率（如果需要）
        if self.rvc.tgt_sr != self.samplerate:
            infer_wav = F.interpolate(
                infer_wav.unsqueeze(0).unsqueeze(0),
                scale_factor=self.samplerate / self.rvc.tgt_sr,
                mode='linear',
                align_corners=False
            ).squeeze()

        # SOLA算法
        conv_input = infer_wav[None, None, :self.crossfade_frame + self.zc]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input**2, torch.ones(1, 1, self.crossfade_frame, device=self.config.device)) + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]

        # 应用交叉淡入淡出
        infer_wav[:self.crossfade_frame] *= self.fade_in_window
        infer_wav[:self.crossfade_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame:self.block_frame + self.crossfade_frame]

        # 将转换后的音频写入输出缓冲区
        outdata[:] = infer_wav[:self.block_frame].repeat(self.channels, 1).t().cpu().numpy()

        total_time = time.perf_counter() - start_time
        print(f"处理时间: {total_time*1000:.2f} ms")

    def start(self):
        print("开始实时语音转换...")
        self.stream.start()

    def stop(self):
        print("实时语音转换已停止。")
        self.stream.stop()

if __name__ == "__main__":
    vc = RealtimeVC()
    try:
        vc.start()
    except KeyboardInterrupt:
        vc.stop()