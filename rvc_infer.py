import torch
from infer.lib.rtrvc import RVC, Config

# 初始化配置
config = Config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.is_half = True if config.device == "cuda" else False

# 初始化RVC
rvc = RVC(
    key=0,  # f0 up key
    formant=0,  # formant shift
    pth_path="path/to/your/model.pth",
    index_path="path/to/your/index.index", 
    index_rate=0.75,
    n_cpu=1,
    inp_q=None,
    opt_q=None,
    config=config
)

# 加载音频
input_wav = torch.tensor(load_audio("path/to/input.wav", sr=16000))

# 设置参数
block_frame_16k = 16000  # 1秒音频
skip_head = 0
return_length = input_wav.shape[0]
f0method = "rmvpe"  # 可选: "pm", "harvest", "crepe", "rmvpe", "fcpe"

# 执行语音转换
output_audio = rvc.infer(
    input_wav,
    block_frame_16k,
    skip_head,
    return_length,
    f0method
)

# 保存输出音频
save_audio("path/to/output.wav", output_audio.cpu().numpy(), sr=rvc.tgt_sr)

print("语音转换完成!")