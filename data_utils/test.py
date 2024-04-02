"""def get_encodec(audio_fn,model):
    samples, sr = ta.load(audio_fn)
    seq_lenth = samples.shape[1] / sr
    print("sr:",sr)
    samples = samples.to('cuda')
    
    if samples.shape[0] > 1:
        samples = torch.mean(samples, dim=0, keepdim=True)
    print("samples audio:",sample_audio.shape)
    with torch.no_grad():
        #model = EncodecModel.encodec_model_24khz().to('cuda')
        model.set_target_bandwidth(6)
        samples = samples.unsqueeze(0)
        codes_raw = model.encode(samples)
        for frame in codes_raw:
            codes,_ = frame
            codes = codes.transpose(0,1)
            emb = model.quantizer.decode(codes)
        emb = emb.transpose(1,2)
        emb = linear_interpolation(emb,seq_len=seq_lenth,output_fps=30,output_len=None)
        emb = emb.squeeze(0).cpu().numpy()
    return emb"""

def get_encodec(audio_fn,model):
    wav, sr = torchaudio.load(audio_fn)
    model.set_target_bandwidth(6.0)
    #print(sr,wav.shape)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    #print(wav.shape)
    seq_lenth = wav.shape[1]/model.sample_rate
    #print(seq_lenth)
    wav = wav.unsqueeze(0).to("cuda")
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze(0)+1
    
    token_sample = get_target_token(fps=30,codes=codes,duration_time=seq_lenth,num_codes_layer=-3)
      # [B, n_q, T]
    return token_sample

def interpolate_vector(input_vector, outdim):
    indim = input_vector.shape[1]
    interval = indim / outdim
    # 生成采样索引
    idx = (np.arange(outdim) * interval).astype(int)

    # 等间隔采样
    output_vector = input_vector[:, idx]

    return output_vector

def get_target_token(fps,codes,duration_time,num_codes_layer):
    seq_len = fps*duration_time
    #print(codes.shape) ### 8x750
    token_codes = codes[num_codes_layer,:].unsqueeze(0) ### 1x750
    for t in token_codes:
        p = torch.unique_consecutive(t)
    print(p.shape)
    token_sample = interpolate_vector(token_codes,seq_len) ### 1x300
    #print(token_sample.shape)
    return token_sample


import numpy as np


if __name__ == "__main__":
    audio_fn = '/mnt/nj-aigc/usr/pengwenshuo/TalkSHOW/demo_audio/214428-00_00_58-00_01_08.wav'
    from encodec import EncodecModel
    from encodec.utils import convert_audio
    from pathlib import Path
    import torchaudio
    import torch
    model = EncodecModel.encodec_model_24khz(repository=Path("/mnt/nj-aigc/dataset/pengwenshuo/encodec")).to('cuda')
    codes = get_encodec(audio_fn,model)
    codes = codes+1
    for code in codes:
        c = torch.unique_consecutive(code)
    print(c.shape)
    #print(torch.max(codes),torch.min(codes))
    #token = get_target_token(fps=30,codes=codes,duration_time=duration_time,num_codes_layer=-1)

    
