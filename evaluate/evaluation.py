import numpy as np
from numpy.linalg import norm
import os
import whisper
import jiwer
from ..prepare.preprocess_f0 import compute_f0
import librosa
import crepe
import torch
from pymcd.mcd import Calculate_MCD

out_dict = {}

def compute_f0(filename):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
    audio = audio + torch.randn_like(audio) * 0.001
    # Here we'll use a 10 millisecond hop length
    hop_length = 160
    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1000
    # Select a model capacity--one of "tiny" or "full"
    model = "full"
    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 512
    # Compute pitch using first gpu
    pitch, periodicity = crepe.predict(
        audio,
        sr,
        hop_length,
        fmin,
        fmax,
        model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    # CREPE was not trained on silent audio. some error on silent need filter.pitPath
    periodicity = crepe.filter.median(periodicity, 7)
    pitch = crepe.filter.mean(pitch, 5)
    pitch[periodicity < 0.5] = 0
    pitch = pitch.squeeze(0)
    return pitch
'''
=================================== 
tgt_path: path to speaker embedding file of target singer
conv_folder: path to folder of speaker embedding files of converted songs
===================================
'''
def singer_sim(tgt_path, conv_folder):
    tgt_emb = np.load(tgt_path)
    for conv_npy in os.listdir(conv_folder):
        conv_emb = np.load(os.path.join(conv_folder, conv_npy))
        cos_simmilarity = np.dot(tgt_emb, conv_emb)/norm(tgt_emb)/norm(conv_emb)
        if os.path.splitext(conv_npy)[0] not in out_dict:
            out_dict[os.path.splitext(conv_npy)[0]] = {"spk": cos_simmilarity}
        else:
            out_dict[os.path.splitext(conv_npy)[0]]["spk"] = cos_simmilarity
'''
=================================== 
read the file with same file name in src_folder and conv_folder to compute cer
src_folder: path to audio files of source singer
tgt_folder: path to converted audio files of target singer
===================================
'''
def cer(src_folder, conv_folder):
    model = whisper.load_model("/mnt/data/ycevan/svc/so-vits-svc-5.0/whisper_pretrain/large-v3.pt")
    for audio in os.listdir(src_folder):
        src = model.transcribe(os.path.join(src_folder, audio))
        tgt = model.transcribe(os.path.join(conv_folder, audio))
        cer = jiwer.cer(src['text'], tgt['text'])
        base = os.path.splitext(audio)[0]
        if base not in out_dict:
            out_dict[base] = {"cer": cer}
        else:
            out_dict[base]["cer"] = cer
'''
=================================== 
src_folder: path to audio files of source singer
conv_folder: path to audio files of converted songs of target singer
return: correlation & MSE between f0 
===================================
'''
def f0_corr_and_mse(src_folder, conv_folder):
    for audio in os.listdir(src_folder):
        base = os.path.splitext(audio)[0]
        src = compute_f0(os.path.join(src_folder, audio))
        conv = compute_f0(os.path.join(conv_folder, audio))
        corr = np.corrcoef(src, conv)
        rmse = np.sqrt(((src - conv)**2).mean(axis=0))
        if base not in out_dict:
            out_dict[base] = {"f0_corr": corr[0, 1], "f0_rmse": rmse}
        else:
            out_dict[base]["f0_corr"] = corr[0, 1]
            out_dict[base]["f0_rmse"] = rmse
'''
=================================== 
src_folder: path to audio files of source singer
conv_folder: path to audio files of converted songs of target singer
return: Mel Cepstral Distance 
===================================
'''
def MCD(src_folder, conv_folder):
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    for audio in os.listdir(src_folder):
        base = os.path.splitext(audio)[0]
        mcd = mcd_toolbox.calculate_mcd(os.path.join(src_folder, audio), os.path.join(conv_folder, audio))
        if base not in out_dict:
            out_dict[base] = { "mcd": mcd }
        else:
            out_dict[base]["mcd"] = mcd
'''
out_dict = { "Alto-1#newboy_0000": {'cer':0.95 , 'spk': 0.89, 'f0_corr': 0.99, 'MCD': 0.55} }
'''

        
