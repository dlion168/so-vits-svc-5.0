import numpy as np
from numpy.linalg import norm
import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import whisper
import jiwer
import librosa
import crepe
import torch
from pymcd.mcd import Calculate_MCD
import csv
from tqdm import tqdm



def compute_f0(filename):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio, sr = librosa.load(filename, sr=16000)
    assert sr == 16000
    # Load audio
    audio = torch.tensor(np.copy(audio))[None]
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
conv_folder: path to folder of converted songs
===================================
'''
def singer_sim(tgt_path, conv_folder, out_dict):
    tgt_emb = np.load(tgt_path)
    spk_folder = os.path.join('./_svc_out', 'spk')
    sum_simmilarity = 0
    count = 0
    os.system(f"python prepare/preprocess_speaker.py {conv_folder} {spk_folder} -t 8")
    for conv_npy in tqdm(os.listdir(spk_folder), desc="singer_sim"):
        conv_emb = np.load(os.path.join(spk_folder, conv_npy))
        cos_simmilarity = np.dot(tgt_emb, conv_emb)/norm(tgt_emb)/norm(conv_emb)
        if os.path.splitext(conv_npy)[0] not in out_dict:
            out_dict[conv_npy.split('.')[0]] = {"spk": cos_simmilarity}
        else:
            out_dict[conv_npy.split('.')[0]]["spk"] = cos_simmilarity
        sum_simmilarity += cos_simmilarity
        count += 1
    print(f"Average cosine similarity: { sum_simmilarity / count }")
    return out_dict
'''
=================================== 
read the file with same file name in src_folder and conv_folder to compute cer
src_folder: path to audio files of source singer
tgt_folder: path to converted audio files of target singer
Note: Add initial_prompt="以下是普通话的句子" for chinese to improve accuracy.
===================================
'''
def cer(src_folder, conv_folder, out_dict, language):
    model = whisper.load_model("/mnt/data/ycevan/svc/so-vits-svc-5.0/whisper_pretrain/large-v3.pt")
    sum_cer = 0
    count = 0
    for audio in tqdm(os.listdir(src_folder), desc="CER"):
        src = model.transcribe(os.path.join(src_folder, audio), language=language)
        tgt = model.transcribe(os.path.join(conv_folder, audio), language=language)
        cer = jiwer.cer(src['text'], tgt['text'])
        base = os.path.splitext(audio)[0]
        if base not in out_dict:
            out_dict[base] = {"cer": cer}
        else:
            out_dict[base]["cer"] = cer
        sum_cer += cer
        count += 1
    print(f"Average CER: { sum_cer / count }")
    return out_dict
'''
=================================== 
src_folder: path to audio files of source singer
conv_folder: path to audio files of converted songs of target singer
return: correlation & MSE between f0 
===================================
'''
def f0_corr_and_mse(src_folder, conv_folder, out_dict):
    sum_corr = 0
    sum_rmse = 0
    count = 0
    for audio in tqdm(os.listdir(src_folder), desc="F0"):
        base = os.path.splitext(audio)[0]
        src = compute_f0(os.path.join(src_folder, audio))
        conv = compute_f0(os.path.join(conv_folder, audio))
        if (len(src)>len(conv)):
            src = src[:len(conv)]
        elif (len(src)<len(conv)):
            conv = conv[:len(conv)]
        corr = np.corrcoef(src, conv)
        rmse = np.sqrt(((src - conv)**2).mean(axis=0))
        if base not in out_dict:
            out_dict[base] = {"f0_corr": corr[0, 1], "f0_rmse": rmse.numpy()}
        else:
            out_dict[base]["f0_corr"] = corr[0, 1]
            out_dict[base]["f0_rmse"] = rmse.numpy()
        sum_corr += corr[0, 1]
        sum_rmse += rmse
        count += 1
    print(f"Average f0 corr: { sum_corr / count }")
    print(f"Average f0 RMSE: { sum_rmse / count }")
    return out_dict
'''
=================================== 
src_folder: path to audio files of source singer
conv_folder: path to audio files of converted songs of target singer
return: Mel Cepstral Distance 
===================================
'''
def mcd(src_folder, conv_folder, out_dict):
    sum_mcd = 0
    count = 0
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    for audio in tqdm(os.listdir(src_folder), desc="MCD"):
        base = os.path.splitext(audio)[0]
        mcd = mcd_toolbox.calculate_mcd(os.path.join(src_folder, audio), os.path.join(conv_folder, audio))
        if base not in out_dict:
            out_dict[base] = { "mcd": mcd }
        else:
            out_dict[base]["mcd"] = mcd
        sum_mcd += mcd
        count += 1
    print(f"Average MCD: { sum_mcd / count }")
    return out_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spk', type=str, required=True,
                        help="path to target speaker embedding file")
    parser.add_argument('--src', type=str, required=True,
                        help="path to original testing set folder (before conversion)")
    parser.add_argument('--tgt', type=str, required=True,
                        help="path to converted singing voice folder")
    parser.add_argument('--out_path', type=str, default="_svc_out/evaluation.tsv",
                        help="path to evaluation result folder")
    parser.add_argument('--language', type=str, default=None,
                        help="transcription language for cer")
    args = parser.parse_args()
    
    out_dict = {}
    #out_dict = singer_sim(args.spk, args.tgt, out_dict)
    #out_dict = cer(args.src, args.tgt, out_dict, language=args.language)
    out_dict = f0_corr_and_mse(args.src, args.tgt, out_dict)
    #out_dict = mcd(args.src, args.tgt, out_dict)
    # out_dict = { "Alto-1#newboy_0000": {'cer':0.95 , 'spk': 0.89, 'f0_corr': 0.99, 'f0_mse': 2.35, 'MCD': 0.55} }
    for key in out_dict.keys():
        out_dict[key]['name'] = key
    list_dict = [ out_dict[key] for key in out_dict.keys()]
    labels = ["name", "spk", "cer", "f0_corr", "f0_rmse", "mcd"]
    
    with open(args.out_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=labels, delimiter="\t")
        writer.writeheader()
        for elem in list_dict:
            writer.writerow(elem)
