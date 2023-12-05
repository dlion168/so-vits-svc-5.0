import os 
from tqdm import tqdm
import argparse
from inference import load_csv_pitch
import numpy as np
import json
import concurrent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav_dir", type=str, help="wav", required=True)
    parser.add_argument("-p", "--pit_dir", type=str, help="pit", required=True) # csv for excel
    parser.add_argument("-k", "--pkl_path", type=str, help="pkl", required=True)
    parser.add_argument("-n", "--num_process", type=int, default=10, help="number of parallel process", required=True)
    args = parser.parse_args()
    executor = concurrent.futures.ProcessPoolExecutor(args.num_process)
    os.makedirs(args.pit_dir, exist_ok=True)
    singer_median_pitch = {}
    for wav_folder in tqdm(os.listdir(args.wav_dir)):
        os.makedirs(os.path.join(args.pit_dir, wav_folder), exist_ok=True)
        futures = []
        for wav_file in os.listdir(os.path.join(args.wav_dir, wav_folder)):
            if not os.path.exists(os.path.join(args.pit_dir, wav_folder, os.path.splitext(wav_file)[0]+'.csv')):
                futures.append(executor.submit(
                    os.system, 
                    f"python pitch/inference.py -w {os.path.join(args.wav_dir, wav_folder, wav_file)} -p {os.path.join(args.pit_dir, wav_folder, os.path.splitext(wav_file)[0]+'.csv')}"
                ))
        concurrent.futures.wait(futures)
        pit = []
        for csv_file in os.listdir(os.path.join(args.pit_dir, wav_folder)):
            pit += load_csv_pitch(os.path.join(args.pit_dir, wav_folder, csv_file))
        pit_median = np.median(np.array(pit))
        singer_median_pitch[wav_folder] = pit_median
    with open(args.pkl_path, 'w') as f:
        json.dump(singer_median_pitch, f)
    
