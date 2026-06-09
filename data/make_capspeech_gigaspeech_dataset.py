import os
import pandas as pd
from tqdm import tqdm
import mutagen

dataset, path, ref = 'GigaSpeech','/export/corpora7/CapSpeech-GigaSpeech/', 'gigaspeech'

print(f"Doing {dataset}")
dfs = []
for split in ['test', 'train', 'validation']:
    output_dir = f"data/{dataset}_{split}"

    df = pd.read_csv(f"/export/corpora7/capspeechset/{split}_PT_{ref}_caption.csv")
    metadata = pd.read_csv(f"/export/corpora7/capspeechset/{split}_{ref}.csv")
    
    print(split, df.shape, metadata.shape)
    df = pd.merge(df, metadata, on='audio_path', suffixes=('', '_y'))
    print("after merge:", df.shape)
    print(df.columns.tolist())
    df['split'] = split
    
    os.makedirs(output_dir, exist_ok=True)

    df['duration'] = df['speech_duration']
    df['storage_path'] = [path+a for a in df['audio_path']]
    df['sample_freq'] = 16000
    df['id'] = [p.split('/')[-1][:-4] for p in df['storage_path']]
    df['speaker'] = [p.split('_')[0] for p in df['id']]

    to_rem = [] #'/export/corpora7/CapSpeech-GigaSpeech/gigaspeech/YOU1000000117_S0000062.wav'
    for f in tqdm(list(df['storage_path']), desc='removing missing files'):
        if not os.path.exists(f): to_rem.append(f)
    print(f"{len(to_rem)} files to remove.")
    df = df[~df['storage_path'].isin(to_rem)]
    print('new shape :',df.shape)

    segments = df[['id', 'duration', 'speaker', 'split', 'pitch','age','gender', 'speaking_rate', 'speech_monotony', 'accent']]
    recordings = df[['id', 'duration', 'storage_path', 'sample_freq']]

    segments['corpusid'] = dataset
    # segments['speaker'] = [f"{dataset}_{i}" for i in range(len(segments))]
    print(f"{dataset}: recordings {recordings.shape}, segments {segments.shape}")
    segments.to_csv(f"{output_dir}/segments.csv", index=False)
    recordings.to_csv(f"{output_dir}/recordings.csv", index=False)
    print("Recordings:")
    print(recordings.head())
    print("Segments:")
    print(segments.head())
