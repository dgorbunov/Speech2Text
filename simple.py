import torchaudio
from torchaudio.datasets import LIBRISPEECH
from transformers import pipeline 
import librosa
import numpy as np
import os
import glob
from pathlib import Path
from jiwer import cer
from tqdm import trange, tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-100h")
DATAPATH = "./data"


def load_dataset(root_dir):
    data = []
    path = Path(root_dir)
    print(path)
    for flac_path in path.glob('**/*.flac'):
        dir_path = os.path.dirname(flac_path)
        base_filename = os.path.splitext(os.path.basename(flac_path))[0]
        parts = base_filename.split('-')
        if len(parts) < 2:
            continue
        label_prefix = f"{parts[0]}-{parts[1]}"
        label_file = os.path.join(dir_path, f"{label_prefix}.trans.txt")
        if not os.path.exists(label_file):
            print(f"Label file not found for {flac_path}")
            continue

        labels = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split(maxsplit=1)
                if len(tokens) == 2:
                    uid, transcription = tokens
                    labels[uid] = transcription

        transcription = labels.get(base_filename)
        if transcription is None:
            print(f"Warning: No transcription found for {base_filename} in {label_file}")
            continue

        waveform, sr = librosa.load(flac_path, sr=None)
        data.append((waveform, transcription))

    return data


if __name__ == "__main__":
    root_directory = '/data/LibriSpeech/dev-clean'
    dataset = load_dataset(root_directory)
    print(f"Loaded {len(dataset)} audio-transcription pairs.")
    l = len(dataset)
    sum = 0

    for d in tqdm(dataset):
        audio, text = d
        pred = pipe(audio)
        sum += cer(text, pred['text'])

    print(sum/l)

