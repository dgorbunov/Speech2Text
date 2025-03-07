import os
import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from librispeech import LibriSpeech, NUM_MELS

DATA_PATH = "./data"
TEST_SUBSET = "dev-clean"
OUTPUT_DIR = "./mel_to_audio"
NUM_SAMPLES = 3

class MelToAudio:
    def __init__(self, n_fft=1024, hop_length=160, win_length=400, n_mels=NUM_MELS, sample_rate=16000):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        
        # Create mel spectrogram transform with the same parameters as in LibriSpeech
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels
        )
        
        # Create the inverse mel transform
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        # Griffin-Lim algorithm for phase reconstruction
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_iter=60  # More iterations for better quality
        )
    
    def mel_to_audio(self, mel_spectrogram):
        # Convert mel spectrogram to linear spectrogram
        linear_spectrogram = self.inverse_mel(mel_spectrogram)
        
        # Use Griffin-Lim algorithm to recover phase information and convert to audio
        waveform = self.griffin_lim(linear_spectrogram)
        
        # Normalize the waveform to have reasonable volume
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8) * 0.9
        
        return waveform

def process_dataset(dataset_path, subset, output_dir, num_samples=5):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print(f"Loading LibriSpeech {subset} dataset...")
    librispeech_dataset = LibriSpeech(dataPath=dataset_path, subset=subset)
    
    # Create mel to audio converter with the same parameters as in LibriSpeech
    converter = MelToAudio(
        n_fft=1024,
        hop_length=160,
        win_length=400
    )
    
    print(f"Processing {num_samples} samples...")
    
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Get fixed indices for samples
    indices = list(range(min(num_samples, len(librispeech_dataset.dataset))))
    
    for idx in tqdm(indices):
        # Get the original audio and transcript directly from the LibriSpeech dataset
        waveform, sample_rate, transcript, _, _, _ = librispeech_dataset.dataset[idx]
        
        # Create mel spectrogram directly using the same transform as in the dataset
        mel_spec = librispeech_dataset.mel_transform(waveform).squeeze(0)
        
        # Convert mel spectrogram back to audio
        reconstructed_audio = converter.mel_to_audio(mel_spec)
        
        # Normalize original audio for fair comparison
        original_waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8) * 0.9
        
        # Save files
        sample_dir = output_dir / f"sample_{idx+1}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save original audio
        torchaudio.save(
            sample_dir / "original.wav",
            original_waveform,
            sample_rate
        )
        
        # Save reconstructed audio
        torchaudio.save(
            sample_dir / "reconstructed.wav",
            reconstructed_audio.unsqueeze(0),  # Add channel dimension
            converter.sample_rate
        )
        
        # Save mel spectrogram visualization
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Mel Frequency')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(sample_dir / "mel_spectrogram.png")
        plt.close()
        
        # Create comparison visualization
        plt.figure(figsize=(12, 8))
        
        # Plot original waveform
        plt.subplot(3, 1, 1)
        plt.plot(original_waveform[0].numpy())
        plt.title('Original Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        # Plot mel spectrogram
        plt.subplot(3, 1, 2)
        plt.imshow(mel_spec.numpy(), aspect='auto', origin='lower')
        plt.title('Mel Spectrogram')
        plt.ylabel('Mel Frequency')
        
        # Plot reconstructed waveform
        plt.subplot(3, 1, 3)
        plt.plot(reconstructed_audio.numpy())
        plt.title('Reconstructed Waveform')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(sample_dir / "waveform_comparison.png")
        plt.close()
        
        # Save info file
        with open(sample_dir / "info.txt", "w") as f:
            f.write(f"Transcript: {transcript}\n")
            f.write(f"Mel bands: {NUM_MELS}\n")
            f.write(f"FFT size: {converter.n_fft}\n")
            f.write(f"Hop length: {converter.hop_length}\n")
            f.write(f"Win length: {converter.win_length}\n")
            f.write(f"Original waveform shape: {original_waveform.shape}\n")
            f.write(f"Mel spectrogram shape: {mel_spec.shape}\n")
            f.write(f"Reconstructed audio shape: {reconstructed_audio.shape}\n")
        
        print(f"Processed sample {idx+1}/{num_samples}")
        print(f"Transcript: {transcript}")
        print(f"Files saved to {sample_dir}")
        print()
        
process_dataset(DATA_PATH, TEST_SUBSET, OUTPUT_DIR, NUM_SAMPLES)