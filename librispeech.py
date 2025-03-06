import torch
import torchaudio.transforms as T
from torchaudio.datasets import LIBRISPEECH
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

NUM_MELS = 60 # num mel frequencies
FFT_BINS = 1024 # num fft frequencies
FFT_WIN_LENGTH = 400 # samples in window (400 = 25ms at 16kHz sample rate)
SPEC_HOP_LENGTH = 160 # samples between spectrogram frames (overlap - 160 = 60% overlap)

# Character list for encoding/decoding
# Space is the first character (index 0)
CHAR_MAP = " abcdefghijklmnopqrstuvwxyz'"
NUM_CLASSES = len(CHAR_MAP)
BLANK_INDEX = CHAR_MAP.index(" ")  # Use space as blank for CTC

class LibriSpeech(torch.utils.data.Dataset):
    # Static class properties
    NUM_CLASSES = NUM_CLASSES
    BLANK_INDEX = BLANK_INDEX
    CHAR_MAP = CHAR_MAP
    
    def __init__(self, dataPath, subset):
        # Create data folder if doesn't exist
        Path(dataPath).mkdir(exist_ok=True, parents=True)

        # Load dataset (download if not found)
        self.dataset = LIBRISPEECH(root=dataPath, url=subset, download=True)
        
        # Create mel spectrogram
        self.mel_transform = T.MelSpectrogram(
            n_mels=NUM_MELS,
            n_fft=FFT_BINS,
            win_length=FFT_WIN_LENGTH,
            hop_length=SPEC_HOP_LENGTH
        )
        
        # Create character mapping
        self.char_map = {c: i for i, c in enumerate(CHAR_MAP)}
        
        # Create reverse mapping for decoding
        self.reverse_char_map = {i: c for c, i in self.char_map.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, transcript, _, _, _ = self.dataset[idx]
        # Create mel spectrogram - shape will be (n_mels, time)
        mel_spec = self.mel_transform(waveform).squeeze(0)

        # Convert transcript to character indices
        transcript = transcript.lower()
        transcript_encoded = torch.tensor([self.char_map[c] for c in transcript if c in self.char_map], dtype=torch.long)

        return mel_spec, transcript_encoded

    # Pads spectrograms to max length, keeps transcripts unpadded — necessary for CTC loss
    @staticmethod
    def pad(batch):
        # Unpack batch
        spectrograms, transcripts = zip(*batch)
        
        # First transpose each spectrogram to (time, n_mels)
        spectrograms = [spec.transpose(0, 1) for spec in spectrograms]
        
        # Pad spectrograms to the same length along time dimension (for CTC loss)
        spectrograms = pad_sequence(spectrograms, batch_first=True)
        
        return spectrograms, transcripts

    @staticmethod
    def num_classes():
        return LibriSpeech.NUM_CLASSES

    @staticmethod
    def blank_index():
        return LibriSpeech.BLANK_INDEX

    @staticmethod
    def char_map():
        return LibriSpeech.CHAR_MAP
        
    def get_original_text(self, idx):
        _, _, transcript, _, _, _ = self.dataset[idx]
        return transcript