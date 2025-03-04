import torch
import torchaudio.transforms as T
from torchaudio.datasets import LIBRISPEECH
from torch.nn.utils.rnn import pad_sequence

# Reduced number of mel bands to avoid the warning
NUM_MELS = 80

class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, dataPath, subset):
        self.dataset = LIBRISPEECH(root=dataPath, url=subset, download=True)
        # Increase n_fft to get more frequency bins and use fewer mel bands
        self.mel_transform = T.MelSpectrogram(
            n_mels=NUM_MELS,
            n_fft=1024,  # Increased from default 400
            win_length=400,
            hop_length=160
        )
        # Create character mapping
        self.char_map = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz '")}  # 1-based index

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

    @staticmethod
    def pad(batch):
        # Unpack batch into separate lists for spectrograms and transcripts
        spectrograms, transcripts = zip(*batch)
        
        # First transpose each spectrogram to (time, n_mels)
        spectrograms = [spec.transpose(0, 1) for spec in spectrograms]
        
        # Pad spectrograms to the same length along time dimension
        spectrograms = pad_sequence(spectrograms, batch_first=True)
        
        # Return transcripts as a list of tensors (don't pad them for CTC loss)
        # Each tensor should already be a torch.Tensor from __getitem__
        
        return spectrograms, transcripts