import torch
from torch.utils.data import DataLoader
from librispeech import LibriSpeech

# Load the dataset
print("Loading dataset...")
dataset = LibriSpeech()
print("Dataset loaded.")

# Check a single sample
mel_spec, transcript = dataset[0]
print(f"Single sample:")
print(f"- Mel spectrogram shape: {mel_spec.shape}")
print(f"- Transcript shape: {transcript.shape}")
print(f"- Transcript: {transcript}")

# Test the padding function with a small batch
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LibriSpeech.pad, shuffle=True)

# Get a batch
batch = next(iter(dataloader))
spectrograms, transcripts = batch

print("\nBatch data:")
print(f"- Batch size: {len(spectrograms)}")
print(f"- Padded spectrograms shape: {spectrograms.shape}")
print(f"- Padded transcripts shape: {transcripts.shape}")

# Check individual items in the batch
print("\nIndividual spectrogram shapes in the batch:")
for i in range(batch_size):
    print(f"- Item {i}: {spectrograms[i].shape}")

print("\nIndividual transcript lengths in the batch:")
for i in range(batch_size):
    non_zero = torch.sum(transcripts[i] != 0).item()
    print(f"- Item {i}: {non_zero} (total padded length: {transcripts[i].shape[0]})")
