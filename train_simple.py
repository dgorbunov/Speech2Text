import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enables CPU fallback

# Force CPU for simplicity and reliability
FORCE_CPU = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from librispeech import LibriSpeech
from cnn import SpeechCNN

# Load Dataset
print("Loading dataset...")
train_dataset = LibriSpeech(dataPath="./data", subset="train-clean-100")
print("Dataset loaded.")

# Check a single sample
mel_spec, transcript = train_dataset[0]
print(f"Sample spectrogram shape: {mel_spec.shape}")
print(f"Sample transcript shape: {transcript.shape}")

# Set device
device = torch.device("cpu")  # Force CPU for simplicity
print(f"Using {device}.")

# Define model, optimizer, and loss function
vocab_size = 30  # Adjust based on your character set
model = SpeechCNN(vocab_size)
model.to(device)

# Create a small batch for testing
batch_size = 2  # Very small batch for testing
dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    collate_fn=LibriSpeech.pad, 
    shuffle=True
)

# Get a single batch
print("Getting a batch...")
batch = next(iter(dataloader))
spectrograms, transcripts = batch
transcripts = transcripts[0]
print(f"Batch spectrograms shape: {spectrograms.shape}")
print(f"Batch transcripts shape: {transcripts.shape}")

# Move to device
spectrograms = spectrograms.to(device)
transcripts = transcripts.to(device)

# Forward pass
print("Running forward pass...")
outputs = model(spectrograms)
print(f"Model output shape: {outputs.shape}")

# Compute log probabilities
log_probs = F.log_softmax(outputs, dim=2)
log_probs = log_probs.permute(1, 0, 2)  # (T, B, vocab_size)
print(f"Log probs shape: {log_probs.shape}")

# Get input & target lengths
input_lengths = torch.full((spectrograms.shape[0],), spectrograms.shape[1] // 2, dtype=torch.long)
target_lengths = torch.sum(transcripts != 0, dim=1)
print(f"Input lengths: {input_lengths}")
print(f"Target lengths: {target_lengths}")

# Ensure input lengths are at least as large as target lengths
for i in range(len(input_lengths)):
    if input_lengths[i] < target_lengths[i]:
        input_lengths[i] = target_lengths[i]

# Try CTC loss
print("Computing CTC loss...")
criterion = nn.CTCLoss(blank=0, reduction='mean')
try:
    loss = criterion(log_probs, transcripts, input_lengths, target_lengths)
    print(f"Loss: {loss.item()}")
    print("CTC loss computed successfully!")
except Exception as e:
    print(f"Error in CTC loss: {e}")
    print(f"Shapes - log_probs: {log_probs.shape}, transcripts: {transcripts.shape}")
    print(f"Input lengths: {input_lengths}")
    print(f"Target lengths: {target_lengths}")

print("Test complete!")
