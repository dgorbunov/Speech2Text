import os

# MPS optimizations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE_ASYNC_GPU_COPIES"] = "1"

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from librispeech import LibriSpeech
import torch.nn.functional as F
from speech_lstm import SpeechLSTM
import json

# Training configuration
TRAIN_DATASET = "dev-clean"
VAL_DATASET = "dev-other"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_MAX_NORM = 5.0
WARMUP_EPOCHS = 3

# Create checkpoint directory
Path(CHECKPOINT_DIR).mkdir(exist_ok=True, parents=True)

# Set device
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA - Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
else:
    print("Using CPU - No GPU detected")

# Initialize model
model = SpeechLSTM()
model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create datasets
train_dataset = LibriSpeech(dataPath="./data", subset=TRAIN_DATASET)
val_dataset = LibriSpeech(dataPath="./data", subset=VAL_DATASET)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    num_workers=4,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=val_dataset.collate_fn,
    num_workers=4,
    pin_memory=True
)

# Initialize optimizer and criterion
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
criterion = nn.CTCLoss(blank=LibriSpeech.BLANK_INDEX, reduction='mean')

# Mixed precision training
use_amp = device.type == 'cuda'
if use_amp:
    from torch.amp import GradScaler, autocast
    scaler = GradScaler(device_type='cuda')
    print("Using mixed precision training with CUDA")
else:
    print("Mixed precision training disabled (not using CUDA)")

def train():
    best_val_loss = float('inf')
    stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'best_train_loss': float('inf'),
        'best_val_loss': float('inf')
    }

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0

        # Warmup learning rate
        if epoch < WARMUP_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            spectrograms, transcripts = batch
            batch_size = spectrograms.size(0)

            spectrograms = spectrograms.to(device)

            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(spectrograms)
                    log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)

                    input_lengths = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long)
                    target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
                    targets = torch.cat(transcripts)

                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(spectrograms)
                log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)

                input_lengths = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long)
                target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
                targets = torch.cat(transcripts)

                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples
        val_loss = validate(model, val_loader, criterion, device)

        # Update statistics
        stats['train_losses'].append(train_loss)
        stats['val_losses'].append(val_loss)
        stats['learning_rates'].append(optimizer.param_groups[0]['lr'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stats['best_val_loss'] = best_val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'stats': stats
            }, is_best=True)

        if train_loss < stats['best_train_loss']:
            stats['best_train_loss'] = train_loss

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s")

    print(f"\nTraining completed")
    print(f"Best training loss: {stats['best_train_loss']:.4f}")
    print(f"Best validation loss: {stats['best_val_loss']:.4f}")

    with open(f"{CHECKPOINT_DIR}/training_stats.json", 'w') as f:
        json.dump(stats, f)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            spectrograms, transcripts = batch
            batch_size = spectrograms.size(0)

            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)

            input_lengths = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
            targets = torch.cat(transcripts)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples

def save_checkpoint(state, is_best):
    torch.save(state, f"{CHECKPOINT_DIR}/latest_checkpoint.pt")
    if is_best:
        torch.save(state, f"{CHECKPOINT_DIR}/best_model.pt")

if __name__ == "__main__":
    train()
