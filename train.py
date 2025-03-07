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
import numpy as np

# Training configuration
TRAIN_DATASET = "dev-clean"
VAL_DATASET = "dev-other"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-6
GRAD_CLIP_MAX_NORM = 1.0
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

# Initialize bias in the final layer to reduce blank predictions
with torch.no_grad():
    # Set a negative bias for the blank token (index 0)
    model.fc.bias[LibriSpeech.BLANK_INDEX] = -2.0

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
        batch_losses = []

        # Warmup learning rate
        if epoch < WARMUP_EPOCHS:
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            spectrograms, transcripts = batch
            batch_size = spectrograms.size(0)

            # Skip batches with no valid transcripts
            if all(len(t) == 0 for t in transcripts):
                print("Skipping batch with empty transcripts")
                continue

            spectrograms = spectrograms.to(device)

            try:
                if use_amp:
                    with autocast(device_type='cuda'):
                        outputs = model(spectrograms)
                        log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)

                        # Calculate input lengths based on actual output size
                        input_lengths = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long)
                        target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
                        
                        # Check if any target is longer than input (CTC requirement)
                        if (target_lengths > input_lengths).any():
                            print(f"Warning: Some targets longer than inputs in batch {batch_idx}")
                            print(f"Input lengths: {input_lengths}")
                            print(f"Target lengths: {target_lengths}")
                            # Skip this batch
                            continue
                            
                        targets = torch.cat(transcripts)

                        loss = criterion(log_probs, targets, input_lengths, target_lengths)
                        
                        # Check for NaN or Inf loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN/Inf loss detected in batch {batch_idx}. Skipping.")
                            continue

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(spectrograms)
                    log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)

                    # Calculate input lengths based on actual output size
                    input_lengths = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long)
                    target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
                    
                    # Check if any target is longer than input (CTC requirement)
                    if (target_lengths > input_lengths).any():
                        print(f"Warning: Some targets longer than inputs in batch {batch_idx}")
                        print(f"Input lengths: {input_lengths}")
                        print(f"Target lengths: {target_lengths}")
                        # Skip this batch
                        continue
                        
                    targets = torch.cat(transcripts)

                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                    
                    # Check for NaN or Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected in batch {batch_idx}. Skipping.")
                        continue

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                    optimizer.step()

                # Track batch loss
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                
                # Print occasional batch stats
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}: Loss = {batch_loss:.4f}")
                
                # Clip loss value for averaging (in case some batches still have high loss)
                clipped_loss = min(batch_loss, 100.0)
                total_loss += clipped_loss * batch_size
                total_samples += batch_size
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate average loss, handling edge cases
        if total_samples > 0:
            train_loss = total_loss / total_samples
        else:
            train_loss = float('inf')
            print("Warning: No valid samples in training epoch")
        
        # Calculate median loss (more robust to outliers)
        median_loss = np.median(batch_losses) if batch_losses else float('inf')
        print(f"Median batch loss: {median_loss:.4f}")
        
        # Use a more robust validation approach
        try:
            val_loss = validate(model, val_loader, criterion, device)
        except Exception as e:
            print(f"Error during validation: {e}")
            val_loss = float('inf')

        # Update statistics
        stats['train_losses'].append(float(train_loss))
        stats['val_losses'].append(float(val_loss))
        stats['learning_rates'].append(float(optimizer.param_groups[0]['lr']))

        # Only save checkpoint if loss is valid
        if val_loss < best_val_loss and not np.isinf(val_loss) and not np.isnan(val_loss):
            best_val_loss = val_loss
            stats['best_val_loss'] = float(best_val_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'stats': stats
            }, is_best=True)

        if train_loss < stats['best_train_loss'] and not np.isinf(train_loss) and not np.isnan(train_loss):
            stats['best_train_loss'] = float(train_loss)

        # Only update scheduler if loss is valid
        if not np.isinf(val_loss) and not np.isnan(val_loss):
            scheduler.step(val_loss)
        else:
            print("Skipping scheduler step due to invalid loss")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s")

        # Save checkpoint every epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'stats': stats
        }, is_best=False)

    print(f"\nTraining completed")
    print(f"Best training loss: {stats['best_train_loss']:.4f}")
    print(f"Best validation loss: {stats['best_val_loss']:.4f}")

    with open(f"{CHECKPOINT_DIR}/training_stats.json", 'w') as f:
        json.dump(stats, f)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    batch_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            spectrograms, transcripts = batch
            batch_size = spectrograms.size(0)

            # Skip batches with no valid transcripts
            if all(len(t) == 0 for t in transcripts):
                continue

            spectrograms = spectrograms.to(device)
            
            try:
                outputs = model(spectrograms)
                log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)

                # Calculate input lengths based on actual output size
                input_lengths = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long)
                target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
                
                # Check if any target is longer than input (CTC requirement)
                if (target_lengths > input_lengths).any():
                    print(f"Warning: Some targets longer than inputs in validation batch {batch_idx}")
                    # Skip this batch
                    continue
                    
                targets = torch.cat(transcripts)

                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
                # Check for NaN or Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected in validation batch {batch_idx}. Skipping.")
                    continue
                
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                
                # Clip loss value for averaging
                clipped_loss = min(batch_loss, 100.0)
                total_loss += clipped_loss * batch_size
                total_samples += batch_size
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    # Calculate average loss, handling edge cases
    if total_samples > 0:
        avg_loss = total_loss / total_samples
    else:
        avg_loss = float('inf')
        print("Warning: No valid samples in validation")
    
    # Calculate median loss (more robust to outliers)
    median_loss = np.median(batch_losses) if batch_losses else float('inf')
    print(f"Validation median batch loss: {median_loss:.4f}")
    
    return avg_loss

def save_checkpoint(state, is_best):
    torch.save(state, f"{CHECKPOINT_DIR}/latest_checkpoint.pt")
    if is_best:
        torch.save(state, f"{CHECKPOINT_DIR}/best_model.pt")

if __name__ == "__main__":
    train()
