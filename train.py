import os

# Enable CPU fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Disable upper limit for memory allocations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from tqdm import tqdm
from librispeech import LibriSpeech
from speech_lstm import SpeechLSTM
import numpy as np
import random

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed()

TRAIN_DATASET = "dev-clean"
VAL_DATASET = "dev-other"  
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_MAX_NORM = 5.0
WARMUP_EPOCHS = 2  # Number of epochs for learning rate warmup

# Create checkpoint directory if it doesn't exist
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# Training Loop
def train(model, dataloader, val_dataloader, optimizer, criterion, device, epochs, start_epoch=0, best_loss=float('inf')):
    model.to(device)
    
    stats = {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "times": [],
        "best_train_loss": best_loss,
        "best_val_loss": float('inf')
    }
    
    input_lengths_cache = {}
    scheduler = None
    
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress"):
        model.train()  
        epoch_start_time = time.time()
        total_loss = 0.0
        total_samples = 0  
        batch_count = 0
        
        # Apply learning rate warmup
        if epoch < WARMUP_EPOCHS:
            warmup_factor = (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE * warmup_factor
                print(f"Warmup LR: {param_group['lr']:.6f}")
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            spectrograms, transcripts = batch
            batch_size = spectrograms.size(0)
            
            spectrograms = spectrograms.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(spectrograms)
            
            log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)  
            
            seq_length = outputs.size(1)
            if (batch_size, seq_length) in input_lengths_cache:
                input_lengths = input_lengths_cache[(batch_size, seq_length)]
            else:
                input_lengths = torch.full(size=(batch_size,), fill_value=seq_length, dtype=torch.long)
                input_lengths_cache[(batch_size, seq_length)] = input_lengths
                
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
            
            targets = torch.cat(transcripts)
            
            log_probs = log_probs.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            try:
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
                # Print sample prediction every 50 batches
                if batch_count % 50 == 0:
                    with torch.no_grad():
                        # Get sample prediction
                        sample_output = outputs[0:1]
                        probs = F.softmax(sample_output, dim=2)
                        indices = torch.argmax(probs, dim=2)[0].cpu().numpy()
                        
                        # Simple decoding
                        decoded = []
                        prev_idx = -1
                        for idx in indices:
                            if idx != LibriSpeech.BLANK_INDEX and idx != prev_idx:
                                if idx < len(dataloader.dataset.reverse_char_map):
                                    decoded.append(dataloader.dataset.reverse_char_map[idx])
                            prev_idx = idx
                        
                        prediction = ''.join(decoded)
                        print(f"\nSample prediction: '{prediction}'")
                        
                        # Print unique indices to debug
                        unique_indices = np.unique(indices)
                        print(f"Unique indices: {unique_indices}")
                
                if not torch.isfinite(loss):
                    print("Warning: non-finite loss, skipping batch")
                    continue
                    
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                
                optimizer.step()
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                batch_count += 1
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
            
        epoch_time = time.time() - epoch_start_time
        train_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        val_loss = validate(model, val_dataloader, criterion, device, input_lengths_cache)
        
        # Update learning rate based on validation loss
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2, 
                verbose=True
            )
        scheduler.step(val_loss)
        
        # Save statistics
        stats["epochs"].append(epoch + 1)
        stats["train_losses"].append(train_loss)
        stats["val_losses"].append(val_loss)
        stats["learning_rates"].append(optimizer.param_groups[0]['lr'])
        stats["times"].append(epoch_time)
        
        is_best = val_loss < stats["best_val_loss"]
        if is_best:
            stats["best_val_loss"] = val_loss
            
        if train_loss < stats["best_train_loss"]:
            stats["best_train_loss"] = train_loss
        
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'stats': stats
        }, is_best)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {epoch_time:.2f}s")
    
    tqdm.write(f"\nTraining completed in {sum(stats['times']):.2f} seconds")
    tqdm.write(f"Best training loss: {stats['best_train_loss']:.4f}")
    tqdm.write(f"Best validation loss: {stats['best_val_loss']:.4f}")
    
    with open(f"{CHECKPOINT_DIR}/training_stats.json", 'w') as f:
        json.dump(stats, f)
    
    return stats

def validate(model, dataloader, criterion, device, input_lengths_cache=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():  
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            spectrograms, transcripts = batch
            batch_size = spectrograms.size(0)
            
            spectrograms = spectrograms.to(device)
            
            outputs = model(spectrograms)
            
            log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)
            
            seq_length = outputs.size(1)
            if input_lengths_cache and (batch_size, seq_length) in input_lengths_cache:
                input_lengths = input_lengths_cache[(batch_size, seq_length)]
            else:
                input_lengths = torch.full(size=(batch_size,), fill_value=seq_length, dtype=torch.long)
                if input_lengths_cache is not None:
                    input_lengths_cache[(batch_size, seq_length)] = input_lengths
                    
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
            
            targets = torch.cat(transcripts)
            
            # Make sure everything is on the same device
            log_probs = log_probs.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            try:
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                    
                total_loss += loss.item() * batch_size  
                total_samples += batch_size
            except Exception as e:
                print(f"Error in validation: {e}")
                continue
    
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss

def save_checkpoint(state, is_best):
    torch.save(state, f"{CHECKPOINT_DIR}/latest_checkpoint.pt")
    if is_best:
        torch.save(state, f"{CHECKPOINT_DIR}/best_model.pt")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():  
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Create datasets
train_dataset = LibriSpeech(DATA_DIR, TRAIN_DATASET)
val_dataset = LibriSpeech(DATA_DIR, VAL_DATASET)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=LibriSpeech.pad
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=LibriSpeech.pad
)

# Initialize model and optimizer
model = SpeechLSTM(num_classes=LibriSpeech.NUM_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Use standard PyTorch CTC loss
criterion = nn.CTCLoss(blank=LibriSpeech.BLANK_INDEX, reduction='mean')

# Load checkpoint if available
start_epoch, best_loss = 0, float('inf')
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')
if os.path.exists(checkpoint_path):
    try:
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('train_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            print("Model architecture has changed. Starting fresh training.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting fresh training.")
else:
    print("No checkpoint found. Starting fresh training.")

print(f"\nStarting training:")
print(f"- Epochs: {NUM_EPOCHS} (starting from {start_epoch})")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Learning rate: {LEARNING_RATE}")
print(f"- Device: {device}")
print(f"- Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

train(model, train_loader, val_loader, optimizer, criterion, device, epochs=NUM_EPOCHS, start_epoch=start_epoch, best_loss=best_loss)
