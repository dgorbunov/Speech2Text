import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enables CPU fallback
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable upper limit for memory allocations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from librispeech import LibriSpeech
from cnn import SpeechCNN

# Default settings
NUM_EPOCHS = 10
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 2  # Reduced batch size to prevent memory issues

# Create checkpoint directory if it doesn't exist
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)

# Training Loop
def train(model, dataloader, optimizer, criterion, device, epochs, start_epoch=0, best_loss=float('inf')):
    model.to(device)
    
    # Initialize stats dictionary
    stats = {
        "epochs": [],
        "losses": [],
        "times": [],
        "best_loss": best_loss
    }
    
    # Main training loop with tqdm for progress tracking
    for epoch in tqdm(range(start_epoch, epochs), desc="Training Progress"):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0.0
        batch_count = 0
        
        # Process batches
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            spectrograms, transcripts = batch
            
            # Move data to device
            spectrograms = spectrograms.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(spectrograms)
            
            # Prepare for CTC loss
            log_probs = F.log_softmax(outputs, dim=2).transpose(0, 1)  # (time, batch, classes)
            input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
            
            # Flatten targets into a 1D tensor - transcripts are already tensors
            targets = torch.cat(transcripts)
            
            # Move tensors to device
            log_probs = log_probs.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Calculate loss
            try:
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            except Exception as e:
                tqdm.write(f"Error in CTC loss: {e}")
                continue
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            batch_count += 1
            
            # Clean up memory
            del spectrograms, outputs, log_probs, loss, targets, input_lengths, target_lengths
            if device.type == 'mps':
                torch.mps.empty_cache()
                
            # Add a small delay to allow memory to be freed
            if device.type == 'mps' and batch_count % 5 == 0:
                time.sleep(0.1)
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        epoch_time = time.time() - epoch_start_time
        
        # Update tqdm description with loss
        tqdm.write(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
        
        # Update stats
        stats["epochs"].append(epoch+1)
        stats["losses"].append(avg_loss)
        stats["times"].append(epoch_time)
        
        # Save checkpoint
        is_best = avg_loss < stats["best_loss"]
        if is_best:
            stats["best_loss"] = avg_loss
        
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'stats': stats
        }, is_best)
    
    # Print training summary
    total_time = time.time() - epoch_start_time
    tqdm.write(f"\nTraining completed in {total_time:.2f} seconds")
    tqdm.write(f"Best loss: {stats['best_loss']:.4f}")
    
    # Save final stats
    with open(f"{CHECKPOINT_DIR}/training_stats.json", 'w') as f:
        json.dump(stats, f)
    
    return stats

def save_checkpoint(state, is_best):
    torch.save(state, f"{CHECKPOINT_DIR}/latest_checkpoint.pt")
    if is_best:
        torch.save(state, f"{CHECKPOINT_DIR}/best_model.pt")

def load_checkpoint():
    checkpoint_path = f"{CHECKPOINT_DIR}/latest_checkpoint.pt"
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found, starting from scratch")
        return 0, float('inf')
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Return epoch and best loss
    return checkpoint['epoch'], checkpoint['stats']['best_loss']

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a speech-to-text model')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    args = parser.parse_args()
    
    # Update settings from command line
    FORCE_CPU = args.force_cpu
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    
    # Set device
    if FORCE_CPU:
        device = torch.device("cpu")
        print("Using CPU (forced)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():  
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load dataset
    train_dataset = LibriSpeech(dataPath="./data", subset="train-clean-100")
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.pad
    )
    
    # Create model
    model = SpeechCNN(num_classes=30)  # 30 characters in our vocabulary
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    
    # Load checkpoint if exists
    start_epoch, best_loss = load_checkpoint()
    
    # Print training info
    print(f"\nStarting training:")
    print(f"- Epochs: {NUM_EPOCHS} (starting from {start_epoch})")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Samples: {len(train_dataset)}")
    print(f"- Device: {device}")
    
    # Train the model
    train(model, train_loader, optimizer, criterion, device, epochs=NUM_EPOCHS, start_epoch=start_epoch, best_loss=best_loss)