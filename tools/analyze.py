import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from librispeech import LibriSpeech
from speech_lstm import SpeechLSTM

# Configuration
TEST_DATASET = "test-clean"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./analysis"
BATCH_SIZE = 8 

# Create results directory
Path(RESULTS_DIR).mkdir(exist_ok=True)

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load test dataset
test_dataset = LibriSpeech(dataPath="./data", subset=TEST_DATASET)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=test_dataset.collate_fn,
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

# Initialize model
model = SpeechLSTM(num_classes=LibriSpeech.NUM_CLASSES)

# Load the best checkpoint
best_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')

checkpoint_path = best_checkpoint_path if os.path.exists(best_checkpoint_path) else latest_checkpoint_path

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Print the reported training and validation losses
    if 'stats' in checkpoint:
        stats = checkpoint['stats']
        print(f"Best training loss: {stats.get('best_train_loss', 'N/A')}")
        print(f"Best validation loss: {stats.get('best_val_loss', 'N/A')}")
else:
    print("No checkpoint found. Using untrained model.")

# Move model to device
model.to(device)
model.eval()

def analyze_model_outputs():
    print("\nAnalyzing model outputs...")
    
    # Get a single batch for analysis
    batch = next(iter(test_loader))
    spectrograms, transcripts = batch
    batch_size = spectrograms.size(0)
    
    spectrograms = spectrograms.to(device)
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(spectrograms)
        
        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=2)
        
        # Get log probabilities
        log_probs = F.log_softmax(outputs, dim=2)
        
        # Analyze each sample in the batch
        for i in range(min(batch_size, 3)):  # Analyze up to 3 samples
            print(f"\nSample {i+1} Analysis:")
            
            # Get ground truth
            ground_truth = ''.join([test_dataset.reverse_char_map[idx.item()] for idx in transcripts[i]])
            print(f"Ground Truth: '{ground_truth}'")
            
            # Get prediction
            sample_probs = probs[i]
            indices = torch.argmax(sample_probs, dim=1).cpu().numpy()
            
            # Get unique indices and their counts
            unique_indices, counts = np.unique(indices, return_counts=True)
            print(f"Unique predicted indices: {unique_indices}")
            print(f"Counts: {counts}")
            
            # Simple decoding (merge repeated and remove blanks)
            decoded = []
            prev_idx = -1
            for idx in indices:
                if idx != LibriSpeech.BLANK_INDEX and idx != prev_idx:
                    if idx < len(test_dataset.reverse_char_map):
                        decoded.append(test_dataset.reverse_char_map[idx])
                prev_idx = idx
            
            prediction = ''.join(decoded)
            print(f"Prediction: '{prediction}'")
            
            # Analyze probability distribution
            avg_probs = torch.mean(sample_probs, dim=0)
            top5_classes = torch.topk(avg_probs, 5)
            
            print("Top 5 average probabilities:")
            for j in range(5):
                idx = top5_classes.indices[j].item()
                prob = top5_classes.values[j].item()
                char = test_dataset.reverse_char_map[idx] if idx < len(test_dataset.reverse_char_map) else "UNK"
                print(f"  {char}: {prob:.6f}")
            
            # Check for numerical issues
            zero_probs = (sample_probs == 0).sum().item()
            small_probs = (sample_probs < 1e-10).sum().item()
            total_elements = sample_probs.numel()
            
            print(f"Zero probabilities: {zero_probs}/{total_elements} ({zero_probs/total_elements:.2%})")
            print(f"Very small probabilities: {small_probs}/{total_elements} ({small_probs/total_elements:.2%})")
            
            # Plot probability distribution over time for a few characters
            plt.figure(figsize=(12, 6))
            
            # Plot for blank character
            plt.plot(sample_probs[:, LibriSpeech.BLANK_INDEX].cpu().numpy(), 
                     label=f"Blank ({LibriSpeech.BLANK_INDEX})")
            
            # Plot for 'e' (usually common)
            e_idx = test_dataset.char_map.get('e', -1)
            if e_idx >= 0:
                plt.plot(sample_probs[:, e_idx].cpu().numpy(), label=f"e ({e_idx})")
            
            # Plot for space
            space_idx = test_dataset.char_map.get(' ', -1)
            if space_idx >= 0:
                plt.plot(sample_probs[:, space_idx].cpu().numpy(), label=f"space ({space_idx})")
            
            # Plot for a few more characters
            for char in ['a', 't', 's']:
                char_idx = test_dataset.char_map.get(char, -1)
                if char_idx >= 0:
                    plt.plot(sample_probs[:, char_idx].cpu().numpy(), label=f"{char} ({char_idx})")
            
            plt.title(f"Character Probabilities Over Time (Sample {i+1})")
            plt.xlabel("Time Steps")
            plt.ylabel("Probability")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{RESULTS_DIR}/sample_{i+1}_char_probs.png")
            
            # Plot heatmap of probabilities over time
            plt.figure(figsize=(15, 8))
            plt.imshow(sample_probs.cpu().numpy().T, aspect='auto', cmap='viridis')
            plt.colorbar(label='Probability')
            plt.title(f"Character Probabilities Heatmap (Sample {i+1})")
            plt.xlabel("Time Steps")
            plt.ylabel("Character Index")
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/sample_{i+1}_heatmap.png")
            
        # Analyze overall model behavior
        print("\nOverall Model Analysis:")
        
        # Check for bias towards specific characters
        avg_batch_probs = torch.mean(probs.view(-1, probs.size(-1)), dim=0)
        top5_overall = torch.topk(avg_batch_probs, 5)
        
        print("Top 5 characters across batch:")
        for j in range(5):
            idx = top5_overall.indices[j].item()
            prob = top5_overall.values[j].item()
            char = test_dataset.reverse_char_map[idx] if idx < len(test_dataset.reverse_char_map) else "UNK"
            print(f"  {char}: {prob:.6f}")
        
        # Plot overall probability distribution
        plt.figure(figsize=(10, 6))
        char_indices = range(len(test_dataset.char_map))
        plt.bar(char_indices, avg_batch_probs.cpu().numpy())
        plt.title("Average Probability Distribution Across Characters")
        plt.xlabel("Character Index")
        plt.ylabel("Average Probability")
        plt.xticks(char_indices, [test_dataset.reverse_char_map.get(i, "UNK") for i in char_indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/overall_char_distribution.png")
        
        print(f"\nAnalysis complete. Results saved to {RESULTS_DIR}/")

analyze_model_outputs()
