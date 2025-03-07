import os
import torch
import torch.nn.functional as F
import numpy as np
from librispeech import LibriSpeech
from speech_lstm import SpeechLSTM

# Configuration
TEST_DATASET = "test-clean"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 4

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
    num_workers=0
)

# Initialize model
model = SpeechLSTM(num_classes=LibriSpeech.NUM_CLASSES)

# Load checkpoint
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest_checkpoint.pt')

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
else:
    print("No checkpoint found")

model.to(device)
model.eval()

def debug_model():
    print("\nDebugging model predictions...")
    
    # Get a single batch
    batch = next(iter(test_loader))
    spectrograms, transcripts = batch
    
    # Move to device
    spectrograms = spectrograms.to(device)
    
    with torch.no_grad():
        # Get model outputs
        outputs = model(spectrograms)
        
        # Check for NaN/Inf in raw outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("WARNING: NaN or Inf values in raw model outputs!")
        
        # Get probabilities and log probabilities
        probs = F.softmax(outputs, dim=2)
        log_probs = F.log_softmax(outputs, dim=2)
        
        # Check for numerical issues
        print(f"Raw output stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
        print(f"Probability stats: min={probs.min().item():.8f}, max={probs.max().item():.8f}")
        print(f"Log probability stats: min={log_probs.min().item():.4f}, max={log_probs.max().item():.4f}")
        
        # Check for extreme values
        zero_probs = (probs == 0).sum().item()
        small_probs = (probs < 1e-10).sum().item()
        total_elements = probs.numel()
        print(f"Zero probabilities: {zero_probs}/{total_elements} ({zero_probs/total_elements:.2%})")
        print(f"Very small probabilities: {small_probs}/{total_elements} ({small_probs/total_elements:.2%})")
        
        # Analyze predictions for each sample
        for i in range(min(BATCH_SIZE, len(spectrograms))):
            print(f"\nSample {i+1}:")
            
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
            
            # Check if model is predicting only one class
            if len(unique_indices) == 1:
                print(f"WARNING: Model is predicting only one class: {unique_indices[0]}")
                if unique_indices[0] < len(test_dataset.reverse_char_map):
                    print(f"Character: '{test_dataset.reverse_char_map[unique_indices[0]]}'")
            
            # Simple decoding
            decoded = []
            prev_idx = -1
            for idx in indices:
                if idx != LibriSpeech.BLANK_INDEX and idx != prev_idx:
                    if idx < len(test_dataset.reverse_char_map):
                        decoded.append(test_dataset.reverse_char_map[idx])
                prev_idx = idx
            
            prediction = ''.join(decoded)
            print(f"Prediction: '{prediction}'")
            
            # Check for bias in the FC layer
            fc_bias = model.fc.bias.data.cpu().numpy()
            print(f"FC layer bias for blank token (index {LibriSpeech.BLANK_INDEX}): {fc_bias[LibriSpeech.BLANK_INDEX]:.4f}")
            
            # Top 3 biases
            top_bias_indices = np.argsort(fc_bias)[-3:][::-1]
            print("Top 3 biases in FC layer:")
            for idx in top_bias_indices:
                char = test_dataset.reverse_char_map[idx] if idx < len(test_dataset.reverse_char_map) else "UNK"
                print(f"  {char} (index {idx}): {fc_bias[idx]:.4f}")
            
            # Bottom 3 biases
            bottom_bias_indices = np.argsort(fc_bias)[:3]
            print("Bottom 3 biases in FC layer:")
            for idx in bottom_bias_indices:
                char = test_dataset.reverse_char_map[idx] if idx < len(test_dataset.reverse_char_map) else "UNK"
                print(f"  {char} (index {idx}): {fc_bias[idx]:.4f}")

debug_model()
