import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from jiwer import wer, cer
from pathlib import Path
from librispeech import LibriSpeech
from speech_lstm import SpeechLSTM
from transformer import SpeechTransformer

# Configuration
TEST_DATASET = "test-clean"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
BATCH_SIZE = 32
NUM_TEST_SAMPLES = 10
USE_TRANSFORMER = False

# Create results directory if it doesn't exist
Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)

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
if USE_TRANSFORMER:
    model = SpeechTransformer(num_classes=LibriSpeech.NUM_CLASSES)
else:
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
else:
    print("No checkpoint found. Using untrained model.")

# Move model to device
model.to(device)
model.eval()

# Testing function
def test_model():
    print(f"\nTesting model on {NUM_TEST_SAMPLES} samples from {TEST_DATASET}...")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Device: {device}")
    print(f"- Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    results = []
    total_cer = 0.0
    total_wer = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            spectrograms, transcripts = batch
            spectrograms = spectrograms.to(device)
            
            # Forward pass
            outputs = model(spectrograms)
            
            # Process each sample in the batch
            for i in range(min(spectrograms.size(0), NUM_TEST_SAMPLES - len(results))):
                # Get ground truth
                ground_truth = ''.join([test_dataset.reverse_char_map[idx.item()] for idx in transcripts[i]])
                
                # Get prediction using standard CTC decoding
                output = outputs[i:i+1]  # Keep batch dimension
                
                # Apply temperature scaling to make predictions less confident
                temperature = 1.5  # Higher values = more uniform distribution
                scaled_output = output / temperature
                
                probs = F.softmax(scaled_output, dim=2)
                
                # Debug: Print probability distribution for first few frames
                print("\nDebug - Probability distribution for first 3 frames:")
                top_probs, top_indices = torch.topk(probs[0, :3], k=5, dim=1)
                for frame_idx in range(min(3, probs.shape[1])):
                    print(f"Frame {frame_idx}:")
                    for prob_idx in range(5):
                        char_idx = top_indices[frame_idx, prob_idx].item()
                        char = test_dataset.reverse_char_map[char_idx] if char_idx < len(test_dataset.reverse_char_map) else "?"
                        print(f"  {char} (index {char_idx}): {top_probs[frame_idx, prob_idx].item():.4f}")
                
                # Debug: Print unique indices and their counts
                indices = torch.argmax(probs, dim=2)[0].cpu().numpy()
                unique_indices, counts = np.unique(indices, return_counts=True)
                print("\nDebug - Unique prediction indices and counts:")
                for idx, count in zip(unique_indices, counts):
                    char = test_dataset.reverse_char_map[idx] if idx < len(test_dataset.reverse_char_map) else "?"
                    print(f"  Index {idx} ({char}): {count} occurrences")
                
                # Standard CTC decoding (remove blanks and duplicates)
                decoded = []
                prev_idx = -1
                
                # Get indices with temperature scaling applied
                indices = torch.argmax(probs, dim=2)[0].cpu().numpy()
                
                # Apply a more aggressive decoding strategy
                # 1. Lower the threshold for accepting non-blank tokens
                # 2. Consider top-k predictions instead of just the argmax
                
                # First pass: use argmax but with a bias against blanks
                for t in range(len(indices)):
                    idx = indices[t]
                    
                    # If it's a blank, check if any non-blank has reasonable probability
                    if idx == LibriSpeech.BLANK_INDEX:
                        # Get top 3 predictions for this timestep
                        top_probs, top_indices = torch.topk(probs[0, t], k=3, dim=0)
                        
                        # If any non-blank has probability > 0.2, use it instead
                        for k in range(len(top_indices)):
                            if top_indices[k].item() != LibriSpeech.BLANK_INDEX and top_probs[k].item() > 0.2:
                                idx = top_indices[k].item()
                                break
                    
                    # Apply CTC rules (no duplicates, no blanks)
                    if idx != LibriSpeech.BLANK_INDEX and idx != prev_idx:
                        if idx < len(test_dataset.reverse_char_map):
                            decoded.append(test_dataset.reverse_char_map[idx])
                    
                    prev_idx = idx
                
                prediction = ''.join(decoded)
                
                # Calculate metrics
                sample_cer = cer(ground_truth.lower(), prediction.lower())
                sample_wer = wer(ground_truth.lower(), prediction.lower())
                
                total_cer += sample_cer
                total_wer += sample_wer
                
                # Store result
                results.append({
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'cer': sample_cer,
                    'wer': sample_wer
                })
                
            # Stop after collecting enough samples
            if len(results) >= NUM_TEST_SAMPLES:
                break
    
    # Calculate average metrics
    avg_cer = total_cer / len(results) if results else 1.0
    avg_wer = total_wer / len(results) if results else 1.0
    
    # Print results
    print("\nTest Results:\n")
    for i, result in enumerate(results):
        print(f"Sample {i+1}:")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Prediction: {result['prediction']}")
        print(f"CER: {result['cer']:.4f}")
        print(f"WER: {result['wer']:.4f}\n")
    
    print(f"Average Character Error Rate: {avg_cer:.4f}")
    print(f"Average Word Error Rate: {avg_wer:.4f}")
    
    # Save results to file
    with open(f"{RESULTS_DIR}/test_results.txt", 'w') as f:
        f.write(f"Test Results on {TEST_DATASET}\n")
        f.write(f"Average CER: {avg_cer:.4f}\n")
        f.write(f"Average WER: {avg_wer:.4f}\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"CER: {result['cer']:.4f}\n")
            f.write(f"WER: {result['wer']:.4f}\n\n")
    
    print(f"Results saved to {RESULTS_DIR}/test_results.txt")
    
    return avg_cer, avg_wer, results

test_model()
