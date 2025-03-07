import os

# MPS optimizations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_ENABLE_ASYNC_GPU_COPIES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from jiwer import wer, cer
import matplotlib.pyplot as plt
from librispeech import LibriSpeech
from speech_lstm import SpeechLSTM

# Test configuration
NUM_TEST_SAMPLES = 5
BATCH_SIZE = 64
RESULTS_DIR = "./results"
CHECKPOINT_DIR = "./checkpoints"
TEST_DATASET = "test-clean"

def calculate_cer(reference, hypothesis):
    try:
        # Handle empty strings by adding a placeholder character
        ref = reference if reference.strip() else "empty"
        hyp = hypothesis if hypothesis.strip() else "empty"
        
        # Calculate CER using jiwer
        return cer(ref, hyp)
    except Exception as e:
        print(f"Error calculating CER: {e}")
        return 1.0  # Return worst case if calculation fails

def greedy_decode(output, reverse_char_map, blank_idx=LibriSpeech.BLANK_INDEX):
    # Apply softmax to get probabilities
    probs = F.softmax(output, dim=2)
    
    # Get most likely character indices
    indices = torch.argmax(probs, dim=2)[0].cpu().numpy()
    
    # Simple stats for debugging
    print(f"Unique prediction indices: {np.unique(indices)}")
    
    # Count occurrences of each index for debugging
    unique, counts = np.unique(indices, return_counts=True)
    index_counts = dict(zip(unique, counts))
    print(f"Index counts: {index_counts}")
    
    # Apply CTC decoding rules (merge repeated characters and remove blanks)
    decoded_sequence = []
    prev_idx = -1  # Different from any valid index
    
    for idx in indices:
        # Skip blanks and repeated characters (CTC rules)
        if idx != blank_idx and idx != prev_idx:
            if idx < len(reverse_char_map):
                decoded_sequence.append(reverse_char_map[idx])
        prev_idx = idx
    
    result = ''.join(decoded_sequence)
    return result

def test_model(model, dataset, test_loader, device, num_samples=NUM_TEST_SAMPLES, save_dir=RESULTS_DIR):
    Path(save_dir).mkdir(exist_ok=True)
    
    model.to(device)
    model.eval()
    
    # Create reverse character map for decoding
    reverse_char_map = dataset.reverse_char_map
    
    # Tracking metrics
    total_cer = 0.0
    total_wer = 0.0
    sample_results = []
    
    print(f"\nTesting model on {num_samples} samples from {TEST_DATASET}...")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Test samples: {len(dataset)}")
    print(f"- Device: {device.type}")
    print(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model's final layer bias to check blank token bias
    print(f"- Final layer bias for blank token: {model.fc.bias[LibriSpeech.BLANK_INDEX].item():.4f}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            spectrograms, transcripts = batch
            batch_size = spectrograms.size(0)
            
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            
            # Check output statistics
            if batch_idx == 0:
                print(f"\nOutput stats for first batch:")
                print(f"- Shape: {outputs.shape}")
                print(f"- Min: {outputs.min().item():.4f}")
                print(f"- Max: {outputs.max().item():.4f}")
                print(f"- Mean: {outputs.mean().item():.4f}")
                print(f"- Std: {outputs.std().item():.4f}")
                
                # Check class distribution after softmax
                softmax_probs = F.softmax(outputs[0], dim=1)
                class_means = softmax_probs.mean(dim=0).cpu().numpy()
                top5_classes = np.argsort(class_means)[-5:]
                print(f"- Top 5 average class probabilities:")
                for cls in reversed(top5_classes):
                    print(f"  Class {cls} ({reverse_char_map[cls]}): {class_means[cls]:.6f}")
            
            # Decode each sample in the batch
            for i in range(batch_size):
                sample_output = outputs[i:i+1]  # Keep batch dimension
                ground_truth = ''.join([dataset.reverse_char_map[idx.item()] for idx in transcripts[i]])
                
                # Decode prediction
                predicted_text = greedy_decode(sample_output, reverse_char_map)
                
                # Calculate Character Error Rate
                error_rate = calculate_cer(ground_truth.lower(), predicted_text.lower())
                total_cer += error_rate
                
                # Calculate Word Error Rate using jiwer
                try:
                    # Convert to words for WER calculation
                    # Handle empty strings by adding a placeholder word
                    gt_words = ground_truth.lower() if ground_truth.strip() else "empty"
                    pred_words = predicted_text.lower() if predicted_text.strip() else "empty"
                    
                    # Calculate WER
                    word_error_rate = wer(gt_words, pred_words)
                    total_wer += word_error_rate
                except Exception as e:
                    print(f"Error calculating WER: {e}")
                    word_error_rate = 1.0
                    total_wer += word_error_rate
                
                # Store results for first num_samples
                if len(sample_results) < num_samples:
                    sample_results.append({
                        'ground_truth': ground_truth,
                        'prediction': predicted_text,
                        'cer': error_rate,
                        'wer': word_error_rate
                    })
                    
            # Stop after processing enough samples
            if len(sample_results) >= num_samples:
                break
    
    # Print test results
    print("\nTest Results:\n")
    for i, result in enumerate(sample_results):
        print(f"Sample {i+1}:")
        print(f"Ground Truth: {result['ground_truth']}")
        print(f"Prediction: {result['prediction']}")
        print(f"CER: {result['cer']:.4f}")
        print(f"WER: {result['wer']:.4f}\n")
    
    # Calculate average metrics
    avg_cer = total_cer / len(sample_results) if sample_results else 1.0
    avg_wer = total_wer / len(sample_results) if sample_results else 1.0
    
    # Save results to file
    with open(f"{save_dir}/results.txt", 'w') as f:
        f.write(f"Test Results on {TEST_DATASET}\n")
        f.write(f"Average CER: {avg_cer:.4f}\n")
        f.write(f"Average WER: {avg_wer:.4f}\n\n")
        
        for i, result in enumerate(sample_results):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Ground Truth: {result['ground_truth']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"CER: {result['cer']:.4f}\n")
            f.write(f"WER: {result['wer']:.4f}\n\n")
    
    print(f"Test completed. Results saved to {save_dir}/")
    print(f"Average Character Error Rate: {avg_cer:.4f}")
    print(f"Average Word Error Rate: {avg_wer:.4f}")
    
    return avg_cer, avg_wer, sample_results

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
model = SpeechLSTM()

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

# Test the model
test_model(model, test_dataset, test_loader, device)
