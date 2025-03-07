import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speech_lstm import SpeechLSTM
from librispeech import LibriSpeech

def debug_model():
    print("Loading a small batch of data...")
    dataset = LibriSpeech(dataPath="../data", subset="dev-clean")
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = SpeechLSTM()
    model.to(device)
    model.train()
    
    criterion = nn.CTCLoss(blank=LibriSpeech.BLANK_INDEX, reduction='mean')
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx > 0:
            break
            
        spectrograms, transcripts = batch
        batch_size = spectrograms.size(0)
        
        print(f"Batch size: {batch_size}")
        print(f"Spectrogram shape: {spectrograms.shape}")
        print(f"Transcript lengths: {[len(t) for t in transcripts]}")
        
        # Check for NaN in input
        if torch.isnan(spectrograms).any():
            print("WARNING: NaN values in input spectrograms")
            
        # Check for Inf in input
        if torch.isinf(spectrograms).any():
            print("WARNING: Inf values in input spectrograms")
            
        # Print spectrogram stats
        print(f"Spectrogram min: {spectrograms.min().item()}")
        print(f"Spectrogram max: {spectrograms.max().item()}")
        print(f"Spectrogram mean: {spectrograms.mean().item()}")
        print(f"Spectrogram std: {spectrograms.std().item()}")
        
        # Forward pass with hooks to monitor activations
        activation_stats = {}
        hooks = []
        
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activation_stats[name] = {
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'has_nan': torch.isnan(output).any().item(),
                    'has_inf': torch.isinf(output).any().item()
                }
            return hook
        
        # Register hooks for the updated model architecture
        hooks.append(model.conv1.register_forward_hook(get_activation_hook('conv1')))
        hooks.append(model.conv2.register_forward_hook(get_activation_hook('conv2')))
        hooks.append(model.pool.register_forward_hook(get_activation_hook('pool')))
        hooks.append(model.lstm.register_forward_hook(get_activation_hook('lstm')))
        hooks.append(model.fc.register_forward_hook(get_activation_hook('fc_out')))
        
        try:
            # Forward pass
            outputs = model(spectrograms)
            
            # Check model output
            print(f"Model output shape: {outputs.shape}")
            print(f"Output min: {outputs.min().item()}")
            print(f"Output max: {outputs.max().item()}")
            print(f"Output mean: {outputs.mean().item()}")
            print(f"Output std: {outputs.std().item()}")
            
            # Check for NaN/Inf in output
            if torch.isnan(outputs).any():
                print("WARNING: NaN values in model output")
                
            if torch.isinf(outputs).any():
                print("WARNING: Inf values in model output")
            
            # Apply log softmax
            log_probs = F.log_softmax(outputs, dim=2)
            
            # Check log_probs
            print(f"Log probs min: {log_probs.min().item()}")
            print(f"Log probs max: {log_probs.max().item()}")
            print(f"Log probs mean: {log_probs.mean().item()}")
            
            # Check for NaN/Inf in log_probs
            if torch.isnan(log_probs).any():
                print("WARNING: NaN values in log_probs")
                
            if torch.isinf(log_probs).any():
                print("WARNING: Inf values in log_probs")
                
            # Transpose for CTC
            log_probs = log_probs.transpose(0, 1)
            
            # Prepare for CTC loss
            input_lengths = torch.full(size=(batch_size,), fill_value=outputs.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
            targets = torch.cat(transcripts)
            
            print(f"Input lengths: {input_lengths}")
            print(f"Target lengths: {target_lengths}")
            print(f"Targets shape: {targets.shape}")
            
            # Check if target_lengths are valid
            if (target_lengths <= 0).any():
                print("WARNING: Some target lengths are <= 0")
                
            # Check if input_lengths are valid
            if (input_lengths <= 0).any():
                print("WARNING: Some input lengths are <= 0")
                
            # Check if input_lengths >= target_lengths (CTC requirement)
            if not (input_lengths >= target_lengths).all():
                print("WARNING: Some input lengths are < target lengths")
                print(f"Input lengths: {input_lengths}")
                print(f"Target lengths: {target_lengths}")
            
            # Calculate CTC loss
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            print(f"CTC Loss: {loss.item()}")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN or Inf loss detected")
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
        
        # Print activation statistics
        print("\nLayer activation statistics:")
        for name, stats in activation_stats.items():
            print(f"{name}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Try with a very small model
        print("\nTrying with a minimal model...")
        mini_model = nn.Sequential(
            nn.Linear(spectrograms.shape[1] * spectrograms.shape[2], 64),
            nn.ReLU(),
            nn.Linear(64, LibriSpeech.NUM_CLASSES)
        ).to(device)
        
        # Flatten input
        flat_input = spectrograms.reshape(batch_size, -1)
        mini_output = mini_model(flat_input)
        
        # Check mini model output
        print(f"Mini model output shape: {mini_output.shape}")
        print(f"Mini output min: {mini_output.min().item()}")
        print(f"Mini output max: {mini_output.max().item()}")
        
        # Apply log softmax to mini model output
        mini_log_probs = F.log_softmax(mini_output, dim=1)
        
        print(f"Mini log probs min: {mini_log_probs.min().item()}")
        print(f"Mini log probs max: {mini_log_probs.max().item()}")
        
        # Try a dummy CTC loss calculation
        dummy_input_lengths = torch.full(size=(batch_size,), fill_value=1, dtype=torch.long)
        dummy_target_lengths = torch.ones(batch_size, dtype=torch.long)
        dummy_targets = torch.zeros(batch_size, dtype=torch.long)
        
        try:
            dummy_loss = criterion(
                mini_log_probs.unsqueeze(0),
                dummy_targets,
                dummy_input_lengths,
                dummy_target_lengths
            )
            print(f"Dummy CTC Loss: {dummy_loss.item()}")
        except Exception as e:
            print(f"Error in dummy CTC loss: {e}")

if __name__ == "__main__":
    debug_model()
