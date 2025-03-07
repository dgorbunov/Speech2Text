import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from librispeech import LibriSpeech  # Using your custom LibriSpeech class

TRAIN_DATASET = "dev-clean"
VAL_DATASET = "dev-other"
TEST_DATASET = "test-clean"
NUM_TEST_SAMPLES = 5
BATCH_SIZE = 64
RESULTS_DIR = "./results"
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
LEARNING_RATE = 3e-4
NUM_EPOCHS = 15
GRADIENT_ACCUMULATION_STEPS = 4
USE_WANDB = False

Path(RESULTS_DIR).mkdir(exist_ok=True, parents=True)
Path(CHECKPOINT_DIR).mkdir(exist_ok=True, parents=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CHAR_MAP = LibriSpeech.char_map()
NUM_CLASSES = LibriSpeech.num_classes()
BLANK_INDEX = LibriSpeech.blank_index()

# Define Wav2Vec2 model with CTC head
class Wav2VecForCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec.lm_head = nn.Linear(768, NUM_CLASSES)  # 768 is the hidden size of wav2vec2-base
    
    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

def decode_predictions(logits, blank_idx=BLANK_INDEX):
    """Convert logits to text using greedy decoding"""
    predictions = torch.argmax(logits, dim=-1)
    batch_texts = []
    
    for prediction in predictions:
        
        previous = blank_idx
        text_ids = []
        
        for p in prediction:
            p = p.item()
            if p != previous and p != blank_idx:
                text_ids.append(p)
            previous = p
            
        text = ''.join([CHAR_MAP[idx] for idx in text_ids])
        batch_texts.append(text)
        
    return batch_texts

def compute_wer(pred_str, target_str):
    """Calculate Word Error Rate"""
    pred_words = pred_str.split()
    target_words = target_str.split()
    m, n = len(pred_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == target_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    distance = dp[m][n]
    wer = distance / max(len(target_words), 1) * 100
    
    return wer

def load_datasets():
    train_dataset = LibriSpeech(dataPath=DATA_DIR, subset=TRAIN_DATASET)
    val_dataset = LibriSpeech(dataPath=DATA_DIR, subset=VAL_DATASET)
    test_dataset = LibriSpeech(dataPath=DATA_DIR, subset=TEST_DATASET)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=LibriSpeech.pad,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=LibriSpeech.pad,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=LibriSpeech.pad,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def train(model, train_loader, val_loader, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')
    criterion = nn.CTCLoss(blank=BLANK_INDEX, reduction='mean')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, batch in enumerate(progress_bar):
            spectrograms, transcripts = batch
            spectrograms = spectrograms.squeeze(1).view(spectrograms.size(0), -1)
            spectrograms = spectrograms.to(device)  
            
            input_lengths = torch.full((spectrograms.size(0),), spectrograms.size(1), dtype=torch.long)
            
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
            
            # put all on 1 tensor
            targets = torch.cat(transcripts).to(device)
            outputs = model(input_values=spectrograms)
            logits = outputs.logits
            
            # Transpose to [time, batch, classes] 
            logits = logits.transpose(0, 1)
            
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})
            
            if USE_WANDB:
                wandb.log({"train_batch_loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})
        
        val_loss, val_wer = evaluate(model, val_loader, criterion)
        
        train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val WER = {val_wer:.2f}%")
        
        if USE_WANDB:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_wer": val_wer,
                "epoch": epoch
            })
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, val_wer)
            print(f"New best model saved with Val Loss = {val_loss:.4f}")

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_wers = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            spectrograms, transcripts = batch
            spectrograms = spectrograms.to(device)
            
            input_lengths = torch.full((spectrograms.size(0),), spectrograms.size(1), dtype=torch.long)
            
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)
            
            targets = torch.cat(transcripts).to(device)
            
            outputs = model(input_values=spectrograms)
            logits = outputs.logits
            
            logits_for_loss = logits.transpose(0, 1)
            
            loss = criterion(logits_for_loss, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            predicted_texts = decode_predictions(logits)
            
            target_texts = []
            offset = 0
            for length in target_lengths:
                target_text = ''.join([CHAR_MAP[idx.item()] for idx in targets[offset:offset+length]])
                target_texts.append(target_text)
                offset += length
            
            for pred_text, target_text in zip(predicted_texts, target_texts):
                wer = compute_wer(pred_text, target_text)
                all_wers.append(wer)
    
    avg_loss = total_loss / len(dataloader)
    avg_wer = sum(all_wers) / len(all_wers) if all_wers else 0.0
    
    return avg_loss, avg_wer

def save_checkpoint(model, optimizer, epoch, val_loss, val_wer):
    checkpoint_path = Path(CHECKPOINT_DIR) / f"wav2vec_epoch_{epoch}_loss_{val_loss:.4f}_wer_{val_wer:.2f}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_wer': val_wer
    }, checkpoint_path)

def run_inference(model, test_loader, test_dataset, num_samples=NUM_TEST_SAMPLES):
    model.eval()
    test_results = []
    
    with torch.no_grad():
        batch_count = 0
        sample_count = 0
        
        for batch in test_loader:
            if sample_count >= num_samples:
                break
                
            spectrograms, transcripts = batch
            spectrograms = spectrograms.s
            spectrograms = spectrograms.to(device)
            
            outputs = model(input_values=spectrograms)
            logits = outputs.logits
            
            predicted_texts = decode_predictions(logits)
            
            for i in range(min(2, len(predicted_texts))):
                if sample_count >= num_samples:
                    break
                    
                original_text = test_dataset.get_original_text(batch_count * BATCH_SIZE + i)
                
                wer = compute_wer(predicted_texts[i], original_text.lower())
                
                test_results.append({
                    "sample_idx": batch_count * BATCH_SIZE + i,
                    "predicted": predicted_texts[i],
                    "ground_truth": original_text,
                    "wer": wer
                })
                
                sample_count += 1
            
            batch_count += 1
    
    results_path = Path(RESULTS_DIR) / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    
    print(f"Test results saved to {results_path}")
    
    print("\nSample predictions:")
    for i, result in enumerate(test_results):
        print(f"Sample {i}:")
        print(f"  Ground truth: {result['ground_truth']}")
        print(f"  Prediction: {result['predicted']}")
        print(f"  WER: {result['wer']:.2f}%\n")

def main():
    if USE_WANDB:
        wandb.init(project="wav2vec2-finetuning", name=f"wav2vec2-librispeech-{time.strftime('%Y%m%d-%H%M%S')}")
    
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_datasets()
    
    model = Wav2VecForCTC()
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    train(model, train_loader, val_loader, optimizer, scheduler, NUM_EPOCHS)
    
    best_checkpoint = sorted(Path(CHECKPOINT_DIR).glob("*.pt"), key=lambda x: float(str(x).split("_loss_")[1].split("_wer_")[0]))[-1]
    checkpoint = torch.load(best_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint: {best_checkpoint}")
    
    criterion = nn.CTCLoss(blank=BLANK_INDEX, reduction='mean')
    test_loss, test_wer = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test WER: {test_wer:.2f}%")
    
    run_inference(model, test_loader, test_dataset)
    
    # Save final model
    final_model_path = Path(CHECKPOINT_DIR) / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()
