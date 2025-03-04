import json
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Load training stats
stats_path = Path("./checkpoints/training_stats.json")

if not stats_path.exists():
    print(f"No training stats found at {stats_path}")
    exit(1)

with open(stats_path, 'r') as f:
    stats = json.load(f)

# Create a figure with one subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot training loss
if 'losses' in stats:
    ax.plot(stats['epochs'], stats['losses'], 'b-', label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    ax.grid(True)

    # Add best loss as a horizontal line
    if 'best_loss' in stats:
        ax.axhline(y=stats['best_loss'], color='r', linestyle='--', 
                label=f'Best Loss: {stats["best_loss"]:.4f}')
        ax.legend()

# Calculate total training time from individual epoch times
if 'times' in stats:
    total_time = sum(stats['times'])
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

# Save the figure
plt.tight_layout()
plt.savefig('./checkpoints/training_progress.png')
print(f"Training visualization saved to ./checkpoints/training_progress.png")

plt.show()
