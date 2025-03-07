import json
import matplotlib.pyplot as plt
import os
import argparse

def load_training_stats(json_path):
    with open(json_path, 'r') as f:
        stats = json.load(f)
    return stats

def plot_metrics(stats, save_dir=None):
    # Create a figure with subplots based on available metrics
    metrics = []
    epochs = []
    
    # Determine available metrics and epochs
    if isinstance(stats, list):
        # List of dictionaries format
        epochs = list(range(1, len(stats) + 1))
        if len(stats) > 0:
            metrics = [key for key in stats[0].keys() if key != 'epoch']
    elif isinstance(stats, dict):
        # Dictionary with metric names as keys format
        for metric_name, values in stats.items():
            if isinstance(values, list) and len(values) > 0:
                metrics.append(metric_name)
                if len(epochs) == 0:
                    epochs = list(range(1, len(values) + 1))
    
    if not metrics:
        print("No metrics found in the training stats file")
        return
    
    # Create plots
    n_metrics = len(metrics)
    fig_height = 4 * n_metrics
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, fig_height))
    
    # Handle case with only one metric
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if isinstance(stats, list):
            # Extract values for this metric from list of dicts
            values = [entry.get(metric, None) for entry in stats]
            # Filter out None values
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_epochs = [epochs[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]
            
            ax.plot(valid_epochs, valid_values, 'o-', label=metric)
        elif isinstance(stats, dict):
            # Extract values for this metric from dict of lists
            values = stats[metric]
            ax.plot(epochs, values, 'o-', label=metric)
        
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        print(f"Plot saved to {os.path.join(save_dir, 'training_metrics.png')}")
    
    plt.show()
    
parser = argparse.ArgumentParser(description='Visualize training statistics')
parser.add_argument('--stats_file', type=str, default='checkpoints/training_stats.json',
                help='Path to the training stats JSON file')
parser.add_argument('--save_dir', type=str, default='results',
                help='Directory to save the plots')
args = parser.parse_args()

stats = load_training_stats(args.stats_file)
plot_metrics(stats, args.save_dir)
