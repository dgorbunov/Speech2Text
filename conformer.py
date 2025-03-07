import torch
import torch.nn as nn
import torch.nn.functional as F
from librispeech import NUM_MELS

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, conv_kernel_size=31, dropout=0.1):
        super(ConformerBlock, self).__init__()

        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln_attn = nn.LayerNorm(d_model)

        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2, groups=d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ln_conv = nn.LayerNorm(d_model)

        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Feedforward Module 1
        x = x + 0.5 * self.ffn1(x)

        # Multi-Head Self-Attention
        attn_out, _ = self.multihead_attn(x, x, x)
        x = x + attn_out
        x = self.ln_attn(x)

        # Convolutional Module
        conv_out = self.conv_module(x.transpose(1, 2)).transpose(1, 2)
        x = x + conv_out
        x = self.ln_conv(x)

        # Feedforward Module 2
        x = x + 0.5 * self.ffn2(x)

        return x

class ConformerASR(nn.Module):
    def __init__(self, num_classes, d_model=256, n_heads=8, num_layers=6, dim_feedforward=1024, conv_kernel_size=31, dropout=0.1):
        super(ConformerASR, self).__init__()

        self.input_projection = nn.Linear(NUM_MELS, d_model)  # Project input to model dimension
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # Learnable positional encoding

        self.conformer_layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dim_feedforward, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_projection(x)  # (batch, time, d_model)

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.positional_encoding[:, :seq_len, :]

        # Conformer Layers
        for layer in self.conformer_layers:
            x = layer(x)

        # Fully Connected Layer
        x = self.fc(x)  # (batch, time, num_classes)

        return x

