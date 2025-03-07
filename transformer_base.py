import torch
import torch.nn as nn
import torch.nn.functional as F
from librispeech import NUM_MELS

class SpeechTransformer(nn.Module):
    def __init__(self, num_classes, d_model=256, n_heads=8, num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(SpeechTransformer, self).__init__()

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # Assuming max sequence length of 1000

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Feature Projection
        self.input_projection = nn.Linear(NUM_MELS, d_model)

        # Fully Connected Layer to Character Classes
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input shape: (batch, time, n_mels)
        x = self.input_projection(x)  # (batch, time, d_model)

        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, time, d_model)

        # Fully Connected Layer
        x = self.fc(x)  # (batch, time, num_classes)

        return x

