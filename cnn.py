import torch
import torch.nn as nn
import torch.nn.functional as F
from librispeech import NUM_MELS  # Import NUM_MELS from librispeech.py

# Define a simple Speech-to-Text Model (CNN + BiLSTM + CTC)
class SpeechCNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(SpeechCNN, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Input shape: (batch, 1, time, n_mels)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))  # Reduces feature map size
        )

        # BiLSTM for Temporal Context
        self.lstm = nn.LSTM(input_size=(NUM_MELS // 2) * 64,
                            hidden_size=hidden_size, 
                            num_layers=2, 
                            bidirectional=True, 
                            batch_first=True)

        # Fully Connected Layer to Character Classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # Input shape: (batch, time, n_mels)
        # Transpose to (batch, n_mels, time) for CNN
        x = x.transpose(1, 2)
        
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)
        
        x = self.cnn(x)  # CNN feature extraction

        # Rearrange dimensions for LSTM
        batch_size, channels, mels, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, mels)
        x = x.view(batch_size, time, -1)  # (batch, time, channels*mels)

        # LSTM for Temporal Processing
        x, _ = self.lstm(x)

        # Fully Connected Layer
        x = self.fc(x)

        return x

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(SimpleCNN, self).__init__()
        
        # Linear Feature Extractor (replaces the CNN)
        self.feature_extractor = nn.Sequential(
            # Input shape: (batch, time, n_mels)
            nn.Linear(NUM_MELS, 1024),
            nn.ReLU(),
            nn.Linear(1024, (NUM_MELS // 2) * 64),
            nn.ReLU()
        )
        
        
        # BiLSTM for Temporal Context
        self.lstm = nn.LSTM(
            input_size=(NUM_MELS // 2) * 64,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully Connected Layer to Character Classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional


    def forward(self, x):
        # Input shape: (batch, time, n_mels)
        x = self.feature_extractor(x)

        # # Rearrange dimensions for LSTM
        # batch_size, channels, mels, time = x.size()
        # x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, channels, mels)
        # x = x.view(batch_size, time, -1)  # (batch, time, channels*mels)

        # LSTM for Temporal Processing
        x, _ = self.lstm(x)

        # Fully Connected Layer
        x = self.fc(x)

        return x

