import torch
import torch.nn as nn
from librispeech import NUM_MELS, LibriSpeech

# Hyperparameters
LSTM_HIDDEN_SIZE = 256
LSTM_LAYERS = 3
CONV1_OUT_CHANNELS = 32
CONV2_OUT_CHANNELS = 64
DROPOUT_RATE = 0.2

class SpeechLSTM(nn.Module):
    def __init__(self, num_classes=LibriSpeech.NUM_CLASSES):
        super(SpeechLSTM, self).__init__()
        
        # Simple CNN for feature extraction
        self.conv1 = nn.Conv2d(1, CONV1_OUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CHANNELS)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(DROPOUT_RATE)
        
        # Second conv layer
        self.conv2 = nn.Conv2d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CHANNELS)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(DROPOUT_RATE)
        
        # Pooling in both time and frequency dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate feature dimension after pooling
        self.feature_dim = CONV2_OUT_CHANNELS * (NUM_MELS // 2)
        
        # LSTM layer
        self.lstm_hidden_size = LSTM_HIDDEN_SIZE
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=LSTM_LAYERS,
            bidirectional=True,
            dropout=DROPOUT_RATE,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(self.lstm_hidden_size * 2, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                
    def forward(self, x):
        batch_size, time_steps, n_mels = x.size()
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Pool in both dimensions
        x = self.pool(x)
        
        # Get new dimensions after pooling
        _, channels, new_time, new_freq = x.size()
        
        # Reshape for LSTM - flatten the frequency dimension
        x = x.permute(0, 2, 1, 3)  # [batch, new_time, channels, new_freq]
        x = x.reshape(batch_size, new_time, self.feature_dim)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Apply output layer
        x = self.fc(x)
        
        # Scale output
        x = x * 0.1
        
        return x
    
    @classmethod
    def num_classes(cls):
        return LibriSpeech.NUM_CLASSES
        
    def to_torchscript(self):
        self.eval()
        example_input = torch.randn(1, 100, NUM_MELS)
        return torch.jit.trace(self, example_input)
