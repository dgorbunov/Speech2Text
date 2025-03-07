import torch
import torch.nn as nn
from librispeech import NUM_MELS, LibriSpeech
from torch.nn import functional as F

# Hyperparameters
CNN_LAYER1_SIZE = 64
CNN_LAYER2_SIZE = 128
LSTM_HIDDEN_LAYERS = 512
NUM_LSTM_LAYERS = 3
DROPOUT_RATE = 0.2

class SpeechLSTM(nn.Module):
    def __init__(self, num_classes=LibriSpeech.NUM_CLASSES):
        super(SpeechLSTM, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, CNN_LAYER1_SIZE, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_LAYER1_SIZE),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Conv2d(CNN_LAYER1_SIZE, CNN_LAYER2_SIZE, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_LAYER2_SIZE),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Feature dimension for LSTM
        self.feature_dim = 512
        
        # Adaptive pooling for non-MPS devices
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 1))
        
        # Projection layer to convert CNN output to LSTM input
        self.projection = nn.Linear(CNN_LAYER2_SIZE * 4, self.feature_dim)
        
        # LSTM for temporal context
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=LSTM_HIDDEN_LAYERS,
            num_layers=NUM_LSTM_LAYERS,
            bidirectional=True,
            dropout=DROPOUT_RATE,
            batch_first=True
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_LAYERS * 2, LSTM_HIDDEN_LAYERS),
            nn.LayerNorm(LSTM_HIDDEN_LAYERS),
            nn.LeakyReLU(0.1),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(LSTM_HIDDEN_LAYERS, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                
    def forward(self, x):
        batch_size, time_steps, n_mels = x.size()
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Apply CNN
        x = self.cnn(x)
        
        # Get dimensions
        batch_size, channels, height, width = x.size()
        
        # Handle pooling differently based on device
        if x.device.type == 'mps':
            # For MPS devices, use a different approach
            x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
            x = x.reshape(batch_size, width, channels * height)
            
            # Use linear projection to get to feature_dim
            if x.size(2) != self.feature_dim:
                proj = nn.Linear(x.size(2), self.feature_dim).to(x.device)
                x = proj(x)
        else:
            # For CUDA or CPU devices, use a simpler approach
            # Flatten the CNN output and use a fixed projection
            x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
            x = x.contiguous().view(batch_size, width, -1)
            
            # Project to the required feature dimension
            proj = nn.Linear(x.size(2), self.feature_dim).to(x.device)
            x = proj(x)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Apply enhanced projection
        x = self.fc(x)
        
        return x
    
    @classmethod
    def num_classes(cls):
        return LibriSpeech.NUM_CLASSES
        
    def to_torchscript(self):
        self.eval()
        example_input = torch.randn(1, 100, NUM_MELS)
        return torch.jit.trace(self, example_input)
