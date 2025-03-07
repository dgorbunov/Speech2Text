import torch
import torch.nn as nn
from librispeech import NUM_MELS, LibriSpeech

LSTM_HIDDEN_LAYERS = 256
CNN_LAYER1_SIZE = 32
CNN_LAYER2_SIZE = 64
NUM_LSTM_LAYERS = 2
DROPOUT_RATE = 0.3
BLANK_TOKEN_BIAS = -5.0

class SpeechLSTM(nn.Module):
    def __init__(self, num_classes=LibriSpeech.NUM_CLASSES):
        super(SpeechLSTM, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            # First CNN layer with batch normalization
            nn.Conv2d(1, CNN_LAYER1_SIZE, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_LAYER1_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT_RATE),
            
            # Second CNN layer with batch normalization
            nn.Conv2d(CNN_LAYER1_SIZE, CNN_LAYER2_SIZE, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_LAYER2_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Calculate CNN output size for the LSTM input
        cnn_output_height = NUM_MELS // 4  # After two MaxPool2d with stride 2
        self.cnn_output_size = cnn_output_height * CNN_LAYER2_SIZE
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=LSTM_HIDDEN_LAYERS,
            num_layers=NUM_LSTM_LAYERS,
            bidirectional=True,
            dropout=DROPOUT_RATE,
            batch_first=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(LSTM_HIDDEN_LAYERS * 2, num_classes)
        
        # Initialize weights using standard PyTorch initialization
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
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Add bias against blank token predictions
        if hasattr(self, 'fc'):
            blank_idx = LibriSpeech.BLANK_INDEX
            self.fc.bias.data[blank_idx] = BLANK_TOKEN_BIAS
    
    def forward(self, x):
        # x shape: (batch_size, time_steps, n_mels)
        batch_size, time_steps, n_mels = x.size()
        
        # Add channel dimension for CNN
        x = x.unsqueeze(1) # (batch_size, 1, time_steps, n_mels)
        
        # Prepare for CNN: (batch_size, 1, n_mels, time_steps)
        x = x.permute(0, 1, 3, 2)
        
        # Apply CNN layers
        x = self.cnn(x) # (batch_size, channels, n_mels/4, time_steps/4)
        
        # Reshape for LSTM
        x = x.permute(0, 3, 1, 2) # (batch_size, time_steps/4, channels, n_mels/4)
        x = x.reshape(batch_size, x.size(1), -1) # (batch_size, time_steps/4, channels*n_mels/4)
        
        # Apply LSTM
        x, _ = self.lstm(x) # (batch_size, time_steps/4, 2*hidden_size)
        
        # Apply final fully connected layer
        x = self.fc(x) # (batch_size, time_steps/4, num_classes)
        
        return x
    
    @classmethod
    def num_classes(cls):
        return LibriSpeech.NUM_CLASSES
        
    # Faster inference on MPS    
    def to_torchscript(self):
        self.eval()
        example_input = torch.randn(1, 100, NUM_MELS)
        return torch.jit.trace(self, example_input)