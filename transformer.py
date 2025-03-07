import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from librispeech import NUM_MELS, LibriSpeech

LSTM_HIDDEN_LAYERS = 256
CNN_LAYER1_SIZE = 32
CNN_LAYER2_SIZE = 64
NUM_ENCODER_LAYERS = 6  
DROPOUT_RATE = 0.3
BLANK_TOKEN_BIAS = -5.0
NUM_HEADS = 8  
FF_HIDDEN_DIM = 512  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SpeechTransformer(nn.Module):
    def __init__(self, num_classes=LibriSpeech.NUM_CLASSES):
        super(SpeechTransformer, self).__init__()
        
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, CNN_LAYER1_SIZE, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_LAYER1_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Conv2d(CNN_LAYER1_SIZE, CNN_LAYER2_SIZE, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(CNN_LAYER2_SIZE),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT_RATE)
        )

        
        cnn_output_height = NUM_MELS // 4  
        self.cnn_output_size = cnn_output_height * CNN_LAYER2_SIZE
        
        self.positional_encoding = PositionalEncoding(self.cnn_output_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_output_size,
            nhead=NUM_HEADS,
            dim_feedforward=FF_HIDDEN_DIM,
            dropout=DROPOUT_RATE
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)

        
        self.fc = nn.Linear(self.cnn_output_size, num_classes)
        
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
        
        if hasattr(self, 'fc'):
            blank_idx = LibriSpeech.BLANK_INDEX
            self.fc.bias.data[blank_idx] = BLANK_TOKEN_BIAS
    
    def forward(self, x):
        batch_size, time_steps, n_mels = x.size()
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.cnn(x)  
        
        x = x.permute(0, 3, 1, 2)  
        x = x.reshape(batch_size, x.size(1), -1)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x.permute(1, 0, 2))
        
        x = x.permute(1, 0, 2)
        
        x = self.fc(x)
        
        return x
    
    @classmethod
    def num_classes(cls):
        return LibriSpeech.NUM_CLASSES

