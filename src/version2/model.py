import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # --- 1. CNN Feature Extractor (Same as Phase 2) ---
        # Input: (Batch, 1, 32, Width)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((4, 1))
        )
        
        # --- 2. RNN (BiLSTM) ---
        # Bidirectional = True means we get 2x hidden_size output
        self.rnn = nn.LSTM(input_size=256, 
                           hidden_size=hidden_size, 
                           num_layers=2, 
                           bidirectional=True, 
                           batch_first=False) # We use (Time, Batch, Features)
        
        # --- 3. Classifier ---
        # Input is 2 * hidden_size because of bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 1. Run CNN
        features = self.cnn(x) # (Batch, 256, 1, Width/8)
        
        # 2. Reshape for RNN
        features = features.squeeze(2) # (Batch, 256, Width/8)
        features = features.permute(2, 0, 1) # (Time, Batch, 256)
        
        # 3. Run RNN
        # rnn_out shape: (Time, Batch, Hidden*2)
        rnn_out, _ = self.rnn(features)
        
        # 4. Classify
        output = self.fc(rnn_out)
        
        # 5. Log Softmax
        return torch.nn.functional.log_softmax(output, dim=2)