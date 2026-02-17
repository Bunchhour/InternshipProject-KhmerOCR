import torch
import torch.nn as nn

class SimpleOCR(nn.Module):
    def __init__(self, num_classes):
        super(SimpleOCR, self).__init__()
        
        # --- Feature Extractor (Simple CNN) ---
        # Input shape: (Batch, 1, 32, Width)
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Height: 32 -> 16, Width: W -> W/2

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            # Height: 16 -> 8, Width: W/2 -> W/4

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            # Height: 8 -> 4, Width: W/4 -> W/8
            
            # Layer 4 (Final Collapse)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Crucial: We pool Height (4->1) but keep Width (stride=1 for width)
            # This prepares the data to be a "sequence"
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)), 
            # Height: 4 -> 1, Width: W/8 -> W/8
        )
        
        # --- Classifier ---
        # Projects 256 features to the number of Khmer characters
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, 32, Width)
        
        # 1. Run CNN
        features = self.cnn(x) 
        # Output shape: (Batch, 256, 1, Width/8)
        
        # 2. Reshape for CTC
        # Remove the height dimension (which is now 1)
        features = features.squeeze(2) 
        # Shape: (Batch, 256, Width/8)
        
        # Permute: Swap dimensions to (Width, Batch, Channels)
        # In OCR, Width represents "Time"
        features = features.permute(2, 0, 1) 
        # Shape: (Time, Batch, 256)
        
        # 3. Classify
        output = self.fc(features) 
        # Shape: (Time, Batch, NumClasses)
        
        # 4. Log Softmax (Required for CTC Loss)
        # We apply it on the last dimension (Classes)
        return torch.nn.functional.log_softmax(output, dim=2)