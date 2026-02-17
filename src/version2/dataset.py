import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class KhmerOCRDataset(Dataset):
    def __init__(self, dataset_name="seanghay/khmer-hanuman-100k", split="train", transform=None, converter=None):
        """
        Args:
            dataset_name: HuggingFace dataset path
            split: Which slice of data to load (e.g., 'train[:2000]')
            converter: Instance of KhmerLabelConverter (from src.utils)
        """
        print(f"📥 Loading dataset: {dataset_name} ({split})...")
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transform
        self.converter = converter
        
        if converter is None:
            raise ValueError("You must provide a valid 'converter' object to KhmerOCRDataset!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Get the data sample
        item = self.dataset[idx]
        pil_image = item['image'] # Original PIL Image
        text = item['text']       # Original Khmer String

        # 2. Convert PIL Image to Grayscale (L mode)
        # We convert to numpy array immediately
        image = np.array(pil_image.convert("L")) 

        # 3. Resize Image (Fixed Height=32, Preserve Aspect Ratio)
        h, w = image.shape
        target_h = 32
        
        # Calculate new width to maintain aspect ratio
        target_w = int(w * (target_h / h))
        
        # Use OpenCV to resize
        image = cv2.resize(image, (target_w, target_h))

        # 4. Normalization (Scale pixels to 0-1 range)
        image = image.astype(np.float32) / 255.0
        
        # 5. Add Channel Dimension -> (1, 32, W)
        # PyTorch expects (Channels, Height, Width)
        image = np.expand_dims(image, axis=0) 

        # 6. Encode Label (Text -> IDs)
        encoded_text = self.converter.encode(text)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "label": encoded_text,
            "original_text": text,
            "label_len": len(encoded_text)
        }