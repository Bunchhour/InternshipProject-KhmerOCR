import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your modules

# from src.utils import KhmerLabelConverter # ⚠️Need to check
from utils import KhmerLabelConverter
# from src.model import SimpleOCR
from model import SimpleOCR
from datasets import load_dataset
from dataset import KhmerOCRDataset

def train():
    # --- COFIG ---
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCH = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    print(f"Training on {DEVICE}....")

    # 1. Setup Data
    # Load raw data to build vocab
    raw_data = load_dataset("seanghay/khmer-hanuman-100k", split="train[:2000]") # Small subset for Sprint 2
    all_text = "".join([x['text'] for x in raw_data])
    vocab = sorted(list(set(all_text)))
    """ 
    What happens here:

    1. Collect all Khmer characters
    2. Remove duplicates
    3. Sort characters
    """
    converter = KhmerLabelConverter(vocab)
    train_dataset = KhmerOCRDataset(split="train[:2000]", converter=converter)

    # Collate function handles variable width images
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        original_texts = [item['original_text'] for item in batch]
        label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

        # Pad images to the widest in the batch
        max_w = max([img.shape[2] for img in images])
        padded_imgs = torch.zeros(len(images), 1,32, max_w)
        for i, img in enumerate(images):
            w = img.shape[2]
            padded_imgs[i, :, :, :w] = img

        # Flatten labels for CTC
        labels_concat = torch.cat(labels)
        return padded_imgs, labels_concat, label_lengths, original_texts
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 2. setup Model
    model = SimpleOCR(num_classes=converter.get_num_classes()).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criteration = nn.CTCLoss(blank=0, zero_infinity=True)

    # 3. Training Loop
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}")

        for images, targets, target_lengths, origainal_texts in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward Pass
            # Preds shape: (Time, Batch, NumClasses)
            preds = model(images)

            # Calculate Input Lengths (Time steps)
            # CNN reduces width by 8x (2*2*2). So Time = Width // 8

            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long).to(DEVICE)

            # Calculate Loss
            loss = criteration(preds, targets, input_lengths, target_lengths)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        # --- QUICK TEST (Reality Check) ---
        # Decode one prediction to see if it's learning
        with torch.no_grad():
            # Greedy Decode: Take max probability at each step
            _, max_index = torch.max(preds, dim=2) # (Time, Batch)
            pred_indices = max_index[:, 0].cpu().numpy().tolist() # Take first item in batch
            decoded_text = converter.decode(pred_indices)
            
            print(f"\n--- Reality Check (Epoch {epoch+1}) ---")
            print(f"Target: {original_texts[0]}")
            print(f"Pred:   {decoded_text}")
            print(f"--------------------------------------\n")
    # Save Model
    torch.save(model.state_dict(), "checkpoints/sprint2_model.pth")
    print("✅ Model Saved!")

if __name__ == "__main__":
    train()