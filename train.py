import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

from model import CatDogCNN

# ==================== CONSTANTS ====================
CACHE_DIR = Path(os.path.join(os.path.abspath(""), "cache"))

# Model hyperparameters - optimised using optuna
BATCH_SIZE    = 32
LEARNING_RATE = 1.9e-4
DROPOUT       = 0.38
WEIGHT_DECAY  = 8.8e-5

EPOCHS = 50

NUM_WORKERS = 0     # 0 workers is fastest if dataset is loaded into RAM

ROTATION_ANGLE = 20 # Rotation angle for image augmentation
# ===================================================

class CachedDataset(Dataset):
    def __init__(self, cache_dir, split="train", load_into_ram=False):
        cache_path = Path(cache_dir)

        if load_into_ram:
            self.x = np.load(cache_path / f"{split}_images.npy").copy()
            self.y = np.load(cache_path / f"{split}_labels.npy").copy()
        else:
            self.x = np.load(CACHE_DIR / f"{split}_images.npy")
            self.y = np.load(CACHE_DIR / f"{split}_labels.npy")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Shape is (C, H, W), uint8 [0, 255]
        img = self.x[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0

        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        
        return img_tensor, label_tensor

    
def apply_augmentations(images):
    # images shape: (Batch_Size, Channels, Height, Width) on GPU
    batch_size, c, h, w = images.shape
    
    # 1. Random Horizontal Flip (Applied per image or batched)
    # TF functions support batched tensors (B, C, H, W) as well!
    if torch.rand(1).item() > 0.5:
        images = TF.hflip(images)
        
    # 2. Random Rotation
    if torch.rand(1).item() > 0.5:
        angle = int(torch.randint(-ROTATION_ANGLE, ROTATION_ANGLE+1, (1,)).item())
        images = TF.rotate(images, angle)
        
    # 3. Color Jitter (Brightness & Contrast)
    if torch.rand(1).item() > 0.5:
        b_factor = float(torch.rand(1).item() * 0.4 + 0.8)
        images = TF.adjust_brightness(images, b_factor)
        c_factor = float(torch.rand(1).item() * 0.4 + 0.8)
        images = TF.adjust_contrast(images, c_factor)

    if torch.rand(1).item() > 0.5:
        # Crop 80% to 100% of the dimension size
        crop_h = int(h * torch.uniform(0.8, 1.0) if hasattr(torch, 'uniform') else h * (0.8 + 0.2 * torch.rand(1).item()))
        crop_w = int(w * (0.8 + 0.2 * torch.rand(1).item()))
        
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        
        images = TF.resized_crop(images, top, left, crop_h, crop_w, size=(h, w))
    

    return images


def main():
    # Device Configuration for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets using the fast NumPy cache
    # is_train can be set to False for hyperparameter tuning
    train_dataset = CachedDataset(CACHE_DIR, split="train", load_into_ram=True)
    val_dataset = CachedDataset(CACHE_DIR, split="val", load_into_ram=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Initialize Model, Loss, Optimizer, and Scheduler
    model = CatDogCNN(dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    model_save_path = "best_cat_dog_cnn.pth"

    # Training & Validation Loop with tqdm
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loop = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)

            # (Only applied during training)
            images = apply_augmentations(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        val_loop = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_loop.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Step scheduler based on validation loss
        scheduler.step(epoch_val_loss)

        # Print summary for the epoch
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc*100:.2f}%")

        # Early Stopping Check: Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved new best model to {model_save_path}")

    print(f"\nTraining complete. Best model saved with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
