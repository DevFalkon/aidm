import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

from model import CatDogCNN

# ==================== CONSTANTS ====================
CACHE_DIR = Path(os.path.join(os.path.abspath(""), "cache"))
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 25
NUM_WORKERS = 0
# ===================================================

class CachedDataset(Dataset):
    # Dataset class to load pre-processed numpy files
    def __init__(self, cache_dir, split="train", is_train=False, load_into_ram=False):
        cache_path = Path(cache_dir)

        # Load dataset directly into RAM
        # Only use if there is sufficient RAM to load train + val 
        if load_into_ram:
            self.x = np.load(cache_path / f"{split}_images.npy").copy()
            self.y = np.load(cache_path / f"{split}_labels.npy").copy()

        else:
            # Matches your cache file naming: train_images.npy / val_images.npy
            self.x = np.load(CACHE_DIR / f"{split}_images.npy")
            # Matches your cache file naming: train_labels.npy / val_labels.npy
            self.y = np.load(CACHE_DIR / f"{split}_labels.npy")
        
        self.is_train = is_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Scale uint8 [0, 255] pixels down to float32 [0, 1] range
        img = self.x[idx].astype(np.float32) / 255.0
        
        # Optional on-the-fly random horizontal flip for training data
        # Note: Since shape is already (C, H, W), flipping width means flipping the last axis (-1)
        if self.is_train and np.random.rand() > 0.5:
            img = np.flip(img, axis=-1).copy()

        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        
        return img_tensor, label_tensor

def main():
    # Device Configuration for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets using the fast NumPy cache
    train_dataset = CachedDataset(CACHE_DIR, split="train", is_train=True, load_into_ram=True)
    val_dataset = CachedDataset(CACHE_DIR, split="val", is_train=False, load_into_ram=True)

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
    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
