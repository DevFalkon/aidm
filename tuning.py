import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import numpy as np
import optuna
from tqdm import tqdm

from model import CatDogCNN

# ==================== CONSTANTS ====================
CACHE_DIR = Path(os.path.join(os.path.abspath(""), "cache"))
NUM_WORKERS = 0
TUNING_EPOCHS = 5  # Shortened epochs per trial for fast searching
# ===================================================

class CachedDataset(Dataset):
    def __init__(self, cache_dir, split="train", is_train=False, load_into_ram=False):
        cache_path = Path(cache_dir)
        if load_into_ram:
            self.x = np.load(cache_path / f"{split}_images.npy").copy()
            self.y = np.load(cache_path / f"{split}_labels.npy").copy()
        else:
            self.x = np.load(CACHE_DIR / f"{split}_images.npy")
            self.y = np.load(CACHE_DIR / f"{split}_labels.npy")
        self.is_train = is_train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx].astype(np.float32) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return img_tensor, label_tensor

def objective(trial, train_dataset, val_dataset, device):
    # 1. Suggest hyperparameter values to test
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # 2. Build DataLoaders with the trial's suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Initialize model, loss, and optimizer
    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Train for a few short epochs to evaluate this configuration
    for epoch in range(TUNING_EPOCHS):
        model.train()
        train_loop = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{TUNING_EPOCHS}", leave=False)
        
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loop.set_postfix(loss=loss.item())

    # 5. Calculate validation loss for this trial
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    epoch_val_loss = val_loss / len(val_dataset)
    return epoch_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets into RAM once so trials run instantly
    print("Loading datasets into RAM...")
    train_dataset = CachedDataset(CACHE_DIR, split="train", is_train=False, load_into_ram=True)
    val_dataset = CachedDataset(CACHE_DIR, split="val", is_train=False, load_into_ram=True)

    # Run Optuna Study
    print("Starting hyperparameter tuning study...")
    study = optuna.create_study(direction="minimize")
    
    # Optuna automatically prints a clean study progress bar to stdout by default, 
    # and our inner tqdm loop tracks each training batch.
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, device), n_trials=15)

    print("\nHyperparameter tuning complete!")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Validation Loss: {study.best_value:.4f}")

if __name__ == "__main__":
    main()
    