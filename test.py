import os
import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from model import CatDogCNN

# ==================== CONSTANTS ====================
TEST_DIR        = Path("dataset/test")
MODEL_PATH      = Path("epochs/model_epoch_80.pth")
OUTPUT_CSV      = Path("predictions.csv")

IMAGE_SIZE      = 144   # Matches the training resolution
BATCH_SIZE      = 64
NUM_WORKERS     = 2
# ===================================================

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        # Collect valid image files and sort them naturally (e.g., 1.jpg, 2.jpg, ..., 10.jpg)
        valid_exts = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted(
            [p for p in self.test_dir.iterdir() if p.suffix.lower() in valid_exts],
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Return the tensor and the filename (or ID) for the CSV
        return image, img_path.name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Define preprocessing (Matches training: resize to 144x144 and scale to [0, 1])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # 2. Load Test Dataset
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found at {TEST_DIR.resolve()}")
        
    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True
    )
    print(f"Found {len(test_dataset)} test images.")

    # 3. Initialize Model and Load Best Weights
    model = CatDogCNN().to(device)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH.resolve()}")
        
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Handles both direct state_dict saves (best_cat_dog_cnn.pth) and full checkpoint dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"Loaded weights from {MODEL_PATH}")

    # 4. Run Inference
    results = []
    with torch.no_grad():
        for images, filenames in tqdm(test_loader, desc="Classifying"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for filename, label in zip(filenames, predicted.cpu().tolist()):
                # Extract numeric ID if filenames are like '1.jpg' (common for Kaggle), 
                # otherwise keep the filename
                file_id = Path(filename).stem if Path(filename).stem.isdigit() else filename
                results.append((file_id, label))

    # 5. Save Predictions to CSV
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows(results)

    print(f"\nInference complete! Saved {len(results)} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()