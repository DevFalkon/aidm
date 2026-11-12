# import os and path to handle directories and paths
import os
from pathlib import Path

# For parallel processing
from concurrent.futures import ProcessPoolExecutor 
from functools import partial

import numpy as np      # numpy for creating arrays and storing numpy objects
from PIL import Image   # PIL for image compression
from tqdm import tqdm   # tqdm to show progress bar for visualization


ROOT_DIR = Path(os.path.abspath(os.curdir))     # Abs path to project root directory

DATASET_DIR = Path(os.path.join(ROOT_DIR, "dataset")) # Abs path of dataset directory
CACHE_DIR   = Path(os.path.join(ROOT_DIR, "cache") )  # Abs path of chache directory (to stored cached images)

# paths of the dirs to be cached
TRAIN_DIR = Path(os.path.join(DATASET_DIR, "train"))
VALIDATE_DIR = Path(os.path.join(DATASET_DIR, "val"))

IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Size to resize the image to (in px)
IMG_SIZE = (128, 128) 

def load_single_image(item, size):
    # Worker fn to load and pad a single img preserving aspect ratio
    # return: (np_img_arr, img_label, img_path, error)
    
    img_path, label = item
    target_w, target_h = size

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_w, orig_h = img.size

            # Compute scaling factor to fit within target size maintaining aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # Resize the image using high-quality resampling
            img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            # Create a new black canvas of the target size
            new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))

            # Paste the resized image into the center of the black canvas
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            new_img.paste(img_resized, (paste_x, paste_y))

            img_arr = np.array(new_img, dtype=np.uint8)
            return img_arr, label, str(img_path), None
        
    except Exception as e:
        return None, None, str(img_path), f"{type(e).__name__}: {e}"
    

def process_split(split_name, split_path, workers=8):
    # Find all classes for given split (different sub-directories in the split)
    classes = sorted([sub_dir.name for sub_dir in split_path.iterdir() if sub_dir.is_dir()])

    # Convert class name to numbered labels
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"Processing '{split_name}' split. Found classes: {classes}")

    # image_items: (img_path, img_class)
    image_items = [
        (split_path / cl / img_name, class_to_idx[cl])
        for cl in classes
        for img_name in os.listdir(split_path / cl)
        if img_name.lower().endswith(IMAGE_FILE_EXTENSIONS)
    ]

    n_images = len(image_items)

    # Pre-allocate numpy arrays to avoid slow Python list growth/reallocation
    # Array len = n_samples x img_size
    # img_size  = img_width x img_height x 3 (RGB ch)
    # For our NN, img_widht = img_height
    x_arr = np.empty((n_images, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)

    labels_list = []    # arr to store img label
    paths_list  = []    # for tracking and debugging during training
    
    load_func = partial(load_single_image, size=IMG_SIZE)

    # Use multiprocessing to decode and resize across all CPU cores simultaneously
    valid_images = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = ex.map(load_func, image_items, chunksize=64)
        for img_arr, label, path_str, err in tqdm(results, total=n_images, desc=f"Resizing {split_name}"):
            if err:
                print(f"Failed to load {path_str}: {err}")
                continue
            
            x_arr[valid_images] = img_arr
            # to keep track np objects with their img paths
            # for testing and visual confirmation
            paths_list.append(path_str)

            labels_list.append(label)
            valid_images += 1

    # Trim arrays if any images failed to load
    if valid_images < n_images:
        x_arr = x_arr[:valid_images]

    # Transpose images from (N, H, W, C) to PyTorch's (N, C, H, W) format before saving
    x_arr = x_arr.transpose(0, 3, 1, 2)
    labels_arr = np.array(labels_list, dtype=np.int64)
    paths_arr = np.array(paths_list)

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / f"{split_name}_images.npy", x_arr)
    np.save(CACHE_DIR / f"{split_name}_labels.npy", labels_arr)
    np.save(CACHE_DIR / f"{split_name}_paths.npy", paths_arr)    
        
    print(f"Successfully cached {split_name}: {x_arr.shape}")


def main():
    # dataset split aaloing with its abs path
    splits = {
        "train": TRAIN_DIR,
        "val": VALIDATE_DIR,
    }

    # Number of CPU cores for parallel proceccing
    # Multiple cores used to increase speed of image caching
    workers = min(8, os.cpu_count() or 1)

    for split_name, split_path in splits.items():
        # Check to make sure split path is correct
        if not split_path.exists():
            print(f"Skipping {split_name}: {split_path} does not exist.")
            continue

        # Process each split individually:
        # Need to cache test and validate in seperate 
        # files that can be identified later
        process_split(split_name, split_path, workers=workers)
        
    print(f"\nCaching complete! Arrays saved in: {CACHE_DIR}")


if __name__ == "__main__":
    main()