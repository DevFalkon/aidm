# Cats vs. Dogs VGG-Style CNN Classifier

A high-performance, resolution-agnostic custom Convolutional Neural Network (CNN) built from scratch in PyTorch to classify images from the Cats vs. Dogs dataset. This project leverages a VGG-inspired architecture, robust data augmentations, hyperparameter optimization via Optuna, and strict reproducibility controls to push validation accuracy.

---

## Project Highlights

- **VGG-Style Architecture:** Uses stacked $3 \times 3$ convolutional layers, Batch Normalization, and ReLU activations grouped into four modular blocks.
- **Resolution-Agnostic:** Built with **Global Average Pooling** instead of massive fully connected layers, allowing the network to handle any image dimension divisible by 16 (optimized at **$144 \times 144$**).
- **Hyperparameter Optimization:** Tuned using **Optuna** to locate optimal learning rates, batch sizes, dropout rates, and weight decay penalties.
- **Robust Training Pipeline:** Features dynamic on-the-fly GPU augmentations, learning rate scheduling (`ReduceLROnPlateau`), periodic checkpointing, and model resumption capabilities.
- **Strict Reproducibility:** Fixed random seeding across Python, NumPy, and PyTorch (along with deterministic CuDNN flags) to guarantee identical training trajectories across runs.

---

## How to Reproduce Our Results?

NOTE: This repo does not contain the Cats vs Dogs dataset, place the dataset in the same directory as this project. Make sure the Dataset has the following sub-directories: train, val and test (ie dataset/train, dataset/val and dataset/test).

### Setting Up python environment

1. create and activate python virtual environment

```
python -m venv env
env\Scripts\activate
```

2. Install dependancies

Note: the requirements.txt presumes your cuda version is 13.2, if not (or no cuda), install the correct pytorch version from: https://pytorch.org/get-started/locally/

Installatoin of requirements with cuda 13.2

```
pip install -r requirements_with_cuda132.txt
```

Installation of requriements (manually install pytorch from the link above)

```
pip install -r requirements_no_cuda.txt
```

### Pre-processing Train and Validate Dataset

In order to reduce training time, the images in train and val directies need to resized and cached as numpy arrays.

to pre process, run:

```
python pre_process.py
```

This should create a folder called cache with 6 .npy files

### Training the model

NOTE: If you don't want to load the entire cached dataset into RAM, change load_into_ram to False in lines 135 and 136 in train.py

Train model:

```
python train.py
```

### Model Evaluation on Test dataset

NOTE: test.py requires a saved pytorch model, by default: "epochs/model_epoch_80.pth". If your trained model has a different name or path, accordingly modify the path in line 15 of test.py.

Get submission.csv

```
python test.py
```

### (Optional) Model Tuning

NOTE: Our model has already been tuned for Cats vs Dogs dataset

To tune model:

```
python tune.py
```
