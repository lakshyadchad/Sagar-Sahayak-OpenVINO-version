#!/usr/bin/env python3.8
"""
INT8 Quantization Script for Image Enhancement Model
Requires Python 3.8 with OpenVINO 2024.4.0

To run: py -3.8 quantize_model.py
"""
import sys

# Check Python version
if sys.version_info < (3, 8) or sys.version_info >= (3, 9):
    print("❌ ERROR: This script requires Python 3.8")
    print(f"   Current Python version: {sys.version}")
    print("\n✅ Solution: Run with Python 3.8:")
    print("   py -3.8 quantize_model.py")
    sys.exit(1)

import nncf
import openvino as ov
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

# --- CONFIGURATION ---
MODEL_PATH = "" # Input (The FP16 model from Step 2)
OUTPUT_PATH = "" # Output
CALIBRATION_DATA_DIR = "" # Use your Test Set (Murky Images)

class ImageDataset(Dataset):
    """Custom dataset to load images from a flat directory."""
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        # Get all image files (jpg, jpeg, png)
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.image_files.extend(list(self.image_dir.glob(ext)))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images for calibration")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return dummy label (not used)


def transform_fn(data_item):
    """
    This function tells NNCF how to unpack the data from the loader.
    NNCF passes the raw item from the DataLoader (image, label).
    We only need the image.
    """
    images, _ = data_item
    return images

def quantize_enhancer():
    print("--- Starting INT8 Quantization (Post-Training) ---")

    # 1. Load the OpenVINO Model
    core = ov.Core()
    print(f"Loading FP16 Model: {MODEL_PATH}")
    model = core.read_model(MODEL_PATH)

    # 2. Prepare Calibration Dataset
    # We need ~100-300 images. The model needs to see 'representative' data.
    # CRITICAL: The transforms here must MATCH your training transforms (Resize 256x256)
    print("Preparing Calibration Data...")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), # Converts to [0, 1] range
        # Note: OpenVINO expects N,C,H,W. DataLoader provides this.
    ])

    # Load images from flat directory
    dataset = ImageDataset(CALIBRATION_DATA_DIR, transform=transform)
    
    # We only need about 100 images for calibration
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Wrap it in an NNCF Dataset object
    calibration_dataset = nncf.Dataset(loader, transform_fn)

    # 3. Run Quantization
    print("Running NNCF Quantization... (This might take a minute)")
    # subset_size=300 means it will look at 300 images to calibrate.
    quantized_model = nncf.quantize(model, calibration_dataset, subset_size=300)

    # 4. Save the INT8 Model
    print(f"Saving INT8 Model to: {OUTPUT_PATH}")
    ov.save_model(quantized_model, OUTPUT_PATH)
    
    print("Success! INT8 Quantization Complete.")

if __name__ == "__main__":
    quantize_enhancer()