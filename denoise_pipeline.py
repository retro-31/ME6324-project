import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

INPUT_DIR = "./data"
OUTPUT_DIR = "./dataset_resized"
DENOISED_DIR = "./dataset_denoised"
TARGET_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DENOISED_DIR, exist_ok=True)

def resize_images(input_dir, output_dir, size=256):
    for root, _, files in os.walk(input_dir):
        rel = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel)
        os.makedirs(save_dir, exist_ok=True)

        for f in tqdm(files, desc=f"Resizing {rel}"):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            try:
                img = Image.open(os.path.join(root, f)).convert("RGB")
                img = img.resize((size, size), Image.BICUBIC)
                img.save(os.path.join(save_dir, f))
            
            except Exception as e:
                print(f"Error processing {f}: {e}")

resize_images(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)