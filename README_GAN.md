# GAN-Based Synthetic Image Generator

A standalone Python script for generating synthetic corrosion and non-corrosion images using Deep Convolutional GANs (DCGAN).

## Overview

This script implements a DCGAN to augment your corrosion detection dataset by generating realistic synthetic images for both CORROSION and NOCORROSION categories.

## Requirements

Make sure you have the following packages installed:

```bash
pip install tensorflow pillow numpy matplotlib
```

Or if you're using the existing environment, the packages should already be installed.

## Directory Structure

The script expects the following structure:

```
ME6324-project/
├── gan_synthetic_image_generator.py  (the main script)
├── data/
│   ├── CORROSION/         (your corrosion images)
│   └── NOCORROSION/       (your non-corrosion images)
└── GAN_Synthetic_Images/  (will be created automatically)
    ├── models/
    ├── synthetic_corrosion/
    ├── synthetic_nocorrosion/
    └── training_samples/
```

## Usage

### Basic Usage

Run with default settings (100 epochs, generate 200 images per category):

```bash
python gan_synthetic_image_generator.py
```

### Custom Settings

```bash
# Train for 150 epochs and generate 500 synthetic images per category
python gan_synthetic_image_generator.py --epochs 150 --num-synthetic 500

# Use smaller batch size (good for limited memory)
python gan_synthetic_image_generator.py --batch-size 16

# Train only for corrosion images
python gan_synthetic_image_generator.py --skip-nocorrosion

# Train only for no-corrosion images
python gan_synthetic_image_generator.py --skip-corrosion
```

### Available Arguments

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 32)
- `--num-synthetic`: Number of synthetic images to generate per category (default: 200)
- `--skip-corrosion`: Skip training for corrosion images
- `--skip-nocorrosion`: Skip training for no-corrosion images

## Output

After running the script, you'll find:

### Generated Files

1. **Models** (`GAN_Synthetic_Images/models/`):
   - `generator_corrosion.h5`
   - `discriminator_corrosion.h5`
   - `generator_nocorrosion.h5`
   - `discriminator_nocorrosion.h5`

2. **Synthetic Images**:
   - `GAN_Synthetic_Images/synthetic_corrosion/` - Generated corrosion images
   - `GAN_Synthetic_Images/synthetic_nocorrosion/` - Generated non-corrosion images

3. **Training Samples** (`GAN_Synthetic_Images/training_samples/`):
   - Sample images generated during training (every 10 epochs)
   - Useful for monitoring training progress

4. **Visualizations** (`GAN_Synthetic_Images/`):
   - `training_history_corrosion.png` - Loss and accuracy plots
   - `training_history_nocorrosion.png` - Loss and accuracy plots
   - `final_samples_corrosion.png` - Final generated samples
   - `final_samples_nocorrosion.png` - Final generated samples

## How It Works

### 1. Architecture

**Generator**: Transforms random noise (100-dim vector) into 128x128x3 images
- Dense layer → Reshape → Conv2DTranspose layers with BatchNorm and ReLU
- Output: tanh activation for pixel values in [-1, 1]

**Discriminator**: Classifies images as real or fake
- Conv2D layers with LeakyReLU and Dropout
- Output: sigmoid activation for binary classification

### 2. Training Process

- **Adversarial Training**: Generator and Discriminator compete
  - Discriminator learns to distinguish real vs fake images
  - Generator learns to create realistic images that fool the discriminator
  
- **Progress Monitoring**: 
  - Prints loss and accuracy every 10 epochs
  - Generates sample images every 10 epochs
  - Saves training history plots

### 3. Image Generation

After training:
- Generates specified number of synthetic images
- Saves as JPG files with sequential naming
- Images are ready to use for dataset augmentation

## Example Output

```
============================================================
GAN-BASED SYNTHETIC IMAGE GENERATOR
For Corrosion Detection Dataset Augmentation
============================================================

✓ Directories created successfully!

############################################################
# PHASE 1: CORROSION IMAGES
############################################################

============================================================
Loading images from: /path/to/data/CORROSION
============================================================
Found 1234 image files
Progress: 100/1234 images loaded...
Progress: 200/1234 images loaded...
...
✓ Successfully loaded 1234 images
✓ Image shape: (1234, 128, 128, 3)

Training GAN for CORROSION
Number of training images: 1234
Batch size: 32
Epochs: 100

Epoch    0/100 | D Loss: 0.6931 | D Acc: 50.00% | G Loss: 0.6931 | Time:    5.2s
Epoch   10/100 | D Loss: 0.5234 | D Acc: 73.44% | G Loss: 0.8123 | Time:   52.3s
...
```

## Tips for Best Results

1. **Training Time**: 
   - ~1-2 hours for 100 epochs on GPU
   - ~5-10 hours on CPU
   - Use GPU for faster training if available

2. **Monitoring Quality**:
   - Check sample images in `training_samples/` directory
   - Look at training history plots for convergence
   - Good GAN: D accuracy around 60-80%, losses stabilize

3. **Adjusting Parameters**:
   - If images look poor: increase epochs (150-200)
   - If out of memory: reduce batch size (16 or 8)
   - For more diversity: generate more images (500-1000)

4. **Using Synthetic Images**:
   - Mix with real images in training dataset
   - Ratio: 1:1 or 2:1 (real:synthetic) works well
   - Can improve model generalization

## Troubleshooting

**Out of Memory Error**:
```bash
python gan_synthetic_image_generator.py --batch-size 16
```

**Training Takes Too Long**:
```bash
python gan_synthetic_image_generator.py --epochs 50
```

**Want to Resume/Generate More Images**:
The script trains from scratch each time. To generate more images with existing models, you can modify the script or use the saved models separately.

## Next Steps

After generating synthetic images:

1. Review the quality in `final_samples_*.png`
2. Copy synthetic images to your training dataset
3. Retrain your CNN classifier with augmented dataset
4. Compare performance with/without synthetic data

---

**Author**: ME6324 Project
**Date**: October 29, 2025
