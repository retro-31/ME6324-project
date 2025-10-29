#!/bin/bash

# Quick Start Script for GAN Synthetic Image Generator
# This script runs the GAN with recommended settings

echo "=========================================="
echo "GAN Synthetic Image Generator - Quick Start"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null
then
    echo "Error: Python3 not found. Please install Python 3.x"
    exit 1
fi

# Check if the data directories exist
if [ ! -d "data/CORROSION" ]; then
    echo "Warning: data/CORROSION directory not found!"
    echo "Please ensure your data is in the correct location."
fi

if [ ! -d "data/NOCORROSION" ]; then
    echo "Warning: data/NOCORROSION directory not found!"
    echo "Please ensure your data is in the correct location."
fi

echo ""
echo "Running GAN training with the following settings:"
echo "  - Epochs: 100"
echo "  - Batch size: 32"
echo "  - Synthetic images per category: 200"
echo ""
echo "This will take approximately 1-2 hours on GPU or 5-10 hours on CPU."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run the GAN script
python3 gan_synthetic_image_generator.py \
    --epochs 100 \
    --batch-size 32 \
    --num-synthetic 200

echo ""
echo "=========================================="
echo "GAN Training Complete!"
echo "=========================================="
echo ""
echo "Check the GAN_Synthetic_Images/ directory for:"
echo "  - Trained models"
echo "  - Synthetic images"
echo "  - Training visualizations"
echo ""
