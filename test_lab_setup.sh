#!/bin/bash
#
# Quick test script for lab GPU
# Run this before starting full training to verify setup
#

echo "=========================================="
echo "GAN Lab GPU Test"
echo "=========================================="
echo ""

# Test 1: Python and TensorFlow
echo "Test 1: Checking Python and TensorFlow..."
python3 -c "
import tensorflow as tf
print(f'  ✓ Python: OK')
print(f'  ✓ TensorFlow version: {tf.__version__}')
"

if [ $? -ne 0 ]; then
    echo "  ✗ TensorFlow not installed!"
    echo "  Install with: pip install tensorflow[and-cuda]"
    exit 1
fi

# Test 2: GPU Detection
echo ""
echo "Test 2: Checking GPU..."
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  ✓ Found {len(gpus)} GPU(s):')
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        print(f'    GPU {i}: {details.get(\"device_name\", \"Unknown\")}')
        print(f'    Compute: {details.get(\"compute_capability\", \"Unknown\")}')
else:
    print('  ✗ No GPU detected!')
    print('  Training will be VERY slow on CPU.')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Test 3: Mixed Precision Support
echo ""
echo "Test 3: Checking mixed precision support..."
python3 -c "
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f'  ✓ Mixed precision: {policy.name}')
print(f'  ✓ Compute dtype: {policy.compute_dtype}')
print(f'  ✓ Variable dtype: {policy.variable_dtype}')
"

# Test 4: Data Directory
echo ""
echo "Test 4: Checking data directories..."
if [ -d "data/CORROSION" ]; then
    count=$(ls data/CORROSION/*.jpg 2>/dev/null | wc -l)
    echo "  ✓ CORROSION: $count images found"
else
    echo "  ✗ data/CORROSION directory not found!"
    exit 1
fi

if [ -d "data/NOCORROSION" ]; then
    count=$(ls data/NOCORROSION/*.jpg 2>/dev/null | wc -l)
    echo "  ✓ NOCORROSION: $count images found"
else
    echo "  ✗ data/NOCORROSION directory not found!"
    exit 1
fi

# Test 5: Quick training test
echo ""
echo "Test 5: Running quick training test (10 epochs)..."
echo "  This will take 1-2 minutes..."

python3 gan_lab.py --category CORROSION --resolution 64 --epochs 10 --batch-size 16 --num-generate 5 > /tmp/gan_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Training test passed!"
    echo "  ✓ Check /tmp/gan_test.log for details"
    
    # Show generated samples
    if [ -d "GAN_Lab_64x64/final_images" ]; then
        count=$(ls GAN_Lab_64x64/final_images/*.png 2>/dev/null | wc -l)
        echo "  ✓ Generated $count test images"
    fi
else
    echo "  ✗ Training test failed!"
    echo "  See /tmp/gan_test.log for error details"
    tail -20 /tmp/gan_test.log
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "✓ ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "Your lab GPU is ready for GAN training!"
echo ""
echo "Recommended next steps:"
echo "  1. For overnight run (64x64):"
echo "     nohup python gan_lab.py --category CORROSION --resolution 64 --epochs 5000 > train.log 2>&1 &"
echo ""
echo "  2. Monitor progress:"
echo "     tail -f train.log"
echo "     ls -lht GAN_Lab_64x64/samples/ | head"
echo ""
echo "  3. For best quality (after 64x64 completes):"
echo "     python gan_lab.py --category CORROSION --resolution 128 --epochs 3000 \\"
echo "       --checkpoint GAN_Lab_64x64/checkpoints/best_generator_corrosion.keras"
echo ""
