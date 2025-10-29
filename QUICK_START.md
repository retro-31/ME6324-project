# GAN Training - Quick Reference

## 🎯 For Lab GPU (High-End GPU Available)

### Before You Start
```bash
./test_lab_setup.sh  # Test GPU setup (2 min)
```

### Basic Training (64x64, Good Quality)
```bash
# CORROSION - overnight run (~4-6 hours)
nohup python gan_lab.py --category CORROSION --resolution 64 \
  --epochs 5000 --batch-size 32 > corrosion.log 2>&1 &

# NOCORROSION - overnight run (~4-6 hours)
nohup python gan_lab.py --category NOCORROSION --resolution 64 \
  --epochs 5000 --batch-size 32 > nocorrosion.log 2>&1 &
```

### High Quality (128x128, Best Results)
```bash
# After 64x64 completes, fine-tune at higher resolution
python gan_lab.py --category CORROSION --resolution 128 \
  --epochs 3000 --batch-size 16 \
  --checkpoint GAN_Lab_64x64/checkpoints/best_generator_corrosion.keras
```

### Monitor Progress
```bash
tail -f corrosion.log                    # Watch training
ls -lht GAN_Lab_64x64/samples/ | head   # Check samples
```

---

## ⚡ Traditional Augmentation (Any Computer)

### Fastest Way - No GPU Needed!
```bash
# CORROSION (already done - 2,970 images generated)
# NOCORROSION (need to run - 2 minutes)
python augment_dataset.py --source data/NOCORROSION \
  --output augmented_data/NOCORROSION --num-per-image 3
```

**Result**: 6,000+ training images ready in 2 minutes!

---

## 📁 Output Locations

### GAN Output
```
GAN_Lab_64x64/final_images/     ← Generated synthetic images
GAN_Lab_64x64/samples/          ← Training progress samples
GAN_Lab_64x64/checkpoints/      ← Saved models
```

### Traditional Augmentation Output
```
augmented_data/CORROSION/       ← 2,970 images ✓
augmented_data/NOCORROSION/     ← ~2,500 images (run script)
```

---

## 🚨 Quick Troubleshooting

### GPU Not Detected
```bash
nvidia-smi  # Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Out of Memory
```bash
# Reduce batch size
python gan_lab.py --category CORROSION --resolution 64 \
  --epochs 5000 --batch-size 8  # <-- smaller batch
```

### Training Too Slow
```bash
# Check GPU usage
nvidia-smi
# Should show python process using GPU
```

---

## ⏱️ Time Estimates

| Method | Time | Quality | Difficulty |
|--------|------|---------|-----------|
| Traditional Augmentation | 2 min | ★★★★☆ | Easy |
| GAN 64x64 | 4-6 hrs | ★★★★☆ | Medium |
| GAN 128x128 | 10-14 hrs | ★★★★★ | Medium |

---

## ✅ My Recommendation

**Best for most cases:**
```bash
python augment_dataset.py --source data/NOCORROSION \
  --output augmented_data/NOCORROSION --num-per-image 3
```
Then use `augmented_data/` for your classifier training.

**Only use GAN if:**
- You have 2-3 days
- You want GAN experience
- Required for publication

---

See `LAB_GPU_SUMMARY.md` and `GAN_LAB_TRAINING_GUIDE.md` for details.
