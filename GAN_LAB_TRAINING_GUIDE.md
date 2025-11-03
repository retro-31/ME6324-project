# GAN Training Guide for Lab GPU

## 🚀 Quick Start (For Lab GPU)

### Step 1: Train at 64x64 (Faster, Learn Structure)
```bash
# CORROSION category
python gan_lab.py --category CORROSION --resolution 64 --epochs 5000 --batch-size 32

# NOCORROSION category  
python gan_lab.py --category NOCORROSION --resolution 64 --epochs 5000 --batch-size 32
```

**Time**: ~30-45 minutes on A100/V100, ~1-2 hours on RTX 4090

### Step 2: Fine-tune at 128x128 (Better Quality)
```bash
# Use the best 64x64 model as starting point
python gan_lab.py --category CORROSION --resolution 128 --epochs 3000 \
  --checkpoint GAN_Lab_64x64/checkpoints/best_generator_corrosion.keras

python gan_lab.py --category NOCORROSION --resolution 128 --epochs 3000 \
  --checkpoint GAN_Lab_64x64/checkpoints/best_generator_nocorrosion.keras
```

**Time**: ~1-2 hours on A100/V100, ~3-4 hours on RTX 4090

---

## 📋 Features

✅ **Optimized for High-End GPUs**
- Mixed precision training (2x faster on modern GPUs)
- Multi-GPU support (automatic if multiple GPUs available)
- Memory-efficient gradient checkpointing
- Prefetched data pipeline

✅ **Stable Training**
- WGAN-GP (Wasserstein GAN with Gradient Penalty)
- No mode collapse issues
- Consistent quality improvements

✅ **Progressive Training**
- Start at 64x64 for fast learning
- Fine-tune at 128x128 for quality
- Transfer learning from lower resolution

✅ **Automatic Checkpointing**
- Saves every 500 epochs
- Keeps best model based on generator loss
- Resume from any checkpoint

---

## 🎯 Recommended Training Schedule

### For Quick Results (Overnight Run)
```bash
# 64x64 only - good enough for most purposes
python gan_lab.py --category CORROSION --resolution 64 --epochs 5000 --num-generate 1000
```

### For Best Quality (Weekend Run)
```bash
# Day 1: Train 64x64 (8 hours)
python gan_lab.py --category CORROSION --resolution 64 --epochs 8000 --batch-size 32

# Day 2: Fine-tune 128x128 (12 hours)
python gan_lab.py --category CORROSION --resolution 128 --epochs 5000 \
  --checkpoint GAN_Lab_64x64/checkpoints/best_generator_corrosion.keras
```

---

## 📊 Expected Results

### 64x64 Resolution
- **Epoch 1000**: Basic shapes and colors visible
- **Epoch 3000**: Clear corrosion patterns forming
- **Epoch 5000**: Good quality, suitable for augmentation

### 128x128 Resolution  
- **Epoch 1000**: Fine details emerging (when starting from 64x64 checkpoint)
- **Epoch 3000**: High-quality realistic images
- **Epoch 5000**: Publication-quality synthetic images

---

## 💾 Output Structure

```
GAN_Lab_64x64/
├── checkpoints/
│   ├── generator_epoch_00500.keras
│   ├── generator_epoch_01000.keras
│   ├── best_generator_corrosion.keras  ← Use this for fine-tuning
│   └── final_generator_corrosion.keras
├── samples/
│   ├── corrosion_epoch_00100.png
│   ├── corrosion_epoch_00200.png
│   └── ...
└── final_images/
    ├── corrosion_0000.png
    ├── corrosion_0001.png
    └── ... (500 images)

GAN_Lab_128x128/
└── [same structure]
```

---

## ⚙️ Advanced Options

### Adjust Batch Size for Your GPU
```bash
# For 8GB GPU (RTX 3070, RTX 4060 Ti)
python gan_lab.py --category CORROSION --resolution 64 --batch-size 16

# For 16GB GPU (RTX 4080, A10)
python gan_lab.py --category CORROSION --resolution 64 --batch-size 64

# For 24GB+ GPU (RTX 4090, A100, V100)
python gan_lab.py --category CORROSION --resolution 64 --batch-size 128
```

### Generate More Images
```bash
# Generate 2000 images instead of default 500
python gan_lab.py --category CORROSION --resolution 64 --epochs 5000 --num-generate 2000
```

### Resume Interrupted Training
```bash
# Resume from specific epoch
python gan_lab.py --category CORROSION --resolution 64 --epochs 8000 \
  --checkpoint GAN_Lab_64x64/checkpoints/generator_epoch_03000.keras
```

---

## 🔧 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python gan_lab.py --category CORROSION --resolution 64 --batch-size 8
```

### Training Too Slow
- Check GPU utilization: `nvidia-smi`
- Ensure TensorFlow is using GPU: Check initial output
- Try reducing resolution or batch size

### Poor Quality Images
- Train longer (try 8000-10000 epochs at 64x64)
- Check discriminator isn't too strong (D_Loss should be ~0 to -50)
- Try starting fresh if mode collapse detected

---

## 📈 Monitoring Training

### Check Progress During Training
```bash
# In another terminal, monitor samples
ls -lht GAN_Lab_64x64/samples/ | head -20

# View latest sample (requires X11 forwarding or copy to local)
display GAN_Lab_64x64/samples/corrosion_epoch_05000.png
```

### Expected Loss Values
- **D_Loss**: -10 to -50 (negative is normal for WGAN-GP)
- **G_Loss**: -5 to -20 (should decrease over time)

If G_Loss increases consistently, try lowering learning rate or increasing discriminator training steps.

---

## 🎓 Training Tips

1. **Start with 64x64**: Much faster to train and easier to debug
2. **Monitor samples**: Check every 500-1000 epochs to ensure quality improves
3. **Use best checkpoint**: Don't always use final model - best might be from epoch 4000
4. **Progressive training works better**: 64x64 → 128x128 gives better results than direct 128x128
5. **Patience pays off**: GAN training is slow - 5000+ epochs is normal for good results

---

## 🚀 Production Workflow

```bash
# 1. Train base model overnight (8 hours)
nohup python gan_lab.py --category CORROSION --resolution 64 \
  --epochs 8000 --batch-size 32 > corrosion_64_train.log 2>&1 &

# 2. Check results next morning
tail -f corrosion_64_train.log
ls GAN_Lab_64x64/samples/ | tail -10

# 3. Fine-tune if quality is good
nohup python gan_lab.py --category CORROSION --resolution 128 \
  --epochs 5000 --checkpoint GAN_Lab_64x64/checkpoints/best_generator_corrosion.keras \
  > corrosion_128_train.log 2>&1 &

# 4. Generate production dataset
# Final images will be in GAN_Lab_128x128/final_images/
```

---

## 📦 Hardware Requirements

| GPU | VRAM | Max Batch (64x64) | Max Batch (128x128) | Training Time (5000 epochs) |
|-----|------|-------------------|---------------------|----------------------------|
| RTX 3060 | 12GB | 32 | 8 | 2-3 hours / 6-8 hours |
| RTX 4070 | 12GB | 64 | 16 | 1-2 hours / 4-6 hours |
| RTX 4080 | 16GB | 128 | 32 | 45-60 min / 2-3 hours |
| RTX 4090 | 24GB | 256 | 64 | 30-40 min / 1-2 hours |
| A100 | 40GB | 512 | 128 | 20-25 min / 45-60 min |
| A100 | 80GB | 512 | 256 | 15-20 min / 30-45 min |

---

## ✅ Quality Checklist

Before using generated images:

- [ ] Samples show clear improvement from epoch 1000 to 5000
- [ ] Final images look realistic (not noisy or garbled)
- [ ] Variety in generated images (not all identical)
- [ ] No obvious artifacts or mode collapse
- [ ] Images match the style of training data

If any checkbox fails, train longer or adjust hyperparameters!

---

## 🎯 Expected Output

After successful training, you'll have:
- **500-2000 synthetic images** in `final_images/`
- **Checkpoints** for future fine-tuning
- **Sample grids** showing training progression
- **Ready-to-use images** for dataset augmentation

Total training time: 4-8 hours for both categories (64x64 + 128x128)
