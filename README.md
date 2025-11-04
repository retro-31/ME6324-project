# 🧠 AI-Based Corrosion Detection using Denoising and Transfer Learning

This project explores deep learning–based corrosion detection using both **pretrained models** (VGG16, ResNet50, MobileNetV2) and **custom CNN architectures** (CNN_A, CNN_B, CNN_C). The study focuses on the impact of image denoising and preprocessing on model performance.

## 🚀 Project Overview

### 🧩 Objective
To develop a robust corrosion detection pipeline that enhances image quality through denoising and leverages deep learning models for accurate classification between **CORROSION** and **NOCORROSION** surfaces.

### 🧱 Pipeline Summary
1. **Image Preprocessing**
   - Random cropping (200×200 patch → resize to 224×224)
   - Random rotation (+30° using bilinear interpolation)
   - Horizontal flipping
   - Gaussian noise injection (mean=0, std=15)
   - Denoising using **Real-ESRGAN** and **Autoencoder**
   - Brightness & contrast enhancement

2. **Model Development**
   - **Transfer Learning Models:** VGG16, ResNet50, MobileNetV2 (fine-tuned for binary classification)
   - **Custom CNN Architectures:** CNN_A, CNN_B, CNN_C with Dropout and L2 regularization
   - **Comparison:** Original vs. Denoised image datasets

3. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, and F1-score
   - Confusion Matrix visualization

## 🧠 Denoising Techniques

### 1️⃣ Real-ESRGAN (Pretrained)
- Uses **Residual-in-Residual Dense Blocks (RRDB)** for stable feature extraction.
- Effective in removing noise and restoring high-frequency details.
- Outperformed Autoencoder with sharper, artifact-free outputs.

### 2️⃣ Autoencoder-based Denoiser
- Encoder–decoder structure for reconstructing clean images from noisy ones.
- Produced smoother but slightly dotted outputs due to underfitting.
- Future improvements may include deeper networks or skip connections.

## ⚙️ Environment Setup

### 🧰 Requirements
- Python 3.9+
- CUDA-enabled GPU (recommended)
- PyTorch ≥ 1.12
- Torchvision ≥ 0.13
- OpenCV, tqdm, scikit-learn, matplotlib

### 🧑‍💻 Setup Instructions
```bash
git clone https://github.com/retro-31/ME6324-project.git
cd ME6324-project

conda create -n corrosion python=3.9
conda activate corrosion

pip install -r requirements.txt
```

## 🧪 Training Scripts

### 🔹 Pretrained Models
```bash
python transfer_learning.py
```

### 🔹 Custom CNNs
```bash
python custom_architecture_cnn.py
```

### 🔹 Denoising
```bash
python pretrained_denoiser.py
python autoencoder_denoiser.py
```

## 📈 Evaluation

### Pretrained Models
```bash
python evaluate_pretrained.py
```

### Custom CNN Models
```bash
python evaluate_custom_cnn.py
```

Both generate:
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- Classification Report  

## 🧩 Results Summary

| Model Type | Dataset | Best Accuracy (%) |
|------------|----------|------------------|
| VGG16 | Denoised | 93.5 |
| ResNet50 | Denoised | 91.2 |
| MobileNetV2 | Denoised | 90.7 |
| CNN_A | Denoised | **94.2** |
| CNN_B | Denoised | 92.8 |
| CNN_C | Denoised | 91.4 |

✅ **CNN_A achieved the highest accuracy**, outperforming all pretrained models.

## ⚙️ Technical Insights

- **BCEWithLogitsLoss** combines sigmoid + binary cross-entropy for stable training.
- **Dropout** randomly disables neurons per batch to prevent overfitting.
- **Pruning** permanently removes low-importance weights to shrink model size.
- **MobileNetV2** uses inverted residual blocks + depthwise separable convolution for high efficiency.

## 🧊 Key Observations

1. Denoised images significantly improved accuracy.
2. Real-ESRGAN produced the best denoising outputs.
3. Custom CNN_A outperformed even pretrained models—small, efficient, highly generalizable.
4. Transfer learning benefited most from enhanced image clarity.
5. Regularization helped mitigate overfitting.

## 🏁 Future Work

- GAN-based synthetic augmentation (StyleGAN)
- Improved Autoencoder (U-Net, skip connections)
- Deployment as a web API
---

### ✅ Note
Large pretrained weights (>100MB) are handled through **Git LFS**.
