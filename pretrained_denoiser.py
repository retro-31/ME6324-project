import os
import cv2
import time
import torch
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR = "./data"                     # Folder with original noisy images
OUTPUT_DIR = "./data/denoised_images"    # Folder to save enhanced images
MODEL_PATH = "./weights/RealESRGAN_x4plus.pth"  # Path to pretrained model
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("weights", exist_ok=True)

# ============================================================
# GPU / DEVICE SETUP
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ============================================================
# DOWNLOAD / VALIDATE MODEL FILE
# ============================================================
def validate_or_download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10_000_000:
        print("📥 Downloading pretrained Real-ESRGAN model...")
        os.system(f"wget -q {MODEL_URL} -O {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10_000_000:
            raise RuntimeError("❌ Model download failed or file is corrupted.")
    print("✅ Pretrained model ready!")

validate_or_download_model()

# ============================================================
# MODEL INITIALIZATION
# ============================================================
model = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23,
    num_grow_ch=32, scale=4
)

denoiser = RealESRGANer(
    scale=4,
    model_path=MODEL_PATH,
    model=model,
    tile=400,          # reduce this if you hit OOM
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available(),  # use float16 on GPU
    device=device
)

print("✅ RealESRGAN model loaded successfully!")

# ============================================================
# PROCESS IMAGES
# ============================================================
def get_all_images(root_dir):
    """Recursively collect all .jpg/.png/.jpeg images."""
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, f))
    return image_paths

image_list = get_all_images(INPUT_DIR)
print(f"🧾 Found {len(image_list)} images to enhance in '{INPUT_DIR}'.")

start_time = time.time()

for img_path in tqdm(image_list, desc="Enhancing images"):
    rel_path = os.path.relpath(img_path, INPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ Skipping {img_path}, cannot open.")
        continue

    try:
        output, _ = denoiser.enhance(img, outscale=1)
        cv2.imwrite(out_path, output)
    except RuntimeError as e:
        print(f"❌ GPU memory issue on {img_path}: {e}")
        print("➡️ Retrying with smaller tile size...")
        denoiser.tile = 200
        output, _ = denoiser.enhance(img, outscale=1)
        cv2.imwrite(out_path, output)
    except Exception as e:
        print(f"❌ Error enhancing {img_path}: {e}")

elapsed = time.time() - start_time
print(f"\n✅ Denoising/enhancement completed in {elapsed/60:.2f} minutes.")
print(f"📁 Enhanced images saved in: {OUTPUT_DIR}")
