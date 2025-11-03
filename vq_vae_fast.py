import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
from tqdm import tqdm

# ===============================
# CONFIGURATION
# ===============================
DATA_DIR = "dataset"   # <-- inside your project folder
OUT_DIR = "vqvae_outputs"
RESOLUTION = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# MODEL DEFINITIONS
# ===============================

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, latent_dim, 3, 1, 1),
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_channels=128, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 3, 1, 1),
            nn.Tanh(),
        )
    def forward(self, z): return self.net(z)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        distances = (
            flat_input.pow(2).sum(1, keepdim=True)
            - 2 * flat_input @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )
        indices = torch.argmin(distances, 1)
        quantized = self.embeddings(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        e_latent_loss = (quantized.detach() - inputs).pow(2).mean()
        q_latent_loss = (quantized - inputs.detach()).pow(2).mean()
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss

# ===============================
# DATASET
# ===============================
transform = transforms.Compose([
    transforms.Resize((RESOLUTION, RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ===============================
# INITIALIZE MODEL
# ===============================
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
quantizer = VectorQuantizer().to(DEVICE)

params = list(encoder.parameters()) + list(decoder.parameters()) + list(quantizer.parameters())
optimizer = torch.optim.Adam(params, lr=LR)

os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# TRAINING LOOP
# ===============================
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        z_e = encoder(imgs)
        z_q, q_loss = quantizer(z_e)
        recon = decoder(z_q)
        recon_loss = torch.mean((imgs - recon) ** 2)
        loss = recon_loss + q_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")

    # Save sample reconstructions
    with torch.no_grad():
        sample = recon[:16]
        utils.save_image((sample * 0.5 + 0.5), f"{OUT_DIR}/epoch_{epoch+1}_samples.png")

    # Save checkpoint
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'quantizer': quantizer.state_dict()
    }, f"{OUT_DIR}/vqvae_epoch_{epoch+1}.pth")

print("✅ Training Complete! Generated images saved in:", OUT_DIR)
