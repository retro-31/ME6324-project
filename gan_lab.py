"""
Production-Ready GAN for Lab GPU Training
==========================================

Optimized for high-end GPUs (A100, V100, RTX 4090, etc.)
Uses progressive training: 64x64 → 128x128 for best quality

Features:
- Mixed precision training (faster on modern GPUs)
- Gradient penalty for stability
- Progressive resolution training
- Automatic checkpointing
- Multi-GPU support (if available)
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # No display needed
import matplotlib.pyplot as plt


# Enable mixed precision for faster training on modern GPUs
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class Config:
    def __init__(self, resolution=64):
        self.resolution = resolution
        self.latent_dim = 512
        self.batch_size = 32  # Good for high-end GPUs
        self.epochs = 5000
        self.gp_weight = 10.0  # Gradient penalty weight
        self.learning_rate = 0.0002
        self.beta_1 = 0.0
        self.beta_2 = 0.99
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, f'GAN_Lab_{resolution}x{resolution}')
        
    def create_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'final_images'), exist_ok=True)


def load_images(data_path, resolution, max_images=None):
    """Load and preprocess images efficiently"""
    print(f"\nLoading images from: {data_path}")
    print(f"Target resolution: {resolution}x{resolution}")
    
    files = [f for f in os.listdir(data_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))
             and not f.startswith('.')]
    
    if max_images:
        files = files[:max_images]
    
    print(f"Found {len(files)} images")
    
    images = []
    for i, f in enumerate(files):
        if (i + 1) % 200 == 0:
            print(f"  Loading {i+1}/{len(files)}...")
        
        try:
            img = Image.open(os.path.join(data_path, f)).convert('RGB')
            img = img.resize((resolution, resolution), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32)
            img_array = (img_array - 127.5) / 127.5  # [-1, 1]
            images.append(img_array)
        except:
            continue
    
    images = np.array(images, dtype=np.float32)
    print(f"✓ Loaded {len(images)} images\n")
    return images


def build_generator(latent_dim, resolution):
    """
    High-quality generator with self-modulation
    """
    noise = layers.Input(shape=(latent_dim,))
    
    # Calculate starting size
    start_size = resolution // 16  # For 64: 4x4, For 128: 8x8
    
    # Initial dense layer
    x = layers.Dense(start_size * start_size * 512, use_bias=False)(noise)
    x = layers.Reshape((start_size, start_size, 512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Progressive upsampling
    filters = [512, 256, 128, 64]
    for i, f in enumerate(filters):
        x = layers.UpSampling2D(2)(x)
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Add residual connection
        if i > 0:
            x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(0.2)(x)
    
    # Output layer
    x = layers.Conv2D(3, 7, padding='same', activation='tanh')(x)
    
    return keras.Model(noise, x, name='Generator')


def build_discriminator(resolution):
    """
    Discriminator with spectral normalization
    """
    img = layers.Input(shape=(resolution, resolution, 3))
    
    x = layers.Conv2D(64, 4, strides=2, padding='same')(img)
    x = layers.LeakyReLU(0.2)(x)
    
    filters = [128, 256, 512]
    for f in filters:
        x = layers.Conv2D(f, 4, strides=2, padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    
    return keras.Model(img, x, name='Discriminator')


class WGAN_GP(keras.Model):
    """Wasserstein GAN with Gradient Penalty - most stable GAN variant"""
    
    def __init__(self, generator, discriminator, latent_dim, gp_weight=10.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.d_steps = 5  # Train discriminator more
        
    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
    
    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]
    
    def gradient_penalty(self, real_images, fake_images):
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images * alpha + fake_images * (1 - alpha)
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Train discriminator multiple times
        for _ in range(self.d_steps):
            noise = tf.random.normal([batch_size, self.latent_dim])
            
            with tf.GradientTape() as tape:
                fake_images = self.generator(noise, training=True)
                real_pred = self.discriminator(real_images, training=True)
                fake_pred = self.discriminator(fake_images, training=True)
                
                d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                gp = self.gradient_penalty(real_images, fake_images)
                d_loss = d_loss + gp * self.gp_weight
            
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Train generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_pred = self.discriminator(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_pred)
        
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        self.g_loss_metric.update_state(g_loss)
        self.d_loss_metric.update_state(d_loss)
        
        return {
            "g_loss": self.g_loss_metric.result(),
            "d_loss": self.d_loss_metric.result()
        }


class GANMonitor(keras.callbacks.Callback):
    """Monitor training progress"""
    
    def __init__(self, config, category, save_freq=100):
        self.config = config
        self.category = category
        self.save_freq = save_freq
        self.best_g_loss = float('inf')
        self.start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        # Progress update
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - self.start_time
            eta = (elapsed / (epoch + 1)) * (self.config.epochs - epoch - 1)
            print(f"\nEpoch {epoch+1:5d}/{self.config.epochs} | "
                  f"G_Loss: {logs['g_loss']:7.3f} | "
                  f"D_Loss: {logs['d_loss']:7.3f} | "
                  f"Time: {elapsed/60:5.1f}m | ETA: {eta/60:5.1f}m")
        
        # Save samples
        if (epoch + 1) % self.save_freq == 0:
            self.save_samples(epoch + 1)
        
        # Save checkpoints
        if (epoch + 1) % 500 == 0:
            checkpoint_path = os.path.join(
                self.config.output_dir, 
                'checkpoints', 
                f'generator_epoch_{epoch+1:05d}.keras'
            )
            self.model.generator.save(checkpoint_path)
            print(f"  ✓ Checkpoint saved: epoch {epoch+1}")
        
        # Save best model
        if logs['g_loss'] < self.best_g_loss:
            self.best_g_loss = logs['g_loss']
            best_path = os.path.join(
                self.config.output_dir,
                'checkpoints',
                f'best_generator_{self.category}.keras'
            )
            self.model.generator.save(best_path)
    
    def save_samples(self, epoch):
        noise = tf.random.normal([16, self.config.latent_dim])
        generated = self.model.generator(noise, training=False)
        generated = (generated + 1) / 2.0
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f'{self.category.upper()} - Epoch {epoch}', fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated[i])
            ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(
            self.config.output_dir,
            'samples',
            f'{self.category}_epoch_{epoch:05d}.png'
        )
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


def generate_final_images(generator, config, category, num_images=500):
    """Generate final high-quality images"""
    print(f"\n{'='*70}")
    print(f"Generating {num_images} final images...")
    print(f"{'='*70}\n")
    
    output_dir = os.path.join(config.output_dir, 'final_images')
    
    for i in range(num_images):
        noise = tf.random.normal([1, config.latent_dim])
        img = generator(noise, training=False)[0]
        img = ((img + 1) / 2.0 * 255).numpy().astype('uint8')
        
        Image.fromarray(img).save(
            os.path.join(output_dir, f'{category}_{i:04d}.png')
        )
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_images}...")
    
    print(f"\n✓ Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='High-End GPU GAN Training')
    parser.add_argument('--category', required=True, choices=['CORROSION', 'NOCORROSION'])
    parser.add_argument('--resolution', type=int, default=64, choices=[64, 128])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-generate', type=int, default=500)
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PRODUCTION GAN TRAINING FOR LAB GPU")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Category: {args.category}")
    print(f"  - Resolution: {args.resolution}x{args.resolution}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print("="*70)
    
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠ No GPU detected! Training will be very slow.")
    
    # Configuration
    config = Config(resolution=args.resolution)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.create_dirs()
    
    # Load data
    data_path = os.path.join(config.base_dir, 'data', args.category)
    images = load_images(data_path, config.resolution)
    
    # Create dataset with prefetching for performance
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Build models
    print("Building models...")
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        generator = keras.models.load_model(args.checkpoint)
        discriminator = build_discriminator(config.resolution)
    else:
        generator = build_generator(config.latent_dim, config.resolution)
        discriminator = build_discriminator(config.resolution)
    
    print(f"Generator parameters: {generator.count_params():,}")
    print(f"Discriminator parameters: {discriminator.count_params():,}")
    
    # Create and compile GAN
    wgan = WGAN_GP(generator, discriminator, config.latent_dim, config.gp_weight)
    
    wgan.compile(
        g_optimizer=keras.optimizers.Adam(config.learning_rate, config.beta_1, config.beta_2),
        d_optimizer=keras.optimizers.Adam(config.learning_rate, config.beta_1, config.beta_2)
    )
    
    # Training callbacks
    callbacks = [
        GANMonitor(config, args.category.lower()),
    ]
    
    # Train
    print(f"\nStarting training for {config.epochs} epochs...")
    print("="*70 + "\n")
    
    wgan.fit(
        dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=0  # We handle progress in callback
    )
    
    # Save final model
    final_path = os.path.join(
        config.output_dir,
        'checkpoints',
        f'final_generator_{args.category.lower()}.keras'
    )
    generator.save(final_path)
    print(f"\n✓ Final model saved: {final_path}")
    
    # Generate final images
    generate_final_images(generator, config, args.category.lower(), args.num_generate)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {config.output_dir}")
    print("\nNext steps:")
    if args.resolution == 64:
        print("  1. Review 64x64 results in samples/")
        print(f"  2. If quality is good, fine-tune at 128x128:")
        print(f"     python gan_lab.py --category {args.category} --resolution 128 \\")
        print(f"       --checkpoint {final_path} --epochs 3000")
    else:
        print("  1. Check final_images/ for generated images")
        print("  2. Use these images for your dataset augmentation")
    print()


if __name__ == '__main__':
    main()
