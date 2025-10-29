"""
GAN-Based Synthetic Image Generator for Corrosion Detection
============================================================

This script implements a Deep Convolutional GAN (DCGAN) to generate synthetic images
for both CORROSION and NOCORROSION categories to augment the training dataset.

Author: Generated for ME6324 Project
Date: October 29, 2025
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# ============================================================================
# CONFIGURATION
# ============================================================================

class GANConfig:
    """Configuration class for GAN hyperparameters"""
    
    def __init__(self):
        # Image parameters
        self.img_height = 128
        self.img_width = 128
        self.img_channels = 3
        
        # GAN parameters
        self.latent_dim = 100  # Dimension of the noise vector
        self.batch_size = 32
        self.epochs = 100
        self.sample_interval = 10  # Generate samples every N epochs
        self.learning_rate = 0.0002
        self.beta_1 = 0.5  # Adam optimizer parameter
        
        # Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.corrosion_data_path = os.path.join(self.base_dir, 'data', 'CORROSION')
        self.nocorrosion_data_path = os.path.join(self.base_dir, 'data', 'NOCORROSION')
        self.output_dir = os.path.join(self.base_dir, 'GAN_Synthetic_Images')
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.synthetic_corrosion_dir = os.path.join(self.output_dir, 'synthetic_corrosion')
        self.synthetic_nocorrosion_dir = os.path.join(self.output_dir, 'synthetic_nocorrosion')
        self.samples_dir = os.path.join(self.output_dir, 'training_samples')
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.output_dir, 
            self.models_dir, 
            self.synthetic_corrosion_dir, 
            self.synthetic_nocorrosion_dir,
            self.samples_dir
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("✓ Directories created successfully!")


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Load and preprocess images for GAN training"""
    
    def __init__(self, config):
        self.config = config
        
    def load_images(self, data_path):
        """
        Load images from directory and preprocess them
        
        Args:
            data_path: Path to image directory
            
        Returns:
            Normalized numpy array of images
        """
        print(f"\n{'='*60}")
        print(f"Loading images from: {data_path}")
        print(f"{'='*60}")
        
        # Get list of image files (excluding hidden files)
        image_files = [
            f for f in os.listdir(data_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
            and not f.startswith('.')
        ]
        
        print(f"Found {len(image_files)} image files")
        
        images = []
        failed_count = 0
        
        for idx, img_file in enumerate(image_files):
            if (idx + 1) % 100 == 0:
                print(f"Progress: {idx + 1}/{len(image_files)} images loaded...")
                
            img_path = os.path.join(data_path, img_file)
            try:
                img = load_img(
                    img_path, 
                    target_size=(self.config.img_height, self.config.img_width)
                )
                img_array = img_to_array(img)
                images.append(img_array)
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Show only first 5 errors
                    print(f"Warning: Error loading {img_file}: {e}")
                continue
        
        if failed_count > 0:
            print(f"⚠ Failed to load {failed_count} images")
        
        images = np.array(images)
        
        # Normalize to [-1, 1] range (required for GAN with tanh activation)
        images = (images - 127.5) / 127.5
        
        print(f"✓ Successfully loaded {len(images)} images")
        print(f"✓ Image shape: {images.shape}")
        print(f"✓ Value range: [{images.min():.2f}, {images.max():.2f}]")
        
        return images


# ============================================================================
# GENERATOR NETWORK
# ============================================================================

def build_generator(config):
    """
    Build the Generator network
    
    The generator takes random noise as input and generates synthetic images.
    Architecture: Dense -> Reshape -> Conv2DTranspose layers with BatchNorm and ReLU
    
    Args:
        config: GANConfig object
        
    Returns:
        Keras Model for the generator
    """
    # Input: random noise vector
    noise_input = Input(shape=(config.latent_dim,), name='noise_input')
    
    # Dense layer to expand the noise
    x = Dense(16 * 16 * 256, activation='relu')(noise_input)
    x = BatchNormalization(momentum=0.8)(x)
    x = Reshape((16, 16, 256))(x)
    
    # Upsample to 32x32
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    
    # Upsample to 64x64
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    
    # Upsample to 128x128
    x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation('relu')(x)
    
    # Output layer: 128x128x3 image with tanh activation (range: -1 to 1)
    output = Conv2D(config.img_channels, kernel_size=3, padding='same', activation='tanh', name='generated_image')(x)
    
    model = Model(noise_input, output, name='Generator')
    return model


# ============================================================================
# DISCRIMINATOR NETWORK
# ============================================================================

def build_discriminator(config):
    """
    Build the Discriminator network
    
    The discriminator classifies images as real or fake.
    Architecture: Conv2D layers with LeakyReLU -> Dense output
    
    Args:
        config: GANConfig object
        
    Returns:
        Keras Model for the discriminator
    """
    # Input: image
    img_input = Input(
        shape=(config.img_height, config.img_width, config.img_channels),
        name='image_input'
    )
    
    # Downsample to 64x64
    x = Conv2D(32, kernel_size=4, strides=2, padding='same')(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    # Downsample to 32x32
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    # Downsample to 16x16
    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    # Downsample to 8x8
    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    
    # Flatten and classify
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid', name='validity')(x)
    
    model = Model(img_input, output, name='Discriminator')
    return model


# ============================================================================
# DCGAN CLASS
# ============================================================================

class DCGAN:
    """Deep Convolutional GAN for image generation"""
    
    def __init__(self, config, generator, discriminator):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        
        # Compile discriminator
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(config.learning_rate, config.beta_1),
            metrics=['accuracy']
        )
        
        # Build combined model (generator + discriminator)
        # For training the generator, we freeze the discriminator
        self.discriminator.trainable = False
        
        # Generator input
        noise = Input(shape=(config.latent_dim,))
        generated_img = self.generator(noise)
        
        # Discriminator classifies generated image
        validity = self.discriminator(generated_img)
        
        # Combined model
        self.combined = Model(noise, validity, name='DCGAN')
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=Adam(config.learning_rate, config.beta_1)
        )
        
        # History tracking
        self.history = {
            'd_loss': [],
            'd_acc': [],
            'g_loss': []
        }
    
    def train(self, images, category_name):
        """
        Train the GAN
        
        Args:
            images: Numpy array of training images
            category_name: Name of the category (for saving outputs)
        """
        print(f"\n{'='*60}")
        print(f"Training GAN for {category_name.upper()}")
        print(f"{'='*60}")
        print(f"Number of training images: {len(images)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")
        print(f"{'='*60}\n")
        
        # Labels for real and fake images
        real_labels = np.ones((self.config.batch_size, 1))
        fake_labels = np.zeros((self.config.batch_size, 1))
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of real images
            idx = np.random.randint(0, images.shape[0], self.config.batch_size)
            real_imgs = images[idx]
            
            # Generate fake images
            noise = np.random.normal(0, 1, (self.config.batch_size, self.config.latent_dim))
            fake_imgs = self.generator.predict(noise, verbose=0)
            
            # Train discriminator on real and fake images
            d_loss_real = self.discriminator.train_on_batch(real_imgs, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            noise = np.random.normal(0, 1, (self.config.batch_size, self.config.latent_dim))
            
            # Train generator (wants discriminator to classify fake images as real)
            g_loss = self.combined.train_on_batch(noise, real_labels)
            
            # Store history
            self.history['d_loss'].append(d_loss[0])
            self.history['d_acc'].append(d_loss[1])
            self.history['g_loss'].append(g_loss)
            
            # Print progress
            if epoch % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch:4d}/{self.config.epochs} | "
                      f"D Loss: {d_loss[0]:6.4f} | D Acc: {100*d_loss[1]:5.2f}% | "
                      f"G Loss: {g_loss:6.4f} | Time: {elapsed_time:6.1f}s")
            
            # Generate sample images at intervals
            if epoch % self.config.sample_interval == 0:
                self.generate_sample_images(epoch, category_name)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Training completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)!")
        print(f"{'='*60}\n")
    
    def generate_sample_images(self, epoch, category_name, num_samples=16):
        """Generate and save sample images during training"""
        noise = np.random.normal(0, 1, (num_samples, self.config.latent_dim))
        generated_imgs = self.generator.predict(noise, verbose=0)
        
        # Rescale images from [-1, 1] to [0, 1]
        generated_imgs = 0.5 * generated_imgs + 0.5
        
        # Plot images in a grid
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f'{category_name.upper()} - Epoch {epoch}', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated_imgs[i])
            ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(
            self.config.samples_dir, 
            f'sample_{category_name}_epoch_{epoch:04d}.png'
        )
        plt.savefig(save_path, dpi=100)
        plt.close()
    
    def save_models(self, category_name):
        """Save generator and discriminator models"""
        gen_path = os.path.join(self.config.models_dir, f'generator_{category_name}.h5')
        disc_path = os.path.join(self.config.models_dir, f'discriminator_{category_name}.h5')
        
        self.generator.save(gen_path)
        self.discriminator.save(disc_path)
        
        print(f"✓ Models saved:")
        print(f"  - Generator: {gen_path}")
        print(f"  - Discriminator: {disc_path}")
    
    def plot_training_history(self, category_name):
        """Plot and save training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        axes[0].plot(self.history['d_loss'], label='Discriminator Loss', alpha=0.7, linewidth=2)
        axes[0].plot(self.history['g_loss'], label='Generator Loss', alpha=0.7, linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{category_name.upper()} - Training Loss', fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot discriminator accuracy
        axes[1].plot(self.history['d_acc'], label='Discriminator Accuracy', 
                    color='green', alpha=0.7, linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title(f'{category_name.upper()} - Discriminator Accuracy', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.output_dir, f'training_history_{category_name}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"✓ Training history plot saved: {save_path}")


# ============================================================================
# IMAGE GENERATION FUNCTIONS
# ============================================================================

def generate_synthetic_images(generator, config, num_images, category_name, output_dir):
    """
    Generate synthetic images using trained generator
    
    Args:
        generator: Trained generator model
        config: GANConfig object
        num_images: Number of images to generate
        category_name: Category name (for filenames)
        output_dir: Directory to save images
    """
    print(f"\n{'='*60}")
    print(f"Generating {num_images} synthetic {category_name.upper()} images...")
    print(f"{'='*60}")
    
    for i in range(num_images):
        # Generate random noise
        noise = np.random.normal(0, 1, (1, config.latent_dim))
        
        # Generate image
        generated_img = generator.predict(noise, verbose=0)
        
        # Rescale from [-1, 1] to [0, 255]
        generated_img = (0.5 * generated_img + 0.5) * 255
        generated_img = generated_img.astype(np.uint8)
        
        # Save image
        img = Image.fromarray(generated_img[0])
        img_path = os.path.join(output_dir, f'synthetic_{category_name}_{i:04d}.jpg')
        img.save(img_path)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_images} images...")
    
    print(f"✓ Successfully generated {num_images} images in {output_dir}")


def visualize_generated_images(generator, config, category_name, num_images=25):
    """
    Visualize a grid of generated images
    
    Args:
        generator: Trained generator model
        config: GANConfig object
        category_name: Category name
        num_images: Number of images to display
    """
    noise = np.random.normal(0, 1, (num_images, config.latent_dim))
    generated_imgs = generator.predict(noise, verbose=0)
    
    # Rescale images
    generated_imgs = 0.5 * generated_imgs + 0.5
    
    # Plot in grid
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle(f'Generated {category_name.upper()} Images - Final Result', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(generated_imgs[i])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(config.output_dir, f'final_samples_{category_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✓ Final sample visualization saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GAN-based Synthetic Image Generator')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-synthetic', type=int, default=200, 
                       help='Number of synthetic images to generate per category')
    parser.add_argument('--skip-corrosion', action='store_true', 
                       help='Skip training for corrosion images')
    parser.add_argument('--skip-nocorrosion', action='store_true', 
                       help='Skip training for no-corrosion images')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GAN-BASED SYNTHETIC IMAGE GENERATOR")
    print("For Corrosion Detection Dataset Augmentation")
    print("="*60 + "\n")
    
    # Initialize configuration
    config = GANConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.create_directories()
    
    # Initialize data loader
    data_loader = DataLoader(config)
    
    # ========================================================================
    # TRAIN GAN FOR CORROSION IMAGES
    # ========================================================================
    
    if not args.skip_corrosion:
        print("\n" + "#"*60)
        print("# PHASE 1: CORROSION IMAGES")
        print("#"*60)
        
        # Load corrosion images
        corrosion_images = data_loader.load_images(config.corrosion_data_path)
        
        if len(corrosion_images) == 0:
            print("ERROR: No corrosion images found. Skipping...")
        else:
            # Build models
            print("\nBuilding Generator and Discriminator networks...")
            generator_corrosion = build_generator(config)
            discriminator_corrosion = build_discriminator(config)
            print("✓ Networks built successfully!")
            
            print("\nGenerator Summary:")
            generator_corrosion.summary()
            
            print("\nDiscriminator Summary:")
            discriminator_corrosion.summary()
            
            # Train GAN
            gan_corrosion = DCGAN(config, generator_corrosion, discriminator_corrosion)
            gan_corrosion.train(corrosion_images, 'corrosion')
            
            # Save models
            gan_corrosion.save_models('corrosion')
            
            # Plot training history
            gan_corrosion.plot_training_history('corrosion')
            
            # Visualize final samples
            visualize_generated_images(generator_corrosion, config, 'corrosion', num_images=25)
            
            # Generate synthetic images
            generate_synthetic_images(
                generator_corrosion, 
                config, 
                args.num_synthetic, 
                'corrosion',
                config.synthetic_corrosion_dir
            )
    
    # ========================================================================
    # TRAIN GAN FOR NOCORROSION IMAGES
    # ========================================================================
    
    if not args.skip_nocorrosion:
        print("\n" + "#"*60)
        print("# PHASE 2: NOCORROSION IMAGES")
        print("#"*60)
        
        # Load no-corrosion images
        nocorrosion_images = data_loader.load_images(config.nocorrosion_data_path)
        
        if len(nocorrosion_images) == 0:
            print("ERROR: No no-corrosion images found. Skipping...")
        else:
            # Build models
            print("\nBuilding Generator and Discriminator networks...")
            generator_nocorrosion = build_generator(config)
            discriminator_nocorrosion = build_discriminator(config)
            print("✓ Networks built successfully!")
            
            # Train GAN
            gan_nocorrosion = DCGAN(config, generator_nocorrosion, discriminator_nocorrosion)
            gan_nocorrosion.train(nocorrosion_images, 'nocorrosion')
            
            # Save models
            gan_nocorrosion.save_models('nocorrosion')
            
            # Plot training history
            gan_nocorrosion.plot_training_history('nocorrosion')
            
            # Visualize final samples
            visualize_generated_images(generator_nocorrosion, config, 'nocorrosion', num_images=25)
            
            # Generate synthetic images
            generate_synthetic_images(
                generator_nocorrosion, 
                config, 
                args.num_synthetic, 
                'nocorrosion',
                config.synthetic_nocorrosion_dir
            )
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    print("\n" + "="*60)
    print("✓ GAN TRAINING AND GENERATION COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {config.output_dir}")
    print(f"  - Models: {config.models_dir}")
    print(f"  - Synthetic corrosion images: {config.synthetic_corrosion_dir}")
    print(f"  - Synthetic no-corrosion images: {config.synthetic_nocorrosion_dir}")
    print(f"  - Training samples: {config.samples_dir}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
