import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, InceptionV3, EfficientNetB0, MobileNetV2
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from tqdm import tqdm

DATA_DIR = "./data"  # Folder containing 'CORROSION' and 'NOCORROSION'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
OUTPUT_DIR = "./trained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MULTI-GPU STRATEGY
# ============================================================
strategy = tf.distribute.MirroredStrategy()
print(f"✅ Using {strategy.num_replicas_in_sync} GPUs")

# ============================================================
# DATA PIPELINE
# ============================================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation'
)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
MODELS = {
    "VGG16": VGG16,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3,
    "EfficientNetB0": EfficientNetB0,
    "MobileNetV2": MobileNetV2,
}

# ============================================================
# TRAINING LOOP
# ============================================================
for model_name, base_model_fn in MODELS.items():
    print(f"\n🚀 Training {model_name}...")

    with strategy.scope():
        base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
        base_model.trainable = False  # Freeze pretrained layers

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    ckpt_path = os.path.join(OUTPUT_DIR, f"{model_name}_best.h5")

    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print(f"✅ Finished training {model_name}. Best model saved to {ckpt_path}")