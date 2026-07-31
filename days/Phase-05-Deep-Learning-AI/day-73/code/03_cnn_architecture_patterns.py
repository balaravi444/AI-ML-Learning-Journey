"""
Day 73 — CNN: Convolutional Neural Networks
Topic: CNN Architecture Patterns
Date: 30 July 2026
Author: Bala Ravi

VGG-style, ResNet-style blocks.
Building blocks for Skin Disease Detector!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def vgg_block(
        x,
        filters: int,
        n_conv: int = 2):
    """
    VGG-style block: N × Conv(3×3) + MaxPool.

    Args:
        x: Input tensor
        filters: Number of filters
        n_conv: Number of conv layers

    Returns:
        Output tensor
    """
    for _ in range(n_conv):
        x = keras.layers.Conv2D(
            filters, (3, 3),
            padding='same',
            activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    return x


def residual_block(x, filters: int):
    """
    ResNet-style residual block.

    Skip connection: output = F(x) + x
    Allows very deep networks without
    vanishing gradients!

    Args:
        x: Input tensor
        filters: Number of filters

    Returns:
        Output tensor with skip connection
    """
    shortcut = x

    x = keras.layers.Conv2D(
        filters, (3, 3),
        padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(
        filters, (3, 3),
        padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    # Match dimensions if needed
    if shortcut.shape[-1] != filters:
        shortcut = keras.layers.Conv2D(
            filters, (1, 1),
            padding='same')(shortcut)

    x = keras.layers.Add()([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def build_custom_cnn_for_skin(
        input_shape: tuple = (224, 224, 3),
        n_classes: int = 7) -> 'keras.Model':
    """
    Custom CNN for skin disease detection.
    Uses VGG-style blocks + GAP.

    This is what Skin Disease Detector
    (Day 77) will be built on!

    Args:
        input_shape: (H, W, C) — 224×224×3
        n_classes: Number of disease classes

    Returns:
        Compiled CNN model
    """
    if not TF_AVAILABLE:
        return None

    inputs = keras.Input(shape=input_shape)

    # Block 1: Learn basic features
    x = vgg_block(inputs, filters=32, n_conv=2)

    # Block 2: Learn intermediate features
    x = vgg_block(x, filters=64, n_conv=2)

    # Block 3: Learn complex features
    x = vgg_block(x, filters=128, n_conv=3)

    # Block 4: Learn disease-specific features
    x = vgg_block(x, filters=256, n_conv=3)

    # Global Average Pooling
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Classifier
    x = keras.layers.Dense(
        512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = keras.layers.Dense(
        n_classes,
        activation='softmax')(x)

    model = keras.Model(inputs, outputs,
                         name='skin_cnn')

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


def architecture_summary() -> None:
    """Print CNN architecture designs."""
    print("=== CNN Architecture Patterns ===\n")

    architectures = {
        'Simple CNN (ours Day 73)': {
            'layers': [
                'Conv2D(32) + BN + MaxPool',
                'Conv2D(64) + BN + MaxPool',
                'Conv2D(128) + BN',
                'GlobalAveragePooling',
                'Dense(256) + Dropout',
                'Dense(n_classes)'
            ],
            'params': '~400K',
            'use_for': 'Simple datasets, fast training'
        },
        'VGG-style (our skin detector)': {
            'layers': [
                '2×Conv2D(32) + MaxPool',
                '2×Conv2D(64) + MaxPool',
                '3×Conv2D(128) + MaxPool',
                '3×Conv2D(256) + MaxPool',
                'GAP + Dense(512) + Dense(256)',
                'Dense(n_classes)'
            ],
            'params': '~5M',
            'use_for': 'Medical imaging, high accuracy'
        },
        'Transfer Learning (Day 74)': {
            'layers': [
                'MobileNetV2 (pretrained, frozen)',
                '→ 160K learned features',
                'GlobalAveragePooling2D',
                'Dense(256) + Dropout',
                'Dense(n_classes)'
            ],
            'params': '~3M total (only 600K trainable)',
            'use_for': 'Small datasets, best approach!'
        }
    }

    for name, arch in architectures.items():
        print(f"📐 {name}")
        print(f"   Parameters: {arch['params']}")
        print(f"   Best for:   {arch['use_for']}")
        print(f"   Layers:")
        for layer in arch['layers']:
            print(f"     → {layer}")
        print()

    print("💡 Which to choose?")
    print("   < 1000 images    → Transfer Learning")
    print("   1K-10K images    → Transfer Learning")
    print("   > 100K images    → Train from scratch")
    print("\n   For Skin Disease Detector (Day 77):")
    print("   Transfer Learning! MobileNetV2! 🔥")


if __name__ == "__main__":
    architecture_summary()

    if TF_AVAILABLE:
        print("\n=== Custom Skin CNN ===\n")
        model = build_custom_cnn_for_skin(
            input_shape=(64, 64, 3),
            n_classes=7)
        model.summary()
    else:
        print("\nTensorFlow not available.")
        print("Install: pip install tensorflow")
