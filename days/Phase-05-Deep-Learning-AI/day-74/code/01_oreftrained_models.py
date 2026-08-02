"""
Day 74 — Transfer Learning
Topic: Pretrained Models + Feature Extraction
Date: 31 July 2026
Author: Bala Ravi

Using MobileNetV2 pretrained on ImageNet
as a feature extractor for our tasks!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    tf.random.set_seed(42)
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed.")
    print("    Run: pip install tensorflow\n")


def load_mobilenetv2(
        input_shape: tuple = (224, 224, 3),
        freeze: bool = True) -> 'keras.Model':
    """
    Load MobileNetV2 pretrained on ImageNet.

    Args:
        input_shape: Input image shape
        freeze: Whether to freeze base weights

    Returns:
        MobileNetV2 base model
    """
    if not TF_AVAILABLE:
        return None

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,   # Remove ImageNet head!
        weights='imagenet')  # Load pretrained weights

    base_model.trainable = not freeze

    return base_model


def build_transfer_model(
        base_model,
        n_classes: int,
        input_shape: tuple = (224, 224, 3)
        ) -> 'keras.Model':
    """
    Build complete transfer learning model.

    Pretrained base (frozen) +
    Custom classification head (trainable)

    Args:
        base_model: Pretrained base
        n_classes: Number of output classes
        input_shape: Input image shape

    Returns:
        Complete compiled model
    """
    if not TF_AVAILABLE:
        return None

    inputs = keras.Input(shape=input_shape)

    # Preprocessing (MobileNetV2 specific)
    x = keras.applications.mobilenet_v2.preprocess_input(
        inputs)

    # Base model (frozen)
    x = base_model(x, training=False)

    # New classification head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        256, activation='relu')(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(
        128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(
        n_classes,
        activation='softmax' if n_classes > 2
        else 'sigmoid')(x)

    model = keras.Model(inputs, outputs,
                         name='transfer_model')

    # Compile with higher LR for head training
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3),
        loss=('sparse_categorical_crossentropy'
               if n_classes > 2
               else 'binary_crossentropy'),
        metrics=['accuracy'])

    return model


def demonstrate_feature_extraction() -> None:
    """Show feature extraction with MobileNetV2."""
    print("=== MobileNetV2 Feature Extraction ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nMobileNetV2 key facts:")
        print("  Total layers:       154")
        print("  Total parameters:   3,538,984")
        print("  ImageNet accuracy:  71.8%")
        print("  Input size:         224×224×3")
        print("  Output (no top):    7×7×1280")
        print("\nAfter GlobalAveragePooling2D:")
        print("  1280-dimensional feature vector")
        print("  per image!")
        print("\n💡 These 1280 features encode:")
        print("   Textures, shapes, colors, patterns")
        print("   Learned from 1.2M ImageNet images!")
        print("   Universal visual features! 🔥")
        return

    print("Loading MobileNetV2...")
    base = load_mobilenetv2(
        input_shape=(224, 224, 3),
        freeze=True)

    total_params = base.count_params()
    trainable_params = sum([
        tf.size(v).numpy()
        for v in base.trainable_variables])

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable (frozen=0): {trainable_params:,}")
    print(f"Output shape:         "
          f"{base.output_shape}")

    print("\nBuilding transfer model for 7 classes...")
    model = build_transfer_model(
        base, n_classes=7,
        input_shape=(224, 224, 3))

    total = model.count_params()
    trainable = sum([
        tf.size(v).numpy()
        for v in model.trainable_variables])

    print(f"\nComplete model:")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params:    {total-trainable:,}")
    print(f"\n  Only {trainable/total*100:.1f}% of "
          f"parameters need training!")
    print(f"  → Fast training with small dataset! 🔥")


def simulate_training_phases() -> None:
    """Show the 2-phase training strategy."""
    print("\n=== 2-Phase Transfer Learning ===\n")

    print("Phase 1: Feature Extraction")
    print("  → Base model FROZEN")
    print("  → Only head layers train")
    print("  → Learning rate: 1e-3 (high)")
    print("  → Epochs: 10-20")
    print("  → Goal: quickly learn basic mapping")
    print()

    print("Phase 2: Fine-tuning")
    print("  → Unfreeze TOP layers of base")
    print("  → All layers train (base very slowly)")
    print("  → Learning rate: 1e-5 (very low!)")
    print("  → Epochs: 10-30")
    print("  → Goal: adapt pretrained features")
    print("         to our specific domain")
    print()

    print("⚠️  CRITICAL: Always Phase 1 first!")
    print("   Fine-tuning without Phase 1 =")
    print("   catastrophic forgetting!")
    print("   Pretrained weights get destroyed! ❌")
    print()

    # Simulate results
    print("Simulated results (7-class skin dataset):")
    print(f"\n{'Phase':<30} | "
          f"{'Val Acc':>8} | "
          f"{'Notes'}")
    print("-" * 55)

    phases = [
        ("Random init (no transfer)",
         0.421, "Need 100K+ images"),
        ("Phase 1: Feature extraction",
         0.823, "Only 1K images needed!"),
        ("Phase 2: Fine-tuning",
         0.891, "+6.8% improvement!"),
        ("Phase 2 + augmentation",
         0.913, "Production quality!")
    ]

    for phase, acc, note in phases:
        bar = '█' * int(acc * 20)
        print(f"{phase:<30} | "
              f"{acc:>8.3f} | "
              f"{note}")

    print(f"\n✅ Transfer Learning:")
    print(f"   0.421 → 0.913 accuracy")
    print(f"   With only 1000 training images!")


def compare_pretrained_models() -> None:
    """Compare popular pretrained models."""
    print("\n=== Pretrained Model Comparison ===\n")

    models_info = {
        'MobileNetV2': {
            'params': '3.4M',
            'imagenet_acc': 71.8,
            'input': '224×224',
            'speed': '⚡⚡⚡',
            'best_for': 'Mobile, fast inference'
        },
        'EfficientNetB0': {
            'params': '5.3M',
            'imagenet_acc': 77.1,
            'input': '224×224',
            'speed': '⚡⚡⚡',
            'best_for': 'Best accuracy/params ratio'
        },
        'ResNet50': {
            'params': '25M',
            'imagenet_acc': 74.9,
            'input': '224×224',
            'speed': '⚡⚡',
            'best_for': 'General purpose'
        },
        'InceptionV3': {
            'params': '23M',
            'imagenet_acc': 77.9,
            'input': '299×299',
            'speed': '⚡⚡',
            'best_for': 'Multi-scale features'
        },
        'VGG16': {
            'params': '138M',
            'imagenet_acc': 71.5,
            'input': '224×224',
            'speed': '⚡',
            'best_for': 'Simple, well-understood'
        }
    }

    print(f"{'Model':<18} | "
          f"{'Params':>8} | "
          f"{'ImageNet':>9} | "
          f"{'Speed':>7} | "
          f"{'Best For'}")
    print("-" * 75)

    for name, info in models_info.items():
        marker = " ← our choice" if name == 'MobileNetV2' else ""
        print(f"{name:<18} | "
              f"{info['params']:>8} | "
              f"{info['imagenet_acc']:>8.1f}% | "
              f"{info['speed']:>7} | "
              f"{info['best_for']}{marker}")

    print(f"\n💡 For Skin Disease Detector (Day 77):")
    print(f"   MobileNetV2 → fast + accurate!")
    print(f"   Works well on free GPU (Colab)!")


if __name__ == "__main__":
    demonstrate_feature_extraction()
    simulate_training_phases()
    compare_pretrained_models()
