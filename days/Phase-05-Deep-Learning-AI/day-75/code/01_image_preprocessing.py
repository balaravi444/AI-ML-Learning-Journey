"""
Day 75 — Image Preprocessing + Augmentation
Topic: Image Loading, Resizing, Normalization
Date: 01 August 2026
Author: Bala Ravi

Everything needed to prepare images for CNN!
Wrong preprocessing = 20-30% accuracy loss.
"""
import numpy as np
import os
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

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def demonstrate_normalization() -> None:
    """Show different normalization strategies."""
    print("=== Normalization Strategies ===\n")

    # Simulate raw pixel values
    raw_pixels = np.array([
        [255, 128, 0],
        [200, 100, 50],
        [150, 75, 25]
    ], dtype=np.float32)

    print(f"Raw pixels (0-255):\n{raw_pixels}\n")

    # Strategy 1: Divide by 255
    norm_255 = raw_pixels / 255.0
    print(f"Strategy 1 (÷255) → 0 to 1:")
    print(f"{norm_255.round(3)}\n")

    # Strategy 2: MobileNetV2 preprocessing
    # Expected range: -1 to 1
    norm_mobilenet = (raw_pixels / 127.5) - 1.0
    print(f"Strategy 2 (MobileNetV2 preprocess):")
    print(f"→ range: -1 to 1")
    print(f"{norm_mobilenet.round(3)}\n")

    # Strategy 3: Zero-mean normalization
    mean = raw_pixels.mean()
    std = raw_pixels.std()
    norm_zero_mean = (raw_pixels - mean) / (std + 1e-8)
    print(f"Strategy 3 (Zero-mean, std=1):")
    print(f"mean={mean:.1f}, std={std:.1f}")
    print(f"{norm_zero_mean.round(3)}\n")

    print(f"⚠️  CRITICAL: Match preprocessing to model!")
    print(f"   MobileNetV2 → use preprocess_input()")
    print(f"   VGG16 → use VGG16 preprocess_input()")
    print(f"   Custom CNN → use /255.0")
    print(f"\n   Wrong normalization = 20-30% accuracy drop!")


def image_resize_strategies() -> None:
    """Show resizing strategies for images."""
    print("\n=== Image Resizing Strategies ===\n")

    target_size = (224, 224)

    strategies = {
        'Stretch': (
            'Resize to exactly 224×224\n'
            '   → May distort aspect ratio\n'
            '   → Simple, commonly used'),
        'Crop (Center)': (
            'Resize shorter side → center crop\n'
            '   → Preserves aspect ratio\n'
            '   → Loses edge information'),
        'Pad': (
            'Resize preserving ratio → pad edges\n'
            '   → No distortion\n'
            '   → Padded area = zeros/black'),
        'Smart Crop': (
            'ML model finds important region\n'
            '   → Best for medical images\n'
            '   → Most complex')
    }

    print(f"Target size: {target_size}")
    print(f"Original images: various sizes\n")

    for strategy, desc in strategies.items():
        print(f"  {strategy}:")
        print(f"   {desc}")
        print()

    if not TF_AVAILABLE:
        print("TensorFlow not available for demo.")
        return

    # Simulate different input sizes
    test_sizes = [
        (480, 640, 3),   # Landscape photo
        (640, 480, 3),   # Portrait photo
        (512, 512, 3),   # Square photo
        (1080, 1920, 3)  # HD photo
    ]

    print(f"{'Original Size':<20} → "
          f"{'After Resize':>15}")
    print("-" * 40)

    for size in test_sizes:
        print(f"{str(size):<20} → "
              f"{str(target_size + (3,)):>15}")


def build_image_pipeline(
        use_augmentation: bool = True,
        batch_size: int = 32) -> None:
    """
    Build efficient tf.data image pipeline.

    Shows production-quality data loading!
    """
    print("\n=== tf.data Image Pipeline ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nPipeline concept:\n")
        print("""
# Step 1: Create dataset from paths
dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, labels))

# Step 2: Load + preprocess
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Step 3: Apply transformations
dataset = (dataset
    .map(load_image,
         num_parallel_calls=tf.data.AUTOTUNE)
    .cache()            # cache in RAM after epoch 1
    .shuffle(1000)      # shuffle buffer
    .batch(32)          # batch for GPU
    .prefetch(          # GPU never waits for CPU!
        tf.data.AUTOTUNE))
        """)
        return

    # Create synthetic dataset to demo
    n_samples = 200
    img_size = 64

    # Synthetic images (simulate loaded images)
    X = np.random.uniform(
        0, 255, (n_samples, img_size, img_size, 3)
    ).astype(np.float32)
    y = np.random.randint(0, 7, n_samples)

    # tf.data pipeline
    dataset = tf.data.Dataset.from_tensor_slices(
        (X, y))

    # Preprocessing function
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [img_size, img_size])
        return image, label

    # Augmentation function
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(
            image, max_delta=0.1)
        image = tf.image.random_contrast(
            image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0, 1)
        return image, label

    # Build pipeline
    train_ds = (dataset
        .map(preprocess,
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache())

    if use_augmentation:
        train_ds = train_ds.map(
            augment,
            num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (train_ds
        .shuffle(buffer_size=200)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    print(f"Dataset size: {n_samples} images")
    print(f"Batch size: {batch_size}")
    print(f"Augmentation: {use_augmentation}")

    # Show one batch
    for images, labels in train_ds.take(1):
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Pixel range: "
              f"[{images.numpy().min():.3f}, "
              f"{images.numpy().max():.3f}]")
        print(f"Unique labels: "
              f"{np.unique(labels.numpy())}")

    print(f"\n✅ Pipeline ready!")
    print(f"   AUTOTUNE: TF optimizes parallelism")
    print(f"   cache(): data stays in RAM after epoch 1")
    print(f"   prefetch(): GPU never waits for CPU! 🔥")


def handle_class_imbalance() -> None:
    """
    Show class imbalance in HAM10000
    and how to fix it.
    """
    print("\n=== Class Imbalance in HAM10000 ===\n")

    # Real HAM10000 distribution
    ham10000_dist = {
        'Melanocytic Nevi':      6705,
        'Melanoma':               1113,
        'Benign Keratosis':       1099,
        'Basal Cell Carcinoma':    514,
        'Actinic Keratosis':       327,
        'Vascular Lesion':         142,
        'Dermatofibroma':          115
    }

    total = sum(ham10000_dist.values())

    print(f"Total images: {total:,}\n")
    print(f"{'Class':<25} | "
          f"{'Count':>6} | "
          f"{'%':>6} | "
          f"{'Class Weight':>13}")
    print("-" * 60)

    class_weights = {}
    for i, (cls, count) in enumerate(
            ham10000_dist.items()):
        pct = count / total * 100
        # Weight = total / (n_classes × count)
        weight = total / (
            len(ham10000_dist) * count)
        class_weights[i] = round(weight, 2)

        bar = '█' * int(pct // 3)
        print(f"{cls:<25} | "
              f"{count:>6} | "
              f"{pct:>5.1f}% | "
              f"{weight:>13.2f} {bar}")

    print(f"\nClass weights for model.fit():")
    print(f"  class_weight = {class_weights}")

    print(f"\n💡 Without class weights:")
    print(f"   Model predicts 'Nevi' for everything")
    print(f"   Gets 66% accuracy — USELESS!")
    print(f"   Misses ALL melanoma cases! 🚨")
    print(f"\n   With class weights:")
    print(f"   Melanoma weighted 12× higher")
    print(f"   Model learns rare classes too! ✅")


if __name__ == "__main__":
    demonstrate_normalization()
    image_resize_strategies()
    build_image_pipeline(use_augmentation=True)
    handle_class_imbalance()
