"""
Day 77 — Skin Disease Detector
Topic: Dataset Preparation + Analysis
Date: 03 August 2026
Author: Bala Ravi

HAM10000 dataset preparation.
Handles class imbalance, creates tf.data pipeline.
"""
import numpy as np
import pandas as pd
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
    from sklearn.utils.class_weight import (
        compute_class_weight)
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─── Disease Classes ──────────────────────────────
DISEASE_CLASSES = {
    'akiec': 0,  # Actinic Keratosis
    'bcc':   1,  # Basal Cell Carcinoma
    'bkl':   2,  # Benign Keratosis
    'df':    3,  # Dermatofibroma
    'mel':   4,  # Melanoma (most dangerous!)
    'nv':    5,  # Melanocytic Nevi
    'vasc':  6   # Vascular Lesion
}

CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular Lesion'
]

SEVERITY = {
    'Actinic Keratosis': ('MEDIUM',
                           'Precancerous lesion'),
    'Basal Cell Carcinoma': ('HIGH',
                              'Most common skin cancer'),
    'Benign Keratosis': ('LOW',
                          'Non-cancerous'),
    'Dermatofibroma': ('LOW',
                        'Benign skin growth'),
    'Melanoma': ('CRITICAL 🚨',
                  'Deadliest skin cancer'),
    'Melanocytic Nevi': ('LOW',
                          'Common mole'),
    'Vascular Lesion': ('LOW',
                         'Blood vessel abnormality')
}

# HAM10000 real distribution
HAM10000_DIST = {
    'nv':    6705,
    'mel':   1113,
    'bkl':   1099,
    'bcc':    514,
    'akiec':  327,
    'vasc':   142,
    'df':     115
}


def analyze_dataset() -> None:
    """Analyze HAM10000 class distribution."""
    print("=== HAM10000 Dataset Analysis ===\n")

    total = sum(HAM10000_DIST.values())
    print(f"Total images: {total:,}")
    print(f"Image size:   450×600 pixels")
    print(f"Format:       JPEG dermoscopy\n")

    print(f"{'Class':<25} | "
          f"{'Code':>5} | "
          f"{'Count':>6} | "
          f"{'%':>6} | "
          f"{'Severity'}")
    print("-" * 70)

    for code, count in sorted(
            HAM10000_DIST.items(),
            key=lambda x: x[1],
            reverse=True):
        cls_idx = DISEASE_CLASSES[code]
        cls_name = CLASS_NAMES[cls_idx]
        pct = count / total * 100
        severity = SEVERITY[cls_name][0]
        bar = '█' * int(pct // 3)
        print(f"{cls_name:<25} | "
              f"{code:>5} | "
              f"{count:>6} | "
              f"{pct:>5.1f}% | "
              f"{severity} {bar}")

    print(f"\n⚠️  Severe class imbalance!")
    print(f"   Nevi: {HAM10000_DIST['nv']/total*100:.0f}% "
          f"vs Dermatofibroma: "
          f"{HAM10000_DIST['df']/total*100:.1f}%")
    print(f"\n   Without correction:")
    print(f"   Model predicts Nevi for everything!")
    print(f"   Gets 67% accuracy but misses ALL melanoma!")
    print(f"\n   Solution: class_weight + augmentation! 🔥")


def compute_class_weights() -> dict:
    """
    Compute class weights for imbalanced dataset.

    Inverse frequency weighting:
    weight = total / (n_classes × class_count)
    """
    if not SKLEARN_AVAILABLE:
        # Manual computation
        total = sum(HAM10000_DIST.values())
        n_classes = len(HAM10000_DIST)
        weights = {}
        for code, count in HAM10000_DIST.items():
            idx = DISEASE_CLASSES[code]
            weights[idx] = round(
                total / (n_classes * count), 3)
        return weights

    counts = np.array(list(HAM10000_DIST.values()))
    labels = np.array([
        DISEASE_CLASSES[code]
        for code in HAM10000_DIST.keys()])
    y_flat = np.concatenate([
        np.full(count, label)
        for label, count in zip(
            labels,
            counts)])

    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_flat),
        y=y_flat)

    class_weights = {
        i: float(round(w, 3))
        for i, w in enumerate(weights)}

    return class_weights


def print_class_weights(
        class_weights: dict) -> None:
    """Display class weights."""
    print("\n=== Class Weights ===\n")
    print("Tells model how much to weight each class.\n")

    print(f"{'Class':<25} | "
          f"{'Weight':>8} | "
          f"{'Meaning'}")
    print("-" * 65)

    for idx, cls_name in enumerate(CLASS_NAMES):
        w = class_weights.get(idx, 1.0)
        meaning = (
            "🚨 Melanoma: HIGHEST priority!"
            if cls_name == 'Melanoma' else
            "Common → low weight"
            if cls_name == 'Melanocytic Nevi' else
            f"Rare → weight {w:.2f}×")
        print(f"{cls_name:<25} | "
              f"{w:>8.3f} | "
              f"{meaning}")

    print(f"\n✅ Melanoma gets highest weight!")
    print(f"   Missing melanoma is CRITICAL.")
    print(f"   Weight tells model: this matters more!")


def generate_synthetic_dataset(
        n_per_class: int = 100,
        img_size: int = 96,
        save_dir: str = (
            "projects/skin_disease_detector/data")
        ) -> tuple:
    """
    Generate synthetic skin disease images
    for local testing.

    In real deployment → use HAM10000!
    Download: kaggle datasets download
              kmader/skin-lesion-analysis-toward-melanoma-detection

    Args:
        n_per_class: Images per disease class
        img_size: Image size (square)
        save_dir: Directory to save data

    Returns:
        (X, y, class_weights) tuples
    """
    print(f"\n=== Generating Synthetic Dataset ===\n")
    print(f"(Real project uses HAM10000!)")
    print(f"Images per class: {n_per_class}")
    print(f"Image size:       {img_size}×{img_size}\n")

    np.random.seed(42)
    n_classes = len(CLASS_NAMES)

    X = []
    y = []

    # Disease-specific color signatures
    # Each disease has different dermoscopic appearance
    disease_profiles = {
        0: {  # Actinic Keratosis - rough, scaly
            'base_color': [0.85, 0.65, 0.55],
            'lesion_color': [0.70, 0.45, 0.35],
            'texture': 'rough'},
        1: {  # Basal Cell Carcinoma - pearly
            'base_color': [0.90, 0.78, 0.68],
            'lesion_color': [0.95, 0.90, 0.85],
            'texture': 'smooth'},
        2: {  # Benign Keratosis - brown
            'base_color': [0.85, 0.70, 0.55],
            'lesion_color': [0.50, 0.30, 0.15],
            'texture': 'rough'},
        3: {  # Dermatofibroma - firm
            'base_color': [0.88, 0.72, 0.62],
            'lesion_color': [0.60, 0.40, 0.30],
            'texture': 'smooth'},
        4: {  # Melanoma - dark, irregular
            'base_color': [0.85, 0.70, 0.60],
            'lesion_color': [0.15, 0.08, 0.05],
            'texture': 'irregular'},
        5: {  # Melanocytic Nevi - regular mole
            'base_color': [0.88, 0.73, 0.63],
            'lesion_color': [0.35, 0.20, 0.10],
            'texture': 'smooth'},
        6: {  # Vascular - red
            'base_color': [0.90, 0.75, 0.65],
            'lesion_color': [0.85, 0.15, 0.15],
            'texture': 'smooth'}
    }

    for class_idx in range(n_classes):
        profile = disease_profiles[class_idx]
        base = np.array(profile['base_color'])
        lesion = np.array(profile['lesion_color'])

        for _ in range(n_per_class):
            # Create base skin tone
            img = np.random.normal(
                base, 0.05,
                (img_size, img_size, 3))

            # Add lesion in random position
            cx = np.random.randint(
                img_size // 4,
                3 * img_size // 4)
            cy = np.random.randint(
                img_size // 4,
                3 * img_size // 4)

            # Irregular shape for melanoma
            if profile['texture'] == 'irregular':
                radii = np.random.uniform(
                    img_size // 8,
                    img_size // 4,
                    size=36)
            else:
                r = np.random.uniform(
                    img_size // 8,
                    img_size // 5)
                radii = np.full(36, r)

            angles = np.linspace(0, 2*np.pi, 36)

            for i in range(img_size):
                for j in range(img_size):
                    dx = i - cx
                    dy = j - cy

                    if dx == 0 and dy == 0:
                        angle_idx = 0
                    else:
                        angle = np.arctan2(dy, dx)
                        angle = (angle + 2*np.pi) % (
                            2*np.pi)
                        angle_idx = int(
                            angle / (2*np.pi) * 36)
                        angle_idx = min(
                            angle_idx, 35)

                    dist = np.sqrt(dx**2 + dy**2)
                    r = radii[angle_idx]

                    if dist < r:
                        alpha = 1 - dist / r
                        img[i, j] = (
                            alpha * lesion +
                            (1 - alpha) * img[i, j])

            # Add noise
            img += np.random.normal(
                0, 0.02,
                (img_size, img_size, 3))
            img = np.clip(img, 0, 1)

            X.append(img.astype(np.float32))
            y.append(class_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    total = len(X)
    print(f"Generated: {total} synthetic images")
    print(f"Shape: {X.shape}")
    print(f"Classes: {n_classes}")

    # Compute class weights
    class_weights = {}
    for cls_idx in range(n_classes):
        count = (y == cls_idx).sum()
        weight = total / (n_classes * count)
        class_weights[cls_idx] = round(weight, 3)

    return X, y, class_weights


def build_tfdata_pipeline(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 16,
        img_size: int = 96,
        augment: bool = True) -> tuple:
    """
    Build tf.data pipeline for skin images.

    Args:
        X: Image array (N, H, W, 3)
        y: Labels
        batch_size: Batch size
        img_size: Target image size
        augment: Apply augmentation to train

    Returns:
        (train_ds, val_ds, test_ds)
    """
    if not TF_AVAILABLE:
        return None, None, None

    # Split: 70% train, 15% val, 15% test
    n = len(X)
    idx = np.random.permutation(n)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X[idx[:train_end]]
    y_train = y[idx[:train_end]]
    X_val = X[idx[train_end:val_end]]
    y_val = y[idx[train_end:val_end]]
    X_test = X[idx[val_end:]]
    y_test = y[idx[val_end:]]

    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    def preprocess(image, label):
        image = tf.image.resize(
            image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        # MobileNetV2 preprocessing
        image = (image / 127.5) - 1.0
        return image, label

    def augment_fn(image, label):
        image = tf.image.random_flip_left_right(
            image)
        image = tf.image.random_flip_up_down(
            image)
        image = tf.image.random_brightness(
            image, max_delta=0.1)
        image = tf.image.random_contrast(
            image, 0.9, 1.1)
        image = tf.image.random_saturation(
            image, 0.9, 1.1)
        image = tf.clip_by_value(image, -1.0, 1.0)
        return image, label

    def make_ds(X_arr, y_arr, shuffle=False,
                 do_augment=False) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(
            (X_arr, y_arr))
        ds = ds.map(
            preprocess,
            num_parallel_calls=tf.data.AUTOTUNE)
        if do_augment:
            ds = ds.map(
                augment_fn,
                num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(
                buffer_size=len(X_arr))
        ds = (ds
              .cache()
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))
        return ds

    train_ds = make_ds(
        X_train, y_train,
        shuffle=True,
        do_augment=augment)
    val_ds = make_ds(X_val, y_val)
    test_ds = make_ds(X_test, y_test)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    analyze_dataset()
    class_weights = compute_class_weights()
    print_class_weights(class_weights)

    X, y, cw = generate_synthetic_dataset(
        n_per_class=50,
        img_size=96)

    if TF_AVAILABLE:
        train_ds, val_ds, test_ds = (
            build_tfdata_pipeline(
                X, y,
                batch_size=16,
                img_size=96,
                augment=True))

        if train_ds is not None:
            for imgs, labels in train_ds.take(1):
                print(f"\nBatch shape: {imgs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Pixel range: "
                      f"[{imgs.numpy().min():.3f}, "
                      f"{imgs.numpy().max():.3f}]")
            print(f"\n✅ Pipeline ready!")
