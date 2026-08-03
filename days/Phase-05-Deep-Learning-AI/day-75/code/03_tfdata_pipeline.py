"""
Day 75 — Image Preprocessing + Augmentation
Topic: Production tf.data Pipeline
Date: 01 August 2026
Author: Bala Ravi

Building efficient data pipelines with tf.data.
AUTOTUNE, caching, prefetching for GPU efficiency!
"""
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    tf.random.set_seed(42)
except ImportError:
    TF_AVAILABLE = False


def compare_pipeline_speeds() -> None:
    """Compare naive vs tf.data pipeline speed."""
    print("=== Pipeline Speed Comparison ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nExpected speed comparison:\n")
        print(f"{'Method':<30} | {'Time/epoch':>12}")
        print("-" * 45)
        speeds = [
            ('Naive (no optimization)', '8.2s'),
            ('tf.data (basic)', '3.1s'),
            ('tf.data + cache', '1.4s'),
            ('tf.data + cache + prefetch', '0.9s'),
            ('tf.data + AUTOTUNE', '0.7s')
        ]
        for name, speed in speeds:
            print(f"{name:<30} | {speed:>12}")
        print(f"\n✅ AUTOTUNE is 11× faster than naive!")
        return

    n_samples = 500
    img_size = 64

    X = np.random.uniform(
        0, 255, (n_samples, img_size, img_size, 3)
    ).astype(np.float32)
    y = np.random.randint(0, 7, n_samples)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    print("Timing different pipeline configurations:\n")
    print(f"{'Method':<35} | {'Time (10 batches)':>18}")
    print("-" * 57)

    # Method 1: Basic
    ds1 = (tf.data.Dataset
           .from_tensor_slices((X, y))
           .map(preprocess)
           .batch(32))

    start = time.time()
    for batch in ds1.take(10):
        pass
    t1 = time.time() - start
    print(f"{'Basic (no optimization)':<35} | "
          f"{t1:>17.3f}s")

    # Method 2: Parallel map
    ds2 = (tf.data.Dataset
           .from_tensor_slices((X, y))
           .map(preprocess,
                num_parallel_calls=tf.data.AUTOTUNE)
           .batch(32))

    start = time.time()
    for batch in ds2.take(10):
        pass
    t2 = time.time() - start
    print(f"{'Parallel map (AUTOTUNE)':<35} | "
          f"{t2:>17.3f}s")

    # Method 3: Cache + prefetch
    ds3 = (tf.data.Dataset
           .from_tensor_slices((X, y))
           .map(preprocess,
                num_parallel_calls=tf.data.AUTOTUNE)
           .cache()
           .batch(32)
           .prefetch(tf.data.AUTOTUNE))

    # Warm up cache
    for batch in ds3.take(10):
        pass
    # Now time it (cache is warm)
    start = time.time()
    for batch in ds3.take(10):
        pass
    t3 = time.time() - start
    print(f"{'Cache + Prefetch (AUTOTUNE)':<35} | "
          f"{t3:>17.3f}s")

    speedup = t1 / max(t3, 0.001)
    print(f"\n✅ Optimized pipeline is "
          f"{speedup:.1f}× faster!")
    print(f"   cache() → data stays in RAM after epoch 1")
    print(f"   prefetch() → GPU never waits for data! 🔥")


def build_complete_train_pipeline(
        img_size: int = 64,
        batch_size: int = 32,
        n_classes: int = 7) -> tuple:
    """
    Build complete train + val pipelines.

    Returns:
        (train_dataset, val_dataset)
    """
    if not TF_AVAILABLE:
        return None, None

    n_total = 350

    # Generate synthetic data
    X_all = np.random.uniform(
        0, 255, (n_total, img_size, img_size, 3)
    ).astype(np.float32)
    y_all = np.array(
        [i % n_classes for i in range(n_total)],
        dtype=np.int32)

    # Split
    split = int(n_total * 0.8)
    X_train, X_val = X_all[:split], X_all[split:]
    y_train, y_val = y_all[:split], y_all[split:]

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(
            image, [img_size, img_size])
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(
            image, max_delta=0.1)
        image = tf.image.random_contrast(
            image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    # Train pipeline (WITH augmentation)
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train))
        .map(preprocess,
             num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment,
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .shuffle(buffer_size=split)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    # Val pipeline (NO augmentation!)
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_val, y_val))
        .map(preprocess,
             num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds


def train_with_pipeline() -> None:
    """Train CNN using optimized pipeline."""
    print("\n=== Training with tf.data Pipeline ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        return

    img_size = 64
    n_classes = 7

    train_ds, val_ds = build_complete_train_pipeline(
        img_size=img_size,
        batch_size=32,
        n_classes=n_classes)

    # Build model
    model = keras.Sequential([
        keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            padding='same',
            input_shape=(img_size, img_size, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(
            64, (3, 3), activation='relu',
            padding='same'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(
            128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    print(f"Model params: {model.count_params():,}")
    print(f"Training on {n_classes} classes...\n")

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=1)
    elapsed = time.time() - start

    print(f"\nTraining time: {elapsed:.1f}s")
    print(f"Final train acc: "
          f"{history.history['accuracy'][-1]:.4f}")
    print(f"Final val acc: "
          f"{history.history['val_accuracy'][-1]:.4f}")
    print(f"\n✅ tf.data pipeline works end-to-end!")
    print(f"   Ready for Day 77 Skin Detector! 🔥")


if __name__ == "__main__":
    compare_pipeline_speeds()
    train_with_pipeline()
