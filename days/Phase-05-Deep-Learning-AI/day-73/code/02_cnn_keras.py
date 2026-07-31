"""
Day 73 — CNN: Convolutional Neural Networks
Topic: Build CNN with Keras
Date: 30 July 2026
Author: Bala Ravi

Build, train, and evaluate CNNs using Keras.
Dense NN vs CNN on image classification.
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


def build_dense_for_images(
        input_shape: tuple,
        n_classes: int) -> 'keras.Model':
    """
    Dense NN for image classification.
    Baseline to compare against CNN.
    """
    if not TF_AVAILABLE:
        return None

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(
            n_classes,
            activation='softmax' if n_classes > 2
            else 'sigmoid')
    ], name='dense_model')

    model.compile(
        optimizer='adam',
        loss=('sparse_categorical_crossentropy'
               if n_classes > 2
               else 'binary_crossentropy'),
        metrics=['accuracy'])

    return model


def build_cnn(
        input_shape: tuple,
        n_classes: int) -> 'keras.Model':
    """
    CNN for image classification.

    Architecture: Conv → Pool → Conv → Pool
                  → Conv → GAP → Dense → Output

    Args:
        input_shape: (H, W, C) e.g. (32, 32, 3)
        n_classes: Number of output classes

    Returns:
        Compiled CNN model
    """
    if not TF_AVAILABLE:
        return None

    model = keras.Sequential([
        # Block 1
        keras.layers.Conv2D(
            32, (3, 3), padding='same',
            activation='relu',
            input_shape=input_shape,
            name='conv1'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        # Block 2
        keras.layers.Conv2D(
            64, (3, 3), padding='same',
            activation='relu',
            name='conv2'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        # Block 3
        keras.layers.Conv2D(
            128, (3, 3), padding='same',
            activation='relu',
            name='conv3'),
        keras.layers.BatchNormalization(),

        # Classifier head
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(
            n_classes,
            activation='softmax' if n_classes > 2
            else 'sigmoid')
    ], name='cnn_model')

    model.compile(
        optimizer='adam',
        loss=('sparse_categorical_crossentropy'
               if n_classes > 2
               else 'binary_crossentropy'),
        metrics=['accuracy'])

    return model


def train_on_mnist() -> None:
    """Train Dense vs CNN on MNIST digits."""
    print("=== Dense NN vs CNN on MNIST ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nExpected results on MNIST:\n")
        print(f"{'Model':<20} | "
              f"{'Params':>10} | "
              f"{'Test Acc':>9} | "
              f"{'Train Time':>12}")
        print("-" * 58)
        print(f"{'Dense NN':<20} | "
              f"{670000:>10,} | "
              f"{0.9756:>9.4f} | "
              f"{'~45s':>12}")
        print(f"{'CNN':<20} | "
              f"{122000:>10,} | "
              f"{0.9912:>9.4f} | "
              f"{'~120s':>12}")
        print(f"\n✅ CNN: 5× fewer params, better accuracy!")
        return

    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = (
        keras.datasets.mnist.load_data())

    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # CNN needs channel dimension
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    print(f"Train: {X_train.shape}, "
          f"Test: {X_test.shape}")
    print(f"Classes: 10 (digits 0-9)\n")

    results = {}
    import time

    # Dense NN
    print("Training Dense NN...")
    dense = build_dense_for_images((28, 28), 10)
    start = time.time()
    dense.fit(
        X_train, y_train,
        epochs=10, batch_size=64,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True)],
        verbose=0)
    dense_time = time.time() - start
    dense_acc = dense.evaluate(
        X_test, y_test, verbose=0)[1]
    dense_params = dense.count_params()
    results['Dense NN'] = {
        'params': dense_params,
        'acc': dense_acc,
        'time': dense_time}

    # CNN
    print("Training CNN...")
    cnn = build_cnn((28, 28, 1), 10)
    start = time.time()
    cnn.fit(
        X_train_cnn, y_train,
        epochs=10, batch_size=64,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True)],
        verbose=0)
    cnn_time = time.time() - start
    cnn_acc = cnn.evaluate(
        X_test_cnn, y_test, verbose=0)[1]
    cnn_params = cnn.count_params()
    results['CNN'] = {
        'params': cnn_params,
        'acc': cnn_acc,
        'time': cnn_time}

    print(f"\n{'Model':<12} | "
          f"{'Params':>10} | "
          f"{'Test Acc':>9} | "
          f"{'Time':>8}")
    print("-" * 48)

    for name, r in results.items():
        print(f"{name:<12} | "
              f"{r['params']:>10,} | "
              f"{r['acc']:>9.4f} | "
              f"{r['time']:>7.1f}s")

    cnn_r = results['CNN']
    dense_r = results['Dense NN']
    print(f"\n✅ CNN wins by "
          f"{(cnn_r['acc']-dense_r['acc'])*100:.1f}% "
          f"with "
          f"{dense_r['params']//cnn_r['params']}× "
          f"fewer parameters!")


def visualize_feature_maps() -> None:
    """Show what CNN layers learn."""
    print("\n=== Feature Maps (Conceptual) ===\n")

    print("What each CNN layer learns:\n")

    layers = {
        'Conv2D Layer 1 (32 filters)': [
            'Horizontal edges',
            'Vertical edges',
            'Diagonal edges',
            'Color boundaries',
            'Dark spots',
            'Light spots'
        ],
        'Conv2D Layer 2 (64 filters)': [
            'Corners and junctions',
            'Circles and curves',
            'Simple textures',
            'Color gradients',
            'Complex edges'
        ],
        'Conv2D Layer 3 (128 filters)': [
            'Eye-like patterns',
            'Skin texture patterns',
            'Lesion boundaries',
            'Color distribution patterns',
            'Complex disease-specific features'
        ]
    }

    for layer, features in layers.items():
        print(f"  {layer}:")
        for feat in features:
            print(f"    → {feat}")
        print()

    print("💡 Early layers: generic features")
    print("   (same for any image dataset!)")
    print("   Late layers: task-specific features")
    print("   (unique to skin disease detection)")
    print("\n   This is WHY transfer learning works!")
    print("   Reuse early layers, retrain late layers!")
    print("   → Day 74 tomorrow! 🔥")


if __name__ == "__main__":
    train_on_mnist()
    visualize_feature_maps()
