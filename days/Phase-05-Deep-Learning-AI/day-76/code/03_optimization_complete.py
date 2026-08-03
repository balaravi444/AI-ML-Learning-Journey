"""
Day 76 — Model Optimization + Callbacks
Topic: Complete Optimization Pipeline
Date: 02 August 2026
Author: Bala Ravi

Full optimized training pipeline ready
for Skin Disease Detector (Day 77)!
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


def build_optimized_model(
        input_shape: tuple,
        n_classes: int,
        dropout_rate: float = 0.4
        ) -> 'keras.Model':
    """
    Build fully optimized CNN model.

    All best practices applied:
    - BatchNormalization after each Conv
    - Dropout before Dense
    - GlobalAveragePooling (not Flatten)
    - L2 regularization on Dense
    """
    if not TF_AVAILABLE:
        return None

    l2 = keras.regularizers.L2(1e-4)

    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = keras.layers.Conv2D(
        32, (3, 3), padding='same',
        kernel_regularizer=l2)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = keras.layers.Conv2D(
        64, (3, 3), padding='same',
        kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = keras.layers.Conv2D(
        128, (3, 3), padding='same',
        kernel_regularizer=l2)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    # Head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        256, activation='relu',
        kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(
        128, activation='relu',
        kernel_regularizer=l2)(x)
    x = keras.layers.Dropout(dropout_rate * 0.5)(x)

    outputs = keras.layers.Dense(
        n_classes,
        activation='softmax' if n_classes > 2
        else 'sigmoid')(x)

    model = keras.Model(inputs, outputs,
                         name='optimized_cnn')
    return model


def phase1_train(
        model,
        train_ds,
        val_ds,
        epochs: int = 20,
        checkpoint_dir: str = 'checkpoints'
        ) -> dict:
    """
    Phase 1: Train with higher LR.
    For transfer learning: head only.
    For custom CNN: full model.
    """
    if not TF_AVAILABLE:
        return {}

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(
                checkpoint_dir, 'phase1_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4,
            min_lr=1e-6,
            verbose=1)
    ]

    print("Phase 1: Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0)

    best_acc = max(history.history['val_accuracy'])
    epochs_run = len(history.history['loss'])
    print(f"  Epochs: {epochs_run}")
    print(f"  Best val acc: {best_acc:.4f}")

    return {
        'history': history,
        'best_acc': best_acc,
        'epochs': epochs_run}


def phase2_finetune(
        model,
        train_ds,
        val_ds,
        epochs: int = 20,
        checkpoint_dir: str = 'checkpoints'
        ) -> dict:
    """
    Phase 2: Fine-tune with very low LR.
    For transfer learning: unfreeze top layers.
    """
    if not TF_AVAILABLE:
        return {}

    # Recompile with low LR
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(
                checkpoint_dir, 'phase2_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1)
    ]

    print("\nPhase 2: Fine-tuning...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0)

    best_acc = max(history.history['val_accuracy'])
    epochs_run = len(history.history['loss'])
    print(f"  Epochs: {epochs_run}")
    print(f"  Best val acc: {best_acc:.4f}")

    return {
        'history': history,
        'best_acc': best_acc,
        'epochs': epochs_run}


def demo_complete_optimization() -> None:
    """Demo complete optimization pipeline."""
    print("=== Complete Optimization Pipeline ===\n")
    print("This runs in Day 77 Skin Disease Detector!\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nPipeline summary for Day 77:\n")

        print("  Step 1: Load HAM10000 images")
        print("  Step 2: Build tf.data pipeline")
        print("          + augmentation + class weights")
        print("  Step 3: Load MobileNetV2 (frozen)")
        print("  Step 4: Add classification head")
        print("  Step 5: Phase 1 — train head")
        print("          LR=1e-3, EarlyStopping,")
        print("          ReduceLROnPlateau")
        print("  Step 6: Unfreeze top layers")
        print("  Step 7: Phase 2 — fine-tune")
        print("          LR=1e-5, EarlyStopping")
        print("  Step 8: Evaluate on test set")
        print("  Step 9: Save + Deploy FastAPI")
        print("\n  Expected accuracy: 88-92%!")
        print("  All from Days 71-76 combined! 🔥")
        return

    # Quick demo with tiny dataset
    img_size = 32
    n_classes = 4
    n_samples = 200

    X = np.random.uniform(
        0, 1, (n_samples, img_size, img_size, 3)
    ).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)

    split = int(n_samples * 0.8)
    train_ds = (tf.data.Dataset
        .from_tensor_slices(
            (X[:split], y[:split]))
        .shuffle(split)
        .batch(16)
        .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset
        .from_tensor_slices(
            (X[split:], y[split:]))
        .batch(16)
        .prefetch(tf.data.AUTOTUNE))

    model = build_optimized_model(
        input_shape=(img_size, img_size, 3),
        n_classes=n_classes)

    total = model.count_params()
    print(f"Model params: {total:,}")

    import tempfile
    checkpoint_dir = tempfile.mkdtemp()

    # Phase 1
    p1 = phase1_train(
        model, train_ds, val_ds,
        epochs=10,
        checkpoint_dir=checkpoint_dir)

    # Phase 2
    p2 = phase2_finetune(
        model, train_ds, val_ds,
        epochs=10,
        checkpoint_dir=checkpoint_dir)

    print(f"\n{'='*50}")
    print(f"  Optimization Complete!")
    print(f"{'='*50}")
    print(f"  Phase 1 best: {p1['best_acc']:.4f}")
    print(f"  Phase 2 best: {p2['best_acc']:.4f}")
    improvement = p2['best_acc'] - p1['best_acc']
    print(f"  Improvement:  "
          f"{improvement:+.4f}")
    print(f"\n  ✅ All Days 71-76 skills combined!")
    print(f"  Day 77: Apply to real skin images! 🔥")


if __name__ == "__main__":
    demo_complete_optimization()
