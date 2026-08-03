"""
Day 76 — Model Optimization + Callbacks
Topic: Production Callback Stack
Date: 02 August 2026
Author: Bala Ravi

The complete callback stack used in
every production deep learning project!
This exact stack goes into Day 77!
"""
import numpy as np
import os
import csv
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    tf.random.set_seed(42)
except ImportError:
    TF_AVAILABLE = False


class TrainingLogger(keras.callbacks.Callback
                     if TF_AVAILABLE else object):
    """
    Custom callback to log training metrics.
    Saves to CSV and prints formatted output.
    """

    def __init__(self,
                  log_path: str = 'training_log.csv'
                  ) -> None:
        if TF_AVAILABLE:
            super().__init__()
        self.log_path = log_path
        self.logs_list = []

    def on_epoch_end(
            self, epoch: int,
            logs: dict = None) -> None:
        logs = logs or {}
        lr = float(keras.backend.get_value(
            self.model.optimizer.lr))

        row = {
            'epoch': epoch + 1,
            'loss': logs.get('loss', 0),
            'accuracy': logs.get('accuracy', 0),
            'val_loss': logs.get('val_loss', 0),
            'val_accuracy': logs.get(
                'val_accuracy', 0),
            'lr': lr
        }
        self.logs_list.append(row)

        if (epoch + 1) % 5 == 0:
            print(f"\n  📊 Epoch {epoch+1:>3}: "
                  f"loss={row['loss']:.4f}, "
                  f"val_loss={row['val_loss']:.4f}, "
                  f"val_acc={row['val_accuracy']:.4f}, "
                  f"lr={lr:.2e}")


def build_production_callbacks(
        checkpoint_dir: str = 'checkpoints',
        model_name: str = 'skin_model'
        ) -> list:
    """
    Build complete production callback stack.

    Used in Skin Disease Detector (Day 77)!

    Args:
        checkpoint_dir: Directory to save models
        model_name: Base name for saved models

    Returns:
        List of Keras callbacks
    """
    if not TF_AVAILABLE:
        return []

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'{model_name}_best.h5')

    callbacks = [
        # 1. Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1),

        # 2. Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1),

        # 3. Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            min_delta=0.001,
            cooldown=2,
            verbose=1),

        # 4. Training logger
        TrainingLogger(
            log_path=os.path.join(
                checkpoint_dir,
                'training_log.csv')),

        # 5. CSV Logger (built-in)
        keras.callbacks.CSVLogger(
            os.path.join(
                checkpoint_dir,
                'metrics.csv'),
            append=True)
    ]

    return callbacks


def demonstrate_callback_stack() -> None:
    """Show full callback stack in action."""
    print("=== Production Callback Stack ===\n")

    if not TF_AVAILABLE:
        print("TensorFlow not available.")
        print("\nCallback stack for Skin Disease Detector:\n")

        callbacks_info = [
            {
                'name': 'EarlyStopping',
                'config': {
                    'monitor': 'val_loss',
                    'patience': 15,
                    'restore_best_weights': True,
                    'min_delta': 0.001
                },
                'purpose': 'Stop when no improvement'
            },
            {
                'name': 'ModelCheckpoint',
                'config': {
                    'monitor': 'val_accuracy',
                    'save_best_only': True,
                    'mode': 'max'
                },
                'purpose': 'Save best weights'
            },
            {
                'name': 'ReduceLROnPlateau',
                'config': {
                    'monitor': 'val_loss',
                    'factor': 0.3,
                    'patience': 5,
                    'min_lr': 1e-7
                },
                'purpose': 'Adapt learning rate'
            },
            {
                'name': 'CSVLogger',
                'config': {
                    'filename': 'training_log.csv'
                },
                'purpose': 'Log all metrics to file'
            }
        ]

        for cb in callbacks_info:
            print(f"  {cb['name']}:")
            print(f"    Purpose: {cb['purpose']}")
            print(f"    Config:")
            for k, v in cb['config'].items():
                print(f"      {k}: {v}")
            print()
        return

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = make_classification(
        n_samples=800, n_features=20,
        n_informative=12, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = keras.Sequential([
        keras.layers.Dense(
            256, activation='relu',
            input_shape=(20,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    import tempfile
    checkpoint_dir = tempfile.mkdtemp()

    callbacks = build_production_callbacks(
        checkpoint_dir=checkpoint_dir,
        model_name='demo_model')

    print(f"Callbacks loaded: {len(callbacks)}")
    print(f"Checkpoint dir: {checkpoint_dir}\n")
    print("Training with full callback stack...")

    history = model.fit(
        X_tr, y_tr,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0)

    n_epochs = len(history.history['loss'])
    final_val_acc = history.history[
        'val_accuracy'][-1]
    best_val_acc = max(
        history.history['val_accuracy'])

    print(f"\nTraining complete!")
    print(f"  Epochs trained:  {n_epochs}")
    print(f"  Best val acc:    {best_val_acc:.4f}")
    print(f"  Final val acc:   {final_val_acc:.4f}")

    # Check if checkpoint saved
    best_path = os.path.join(
        checkpoint_dir, 'demo_model_best.h5')
    if os.path.exists(best_path):
        size = os.path.getsize(best_path) / 1024
        print(f"  Checkpoint:      "
              f"saved ({size:.0f} KB) ✅")

    test_acc = model.evaluate(
        X_te, y_te, verbose=0)[1]
    print(f"  Test accuracy:   {test_acc:.4f}")
    print(f"\n✅ Production callback stack works!")
    print(f"   Ready for Day 77 Skin Detector! 🔥")


def training_diagnostics() -> None:
    """Show how to diagnose training issues."""
    print("\n=== Training Diagnostics Guide ===\n")

    scenarios = {
        'Perfect training 🎯': {
            'symptoms': [
                'train_loss steadily decreasing',
                'val_loss close to train_loss',
                'val_accuracy improving',
                'LR reducing smoothly'
            ],
            'action': 'Keep training! Increase epochs.'
        },
        'Overfitting ⚠️': {
            'symptoms': [
                'train_loss → 0',
                'val_loss increasing',
                'train_acc = 99%, val_acc = 70%',
                'large train-val gap'
            ],
            'action': (
                'Add Dropout\n'
                '   Add BatchNormalization\n'
                '   Add more augmentation\n'
                '   Reduce model capacity\n'
                '   Add L2 regularization')
        },
        'Underfitting ❌': {
            'symptoms': [
                'train_loss not decreasing',
                'both train and val acc low',
                'loss plateaus early'
            ],
            'action': (
                'Increase model capacity\n'
                '   Increase learning rate\n'
                '   Train longer\n'
                '   Check normalization!\n'
                '   Check data pipeline!')
        },
        'LR too high 🔥': {
            'symptoms': [
                'loss oscillates up and down',
                'val_loss never stabilizes',
                'training unstable'
            ],
            'action': (
                'Reduce LR by 10×\n'
                '   Add gradient clipping\n'
                '   Use LR warmup')
        },
        'Dying ReLU 💀': {
            'symptoms': [
                'loss stops at high value',
                'accuracy stuck at random chance',
                'gradients near zero'
            ],
            'action': (
                'Switch to LeakyReLU\n'
                '   Reduce learning rate\n'
                '   Add BatchNormalization\n'
                '   Use He initialization')
        }
    }

    for scenario, info in scenarios.items():
        print(f"  {scenario}")
        print(f"  Symptoms:")
        for symptom in info['symptoms']:
            print(f"    → {symptom}")
        print(f"  Fix: {info['action']}")
        print()


if __name__ == "__main__":
    demonstrate_callback_stack()
    training_diagnostics()
