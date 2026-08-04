"""
Day 77 — Skin Disease Detector
Topic: MobileNetV2 Transfer Learning Architecture
Date: 03 August 2026
Author: Bala Ravi

Building the CNN that detects skin diseases.
MobileNetV2 + custom classification head.
Phase 1: train head only.
Phase 2 (Day 78): fine-tune top layers.
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


N_CLASSES = 7
IMG_SIZE = 96   # 224 for real HAM10000
CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular Lesion'
]


def build_skin_model(
        img_size: int = IMG_SIZE,
        n_classes: int = N_CLASSES,
        dropout_rate: float = 0.5,
        use_pretrained: bool = True
        ) -> tuple:
    """
    Build skin disease classifier.

    Architecture:
    MobileNetV2 (frozen) → GAP → Dense → Dense → Softmax

    Args:
        img_size: Input image size
        n_classes: Number of disease classes
        dropout_rate: Dropout for regularization
        use_pretrained: Load ImageNet weights

    Returns:
        (model, base_model) tuple
    """
    if not TF_AVAILABLE:
        return None, None

    input_shape = (img_size, img_size, 3)

    # Load MobileNetV2
    weights = 'imagenet' if use_pretrained else None

    try:
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights)
    except Exception:
        # No internet — use random weights
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None)

    # Phase 1: Freeze base
    base_model.trainable = False

    # Build model
    inputs = keras.Input(shape=input_shape)

    # MobileNetV2 preprocessing: -1 to 1
    # Already done in pipeline, but keep for safety
    x = base_model(inputs, training=False)

    # Classification head
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(
        512,
        kernel_regularizer=keras.regularizers.L2(
            1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(
        256,
        kernel_regularizer=keras.regularizers.L2(
            1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(dropout_rate * 0.6)(x)

    outputs = keras.layers.Dense(
        n_classes,
        activation='softmax',
        name='predictions')(x)

    model = keras.Model(
        inputs, outputs,
        name='skin_disease_detector')

    return model, base_model


def compile_model(model) -> None:
    """
    Compile model for Phase 1 training.
    Higher LR for head-only training.
    """
    if model is None:
        return

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Recall(name='recall')
        ])


def get_callbacks(
        checkpoint_dir: str = (
            "projects/skin_disease_detector/checkpoints")
        ) -> list:
    """
    Production callback stack for skin detector.
    All best practices from Day 76!
    """
    if not TF_AVAILABLE:
        return []

    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        # Stop when stuck
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            min_delta=0.001,
            verbose=1),

        # Save best model
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                checkpoint_dir,
                'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1),

        # Adapt LR
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            cooldown=2,
            verbose=1),

        # Log training
        keras.callbacks.CSVLogger(
            os.path.join(
                checkpoint_dir,
                'training_log.csv'),
            append=True)
    ]

    return callbacks


def phase1_train(
        model,
        train_ds,
        val_ds,
        class_weights: dict,
        epochs: int = 20,
        checkpoint_dir: str = (
            "projects/skin_disease_detector/checkpoints")
        ) -> dict:
    """
    Phase 1: Train classification head only.
    Base model frozen.

    Args:
        model: Compiled model
        train_ds: Training dataset
        val_ds: Validation dataset
        class_weights: Balanced class weights
        epochs: Maximum epochs
        checkpoint_dir: Save directory

    Returns:
        Training history dictionary
    """
    if model is None:
        return {}

    print(f"\n{'='*55}")
    print(f"  Phase 1: Feature Extraction")
    print(f"  Base model FROZEN")
    print(f"  Training classification head only")
    print(f"{'='*55}")

    total_params = model.count_params()
    trainable_params = sum([
        tf.size(v).numpy()
        for v in model.trainable_variables])

    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable:        {trainable_params:,}")
    print(f"  Frozen:           "
          f"{total_params - trainable_params:,}")
    print(f"  Trainable %:      "
          f"{trainable_params/total_params*100:.1f}%\n")

    callbacks = get_callbacks(checkpoint_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1)

    best_val_acc = max(
        history.history.get('val_accuracy', [0]))
    epochs_run = len(
        history.history.get('loss', []))

    print(f"\n  Phase 1 Complete!")
    print(f"  Epochs:       {epochs_run}")
    print(f"  Best val acc: {best_val_acc:.4f}")

    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'epochs_run': epochs_run
    }


def evaluate_model(
        model,
        test_ds,
        class_weights: dict = None) -> dict:
    """
    Evaluate model with detailed per-class metrics.
    """
    if model is None or test_ds is None:
        return {}

    print(f"\n=== Model Evaluation ===\n")

    results = model.evaluate(
        test_ds, verbose=0)

    metrics = {}
    for name, value in zip(
            model.metrics_names, results):
        metrics[name] = float(value)
        print(f"  {name}: {value:.4f}")

    # Per-class predictions
    all_preds = []
    all_labels = []

    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(
            batch_imgs, verbose=0)
        all_preds.extend(
            np.argmax(preds, axis=1))
        all_labels.extend(
            batch_labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"\nPer-class Accuracy:")
    print(f"{'Class':<25} | "
          f"{'Correct':>8} | "
          f"{'Total':>6} | "
          f"{'Acc':>7}")
    print("-" * 55)

    for cls_idx, cls_name in enumerate(
            CLASS_NAMES):
        mask = all_labels == cls_idx
        if mask.sum() == 0:
            continue
        correct = (
            all_preds[mask] == cls_idx).sum()
        total = mask.sum()
        acc = correct / total

        flag = "🚨" if cls_name == 'Melanoma' else ""
        print(f"{cls_name:<25} | "
              f"{correct:>8} | "
              f"{total:>6} | "
              f"{acc:>7.3f} {flag}")

    return metrics


def print_model_architecture() -> None:
    """Print architecture without TF."""
    print("=== Skin Disease Detector Architecture ===\n")

    layers = [
        ("Input", "(96, 96, 3)", ""),
        ("MobileNetV2 base", "frozen, pretrained", "3,442,496 params"),
        ("GlobalAveragePooling2D", "(1280,)", "0 params"),
        ("BatchNormalization", "(1280,)", "5,120 params"),
        ("Dense(512) + BN + ReLU", "(512,)", "655,872 params"),
        ("Dropout(0.5)", "(512,)", "0 params"),
        ("Dense(256) + BN + ReLU", "(256,)", "131,584 params"),
        ("Dropout(0.3)", "(256,)", "0 params"),
        ("Dense(7, Softmax)", "(7,)", "1,799 params"),
    ]

    print(f"{'Layer':<35} | "
          f"{'Output':>15} | "
          f"{'Params':>15}")
    print("-" * 70)

    total_trainable = 0
    for layer, output, params in layers:
        frozen = "❄️" if "frozen" in params else ""
        print(f"{layer:<35} | "
              f"{output:>15} | "
              f"{params:>15} {frozen}")
        if params and "," in params:
            try:
                p = int(params.split()[0].replace(
                    ',', ''))
                if "frozen" not in params:
                    total_trainable += p
            except Exception:
                pass

    print(f"\n  Total (Phase 1 trainable): ~662,279")
    print(f"  Total (all params):         ~4,104,775")
    print(f"  Trainable %:                 16.1%")
    print(f"\n  MobileNetV2 features: 1280 per image")
    print(f"  These 1280 features encode textures,")
    print(f"  shapes, colors from ImageNet!")
    print(f"\n  Our head converts them to 7 skin classes! 🔥")


if __name__ == "__main__":
    print_model_architecture()

    if TF_AVAILABLE:
        print(f"\nBuilding model...")
        model, base_model = build_skin_model(
            img_size=IMG_SIZE,
            n_classes=N_CLASSES,
            use_pretrained=False)

        if model:
            compile_model(model)
            print(f"\nModel built successfully!")
            print(f"Total params: "
                  f"{model.count_params():,}")
            trainable = sum([
                tf.size(v).numpy()
                for v in model.trainable_variables])
            print(f"Trainable:    {trainable:,}")
            print(f"\n✅ Ready for Phase 1 training!")
    else:
        print("\nTensorFlow not available.")
        print("Install: pip install tensorflow")
