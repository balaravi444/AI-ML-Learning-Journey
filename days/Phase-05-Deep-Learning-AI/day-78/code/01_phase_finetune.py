"""
Day 78 — Skin Disease Detector
Topic: Phase 2 Fine-tuning
Date: 04 August 2026
Author: Bala Ravi

Unfreeze top MobileNetV2 layers.
Very low LR to avoid catastrophic forgetting.
Phase 1: 0.76 → Phase 2: target 0.88-0.92
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


PROJECT_DIR = "projects/skin_disease_detector"
CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR, "checkpoints")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

IMG_SIZE = 96
N_CLASSES = 7
UNFREEZE_FROM = 100    # freeze layers 0-99
PHASE2_LR = 1e-5       # 100× smaller than Phase 1
PHASE2_EPOCHS = 20


def unfreeze_top_layers(
        model,
        base_model,
        unfreeze_from: int = UNFREEZE_FROM
        ) -> None:
    """
    Unfreeze top layers of MobileNetV2 for fine-tuning.

    Strategy:
    - Keep bottom layers frozen (universal features)
    - Unfreeze top layers (task-specific features)
    - Use very low LR to prevent forgetting

    Args:
        model: Full model
        base_model: MobileNetV2 base
        unfreeze_from: Layer index to start unfreezing
    """
    if model is None or base_model is None:
        return

    # Unfreeze base model
    base_model.trainable = True

    # Freeze bottom layers
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    # Keep BatchNorm frozen during fine-tuning!
    # Unfreezing BN can destabilize training
    for layer in base_model.layers:
        if isinstance(
                layer, keras.layers.BatchNormalization):
            layer.trainable = False

    total = len(base_model.layers)
    frozen = sum(
        1 for l in base_model.layers
        if not l.trainable)
    trainable = total - frozen

    print(f"  Base model layers: {total}")
    print(f"  Frozen layers:     {frozen}")
    print(f"  Unfrozen layers:   {trainable}")
    print(f"  Unfrozen from:     layer {unfreeze_from}")

    # Show trainable parameter count
    total_params = model.count_params()
    trainable_params = sum([
        tf.size(v).numpy()
        for v in model.trainable_variables])

    print(f"\n  Total params:     {total_params:,}")
    print(f"  Trainable:        {trainable_params:,}")
    print(f"  Trainable %:      "
          f"{trainable_params/total_params*100:.1f}%")


def compile_for_finetuning(model) -> None:
    """
    Recompile model with very low LR for Phase 2.

    CRITICAL: Must recompile after unfreezing!
    Use 100× smaller LR than Phase 1!
    """
    if model is None:
        return

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=PHASE2_LR),
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Recall(name='recall')
        ])

    print(f"\n  Compiled with LR = {PHASE2_LR}")
    print(f"  (100× smaller than Phase 1)")
    print(f"  Gentle fine-tuning — no forgetting!")


def get_phase2_callbacks(
        checkpoint_dir: str = CHECKPOINT_DIR
        ) -> list:
    """
    Callbacks for Phase 2.
    More patient than Phase 1 — fine-tuning is slower.
    """
    if not TF_AVAILABLE:
        return []

    os.makedirs(checkpoint_dir, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=12,
            restore_best_weights=True,
            min_delta=0.0005,
            verbose=1),

        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                checkpoint_dir,
                'phase2_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1),

        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            cooldown=3,
            verbose=1),

        keras.callbacks.CSVLogger(
            os.path.join(
                checkpoint_dir,
                'phase2_log.csv'),
            append=True)
    ]


def run_phase2(
        model,
        base_model,
        train_ds,
        val_ds,
        class_weights: dict) -> dict:
    """
    Execute complete Phase 2 fine-tuning.

    Args:
        model: Model from Phase 1
        base_model: MobileNetV2 base
        train_ds: Training dataset
        val_ds: Validation dataset
        class_weights: Balanced class weights

    Returns:
        Phase 2 training results
    """
    if model is None:
        return {}

    print(f"\n{'='*55}")
    print(f"  Phase 2: Fine-tuning")
    print(f"  Unfreezing top {N_CLASSES} layers")
    print(f"  LR = {PHASE2_LR} (very low!)")
    print(f"{'='*55}\n")

    # Step 1: Unfreeze
    print("Unfreezing top layers...")
    unfreeze_top_layers(
        model, base_model,
        unfreeze_from=UNFREEZE_FROM)

    # Step 2: Recompile
    print("\nRecompiling with low LR...")
    compile_for_finetuning(model)

    # Step 3: Train
    print(f"\nPhase 2 training...")
    callbacks = get_phase2_callbacks()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1)

    best_val_acc = max(
        history.history.get('val_accuracy', [0]))
    epochs_run = len(
        history.history.get('loss', []))

    print(f"\n  Phase 2 Complete!")
    print(f"  Epochs:       {epochs_run}")
    print(f"  Best val acc: {best_val_acc:.4f}")

    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'epochs_run': epochs_run
    }


def compare_phases(
        phase1_acc: float,
        phase2_acc: float) -> None:
    """Compare Phase 1 vs Phase 2 results."""
    print(f"\n=== Phase Comparison ===\n")

    improvement = phase2_acc - phase1_acc

    print(f"{'Phase':<20} | {'Val Accuracy':>13}")
    print("-" * 37)
    print(f"{'Phase 1 (frozen)':<20} | "
          f"{phase1_acc:>13.4f}")
    print(f"{'Phase 2 (fine-tune)':<20} | "
          f"{phase2_acc:>13.4f}")
    print(f"{'Improvement':<20} | "
          f"{improvement:>+13.4f}")

    if improvement > 0.05:
        print(f"\n🔥 Excellent fine-tuning!")
        print(f"   +{improvement*100:.1f}% accuracy gain!")
    elif improvement > 0.02:
        print(f"\n✅ Good improvement!")
        print(f"   +{improvement*100:.1f}% accuracy gain!")
    elif improvement > 0:
        print(f"\n⚡ Marginal improvement")
        print(f"   +{improvement*100:.1f}% — consider more epochs")
    else:
        print(f"\n⚠️  No improvement in Phase 2")
        print(f"   Try: lower LR, more Phase 1 epochs")

    print(f"\n  For production deployment:")
    print(f"  Need val_accuracy > 0.85")
    print(f"  Need melanoma recall > 0.85")
    status = "✅ READY" if phase2_acc > 0.85 else "⚠️ NEEDS MORE TRAINING"
    print(f"  Status: {status}")


if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("Phase 2 Fine-tuning Summary:\n")
        print("What happens:")
        print("  1. Load Phase 1 model")
        print("  2. Unfreeze layers 100-153 of MobileNetV2")
        print("  3. Recompile with LR=1e-5")
        print("  4. Train 20 more epochs")
        print("  5. EarlyStopping protects best weights")
        print("\nExpected results:")
        print("  Phase 1: val_acc ≈ 0.76")
        print("  Phase 2: val_acc ≈ 0.88-0.92")
        print("  Improvement: +12-16%!")
        print("\nInstall TensorFlow to run:")
        print("  pip install tensorflow")
    else:
        print("Run 03_train_complete.py for full pipeline!")
