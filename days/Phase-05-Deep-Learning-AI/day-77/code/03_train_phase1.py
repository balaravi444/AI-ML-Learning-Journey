"""
Day 77 — Skin Disease Detector
Topic: Phase 1 Training — Feature Extraction
Date: 03 August 2026
Author: Bala Ravi

Complete Phase 1 training script.
Trains classification head on frozen MobileNetV2.
Saves best model for Phase 2 fine-tuning.
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

# Project paths
PROJECT_DIR = "projects/skin_disease_detector"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR, "checkpoints")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

# Config
IMG_SIZE = 96         # 224 for real HAM10000
BATCH_SIZE = 16       # 32 for GPU
N_CLASSES = 7
PHASE1_EPOCHS = 15    # 30-50 for real training


def run_phase1_training() -> dict:
    """
    Complete Phase 1 training pipeline.

    Steps:
    1. Generate/load data
    2. Build tf.data pipeline
    3. Build model
    4. Compute class weights
    5. Train Phase 1
    6. Evaluate
    7. Save model

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("  SKIN DISEASE DETECTOR")
    print("  Phase 1: Feature Extraction Training")
    print("=" * 60)

    if not TF_AVAILABLE:
        print("\n⚠️  TensorFlow not available.")
        print("Install: pip install tensorflow")
        print("\nExpected Phase 1 results:")
        print("  Val accuracy: ~0.72-0.78")
        print("  Val AUC:      ~0.91-0.94")
        print("  Melanoma recall: ~0.68-0.75")
        print("\nPhase 2 (Day 78) will improve these!")
        return {}

    # ── Step 1: Data ──────────────────────────────
    print(f"\n📂 Step 1: Loading Data...")

    from days_day_77_code_01_dataset_preparation import (
        generate_synthetic_dataset,
        build_tfdata_pipeline,
        compute_class_weights,
        analyze_dataset)

    analyze_dataset()

    X, y, _ = generate_synthetic_dataset(
        n_per_class=80,
        img_size=IMG_SIZE,
        save_dir=DATA_DIR)

    class_weights = compute_class_weights()

    print(f"\nClass weights:")
    for idx, w in class_weights.items():
        from days_day_77_code_02_model_architecture import (
            CLASS_NAMES)
        print(f"  [{idx}] {CLASS_NAMES[idx]:<25}: {w:.3f}")

    # ── Step 2: Pipeline ──────────────────────────
    print(f"\n⚙️  Step 2: Building tf.data Pipeline...")

    train_ds, val_ds, test_ds = (
        build_tfdata_pipeline(
            X, y,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            augment=True))

    # ── Step 3: Model ─────────────────────────────
    print(f"\n🧠 Step 3: Building MobileNetV2 Model...")

    from days_day_77_code_02_model_architecture import (
        build_skin_model,
        compile_model,
        get_callbacks,
        phase1_train,
        evaluate_model,
        print_model_architecture)

    print_model_architecture()

    model, base_model = build_skin_model(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        use_pretrained=False)  # True for real training!

    compile_model(model)

    # ── Step 4: Phase 1 Training ──────────────────
    print(f"\n🚀 Step 4: Phase 1 Training...")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    results = phase1_train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        epochs=PHASE1_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR)

    # ── Step 5: Evaluate ──────────────────────────
    print(f"\n📊 Step 5: Evaluation...")

    if test_ds is not None:
        metrics = evaluate_model(
            model, test_ds, class_weights)
    else:
        metrics = {}

    # ── Step 6: Save ──────────────────────────────
    print(f"\n💾 Step 6: Saving Phase 1 Model...")

    import joblib
    phase1_path = os.path.join(
        MODEL_DIR, 'phase1_model.h5')

    try:
        model.save(phase1_path)
        print(f"  ✅ Model saved: {phase1_path}")
    except Exception as e:
        print(f"  ⚠️  Save error: {e}")

    # Save metadata
    metadata = {
        'phase': 1,
        'img_size': IMG_SIZE,
        'n_classes': N_CLASSES,
        'class_weights': class_weights,
        'best_val_acc': results.get(
            'best_val_acc', 0),
        'epochs_run': results.get('epochs_run', 0),
        'test_metrics': metrics,
        'class_names': [
            'Actinic Keratosis',
            'Basal Cell Carcinoma',
            'Benign Keratosis',
            'Dermatofibroma',
            'Melanoma',
            'Melanocytic Nevi',
            'Vascular Lesion']
    }

    meta_path = os.path.join(
        MODEL_DIR, 'metadata.pkl')
    joblib.dump(metadata, meta_path)
    print(f"  ✅ Metadata saved: {meta_path}")

    # ── Summary ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 1 Complete!")
    print(f"{'='*60}")
    print(f"  Best val accuracy: "
          f"{results.get('best_val_acc', 0):.4f}")
    print(f"  Epochs trained:    "
          f"{results.get('epochs_run', 0)}")
    print(f"\n  Next: Day 78 → Phase 2 Fine-tuning!")
    print(f"  Unfreeze top MobileNetV2 layers")
    print(f"  LR = 1e-5, train more carefully")
    print(f"  Expected improvement: +5-8%!")
    print(f"{'='*60}")

    return {
        'model': model,
        'base_model': base_model,
        'phase1_results': results,
        'test_metrics': metrics,
        'class_weights': class_weights,
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'metadata': metadata
    }


if __name__ == "__main__":
    os.makedirs(PROJECT_DIR, exist_ok=True)
    results = run_phase1_training()
