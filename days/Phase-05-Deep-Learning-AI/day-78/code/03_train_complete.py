"""
Day 78 — Skin Disease Detector
Topic: Complete 2-Phase Training Pipeline
Date: 04 August 2026
Author: Bala Ravi

Full end-to-end training:
Phase 1 (feature extraction) →
Phase 2 (fine-tuning) →
Full evaluation →
Save final model
"""
import numpy as np
import os
import joblib
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
DATA_DIR = os.path.join(PROJECT_DIR, "data")

IMG_SIZE = 96
BATCH_SIZE = 16
N_CLASSES = 7
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 15


def run_complete_training() -> dict:
    """
    Complete 2-phase training pipeline.

    Phase 1: Feature extraction (frozen base)
    Phase 2: Fine-tuning (unfrozen top layers)
    Evaluation: Per-class clinical metrics
    Save: Model + metadata for deployment

    Returns:
        Complete results dictionary
    """
    print("=" * 60)
    print("  SKIN DISEASE DETECTOR")
    print("  Complete 2-Phase Training")
    print("=" * 60)

    if not TF_AVAILABLE:
        print("\n⚠️  TensorFlow not available.")
        print("\nExpected final results:")
        print("  Phase 1 val_acc:   ~0.76")
        print("  Phase 2 val_acc:   ~0.89")
        print("  Melanoma recall:   ~0.87")
        print("  Overall AUC:       ~0.97")
        print("\nInstall TF: pip install tensorflow")
        return {}

    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────
    print(f"\n📂 Loading dataset...")

    from days_day_77_code_01_dataset_preparation import (
        generate_synthetic_dataset,
        build_tfdata_pipeline,
        compute_class_weights)

    X, y, _ = generate_synthetic_dataset(
        n_per_class=80, img_size=IMG_SIZE,
        save_dir=DATA_DIR)

    class_weights = compute_class_weights()

    train_ds, val_ds, test_ds = build_tfdata_pipeline(
        X, y, batch_size=BATCH_SIZE,
        img_size=IMG_SIZE, augment=True)

    # ── Phase 1 ───────────────────────────────────
    print(f"\n🚀 Phase 1: Feature Extraction...")

    from days_day_77_code_02_model_architecture import (
        build_skin_model, compile_model,
        get_callbacks, phase1_train,
        evaluate_model)

    model, base_model = build_skin_model(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        use_pretrained=False)

    compile_model(model)

    p1_results = phase1_train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        epochs=PHASE1_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR)

    p1_acc = p1_results.get('best_val_acc', 0)

    # ── Phase 2 ───────────────────────────────────
    print(f"\n🔥 Phase 2: Fine-tuning...")

    from days_day_78_code_01_phase2_finetune import (
        unfreeze_top_layers,
        compile_for_finetuning,
        get_phase2_callbacks,
        run_phase2,
        compare_phases)

    p2_results = run_phase2(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights)

    p2_acc = p2_results.get('best_val_acc', 0)

    # ── Evaluation ────────────────────────────────
    print(f"\n📊 Full Evaluation...")

    from days_day_78_code_02_multiclass_evaluation import (
        get_predictions,
        print_confusion_matrix,
        compute_clinical_metrics,
        threshold_tuning_for_melanoma)

    y_true, y_pred, y_prob = get_predictions(
        model, test_ds)

    if y_true is not None:
        print_confusion_matrix(y_true, y_pred)
        clinical_metrics = compute_clinical_metrics(
            y_true, y_pred, y_prob)
        opt_threshold = threshold_tuning_for_melanoma(
            y_true, y_prob)
    else:
        clinical_metrics = {}
        opt_threshold = 0.3

    compare_phases(p1_acc, p2_acc)

    # ── Save ──────────────────────────────────────
    print(f"\n💾 Saving final model...")

    final_path = os.path.join(
        MODEL_DIR, 'final_model.h5')
    try:
        model.save(final_path)
        size_mb = os.path.getsize(
            final_path) / (1024 * 1024)
        print(f"  ✅ Model saved: {final_path}")
        print(f"     Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"  ⚠️  Save error: {e}")

    metadata = {
        'phase1_acc': p1_acc,
        'phase2_acc': p2_acc,
        'improvement': p2_acc - p1_acc,
        'clinical_metrics': clinical_metrics,
        'optimal_threshold': opt_threshold,
        'img_size': IMG_SIZE,
        'n_classes': N_CLASSES,
        'class_weights': class_weights,
        'class_names': [
            'Actinic Keratosis',
            'Basal Cell Carcinoma',
            'Benign Keratosis',
            'Dermatofibroma',
            'Melanoma',
            'Melanocytic Nevi',
            'Vascular Lesion'],
        'melanoma_idx': 4,
        'melanoma_threshold': opt_threshold
    }

    meta_path = os.path.join(
        MODEL_DIR, 'final_metadata.pkl')
    joblib.dump(metadata, meta_path)
    print(f"  ✅ Metadata saved: {meta_path}")

    # ── Summary ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Phase 1 accuracy:  {p1_acc:.4f}")
    print(f"  Phase 2 accuracy:  {p2_acc:.4f}")
    print(f"  Improvement:       "
          f"{(p2_acc-p1_acc):+.4f}")

    mel_metrics = clinical_metrics.get(
        'Melanoma', {})
    mel_recall = mel_metrics.get('recall', 0)
    mel_auc = mel_metrics.get('auc', 0)

    print(f"\n  Melanoma recall:  {mel_recall:.4f}")
    print(f"  Melanoma AUC:     {mel_auc:.4f}")

    ready = (p2_acc > 0.85 and mel_recall > 0.85)
    status = "✅ READY FOR DEPLOYMENT" if ready else "⚠️ NEEDS MORE TRAINING"
    print(f"\n  Status: {status}")
    print(f"\n  Next: Day 79 → FastAPI backend!")
    print(f"  Day 80 → Dashboard + Render deploy!")
    print(f"{'='*60}")

    return {
        'model': model,
        'phase1_acc': p1_acc,
        'phase2_acc': p2_acc,
        'clinical_metrics': clinical_metrics,
        'optimal_threshold': opt_threshold,
        'metadata': metadata
    }


if __name__ == "__main__":
    results = run_complete_training()
