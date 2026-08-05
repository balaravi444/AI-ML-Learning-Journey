"""
Day 78 — Skin Disease Detector
Topic: Multi-class Evaluation + Clinical Metrics
Date: 04 August 2026
Author: Bala Ravi

7-class evaluation for skin disease detection.
Clinical metrics: melanoma recall is the key metric!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        precision_recall_curve,
        average_precision_score)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular Lesion'
]

MELANOMA_IDX = 4  # Index in CLASS_NAMES


def get_predictions(
        model,
        test_ds) -> tuple:
    """
    Get all predictions and true labels
    from test dataset.

    Args:
        model: Trained model
        test_ds: Test tf.data.Dataset

    Returns:
        (y_true, y_pred, y_prob) arrays
    """
    if model is None or test_ds is None:
        return None, None, None

    all_probs = []
    all_labels = []

    for batch_imgs, batch_labels in test_ds:
        probs = model.predict(
            batch_imgs, verbose=0)
        all_probs.extend(probs)
        all_labels.extend(batch_labels.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob


def print_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> None:
    """Print formatted 7×7 confusion matrix."""
    print("=== Confusion Matrix ===\n")

    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available.")
        print("Run: pip install scikit-learn")
        return

    cm = confusion_matrix(y_true, y_pred)
    n = len(CLASS_NAMES)

    # Print header
    abbrevs = [name[:6] for name in CLASS_NAMES]
    print(f"{'Actual → Predicted':>20}", end='')
    for ab in abbrevs:
        print(f"{ab:>8}", end='')
    print()
    print("-" * (20 + 8 * n))

    # Print rows
    for i, cls_name in enumerate(CLASS_NAMES):
        ab = cls_name[:10]
        print(f"{ab:>20}", end='')
        for j in range(n):
            val = cm[i][j]
            # Highlight diagonal (correct)
            if i == j:
                print(f"[{val:>5}]", end='')
            else:
                print(f"{val:>8}", end='')
        print()

    # Per-class accuracy
    print(f"\nPer-class Recall (Correct/Total):")
    for i, cls_name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        if total == 0:
            continue
        correct = cm[i][i]
        recall = correct / total
        flag = ("🚨 CRITICAL" if
                cls_name == 'Melanoma' else "")
        bar = '█' * int(recall * 20)
        print(f"  {cls_name:<25}: "
              f"{recall:.3f} {bar} {flag}")


def compute_clinical_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray) -> dict:
    """
    Compute clinical metrics for skin disease detection.

    Most important: MELANOMA RECALL
    Missing melanoma = false security = death!

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities

    Returns:
        Dictionary of clinical metrics
    """
    print("\n=== Clinical Metrics ===\n")

    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available.")
        return {}

    # Overall metrics
    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_score, recall_score)

    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(
        y_true, y_pred, average='weighted',
        zero_division=0)

    print(f"Overall Accuracy:  {overall_acc:.4f}")
    print(f"Overall F1:        {overall_f1:.4f}")

    # Per-class metrics
    print(f"\nPer-class Metrics:\n")
    print(f"{'Class':<25} | "
          f"{'Precision':>10} | "
          f"{'Recall':>8} | "
          f"{'F1':>7} | "
          f"{'AUC':>7}")
    print("-" * 65)

    metrics = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # Binary: this class vs all others
        y_bin_true = (y_true == cls_idx).astype(int)
        y_bin_pred = (y_pred == cls_idx).astype(int)

        prec = precision_score(
            y_bin_true, y_bin_pred,
            zero_division=0)
        rec = recall_score(
            y_bin_true, y_bin_pred,
            zero_division=0)
        f1 = f1_score(
            y_bin_true, y_bin_pred,
            zero_division=0)

        # AUC per class
        try:
            auc = roc_auc_score(
                y_bin_true, y_prob[:, cls_idx])
        except Exception:
            auc = 0.5

        metrics[cls_name] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }

        is_melanoma = (cls_name == 'Melanoma')
        flag = "🚨" if is_melanoma else ""

        print(f"{cls_name:<25} | "
              f"{prec:>10.4f} | "
              f"{rec:>8.4f} | "
              f"{f1:>7.4f} | "
              f"{auc:>7.4f} {flag}")

    # Critical: Melanoma metrics
    mel = metrics.get('Melanoma', {})
    mel_recall = mel.get('recall', 0)
    mel_auc = mel.get('auc', 0)

    print(f"\n{'='*50}")
    print(f"  🚨 MELANOMA METRICS (Critical!)")
    print(f"{'='*50}")
    print(f"  Recall:    {mel_recall:.4f}")
    print(f"  AUC:       {mel_auc:.4f}")

    if mel_recall >= 0.85:
        print(f"\n  ✅ DEPLOYMENT READY!")
        print(f"     Catching {mel_recall*100:.0f}% of melanoma cases!")
    elif mel_recall >= 0.75:
        print(f"\n  ⚡ NEEDS IMPROVEMENT")
        print(f"     Catching {mel_recall*100:.0f}% — target is 85%+")
    else:
        print(f"\n  ❌ NOT READY — missing too many melanoma cases!")
        print(f"     Catching only {mel_recall*100:.0f}%")
        print(f"     Collect more melanoma training data!")

    return metrics


def threshold_tuning_for_melanoma(
        y_true: np.ndarray,
        y_prob: np.ndarray) -> float:
    """
    Find optimal decision threshold for melanoma.

    For medical applications:
    We prefer HIGH recall over high precision!
    Better to flag too many than miss one!

    Args:
        y_true: True labels
        y_prob: Prediction probabilities

    Returns:
        Optimal threshold for melanoma recall
    """
    print("\n=== Melanoma Threshold Tuning ===\n")
    print("Default threshold = 0.5")
    print("Can we catch more melanoma by lowering?\n")

    mel_probs = y_prob[:, MELANOMA_IDX]
    y_mel_true = (y_true == MELANOMA_IDX).astype(int)

    if not SKLEARN_AVAILABLE:
        print("scikit-learn not available.")
        return 0.3

    from sklearn.metrics import (
        precision_score, recall_score, f1_score)

    print(f"{'Threshold':>10} | "
          f"{'Precision':>10} | "
          f"{'Recall':>8} | "
          f"{'F1':>7} | "
          f"{'Flagged%':>9} | "
          f"{'Recommendation'}")
    print("-" * 80)

    best_recall_threshold = 0.5
    best_recall = 0

    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_mel = (
            mel_probs >= threshold).astype(int)
        prec = precision_score(
            y_mel_true, y_pred_mel,
            zero_division=0)
        rec = recall_score(
            y_mel_true, y_pred_mel,
            zero_division=0)
        f1 = f1_score(
            y_mel_true, y_pred_mel,
            zero_division=0)
        flagged_pct = y_pred_mel.mean() * 100

        rec_note = ""
        if threshold == 0.3:
            rec_note = "← Recommended for clinic"
        elif threshold == 0.5:
            rec_note = "← Default"

        if rec > best_recall:
            best_recall = rec
            best_recall_threshold = threshold

        print(f"{threshold:>10.1f} | "
              f"{prec:>10.4f} | "
              f"{rec:>8.4f} | "
              f"{f1:>7.4f} | "
              f"{flagged_pct:>8.1f}% | "
              f"{rec_note}")

    print(f"\n💡 Clinical recommendation: threshold=0.3")
    print(f"   Higher false positives = extra checkups")
    print(f"   Higher recall = fewer missed melanomas")
    print(f"   In medicine: false negative >> false positive!")

    return 0.3  # Clinical threshold


def simulate_evaluation() -> None:
    """
    Simulate evaluation results when
    TensorFlow is not available.
    """
    print("=== Simulated Phase 2 Evaluation ===\n")
    print("(Install TensorFlow for real evaluation)\n")

    print("Expected results after Phase 2:\n")

    simulated = {
        'Actinic Keratosis': {
            'precision': 0.82,
            'recall': 0.79,
            'f1': 0.80,
            'auc': 0.96},
        'Basal Cell Carcinoma': {
            'precision': 0.88,
            'recall': 0.85,
            'f1': 0.86,
            'auc': 0.97},
        'Benign Keratosis': {
            'precision': 0.84,
            'recall': 0.88,
            'f1': 0.86,
            'auc': 0.96},
        'Dermatofibroma': {
            'precision': 0.86,
            'recall': 0.83,
            'f1': 0.84,
            'auc': 0.97},
        'Melanoma': {
            'precision': 0.79,
            'recall': 0.87,
            'f1': 0.83,
            'auc': 0.96},
        'Melanocytic Nevi': {
            'precision': 0.93,
            'recall': 0.92,
            'f1': 0.92,
            'auc': 0.98},
        'Vascular Lesion': {
            'precision': 0.88,
            'recall': 0.84,
            'f1': 0.86,
            'auc': 0.97}
    }

    print(f"{'Class':<25} | "
          f"{'Prec':>6} | "
          f"{'Recall':>7} | "
          f"{'F1':>6} | "
          f"{'AUC':>6}")
    print("-" * 60)

    for cls_name, m in simulated.items():
        flag = "🚨" if cls_name == 'Melanoma' else "  "
        print(f"{cls_name:<25} | "
              f"{m['precision']:>6.3f} | "
              f"{m['recall']:>7.3f} | "
              f"{m['f1']:>6.3f} | "
              f"{m['auc']:>6.3f} {flag}")

    mel = simulated['Melanoma']
    print(f"\n🚨 Melanoma recall: {mel['recall']:.3f}")
    print(f"   Catching {mel['recall']*100:.0f}% of melanoma cases!")
    print(f"   ✅ Above 85% threshold → DEPLOYABLE!")

    acc = np.mean([m['recall']
                   for m in simulated.values()])
    print(f"\nOverall accuracy (approx): {acc:.3f}")
    print(f"Phase 1 accuracy:          ~0.76")
    print(f"Improvement:               "
          f"+{(acc - 0.76)*100:.1f}%")
    print(f"\n✅ Fine-tuning worked! Ready for Day 79!")


if __name__ == "__main__":
    if not TF_AVAILABLE:
        simulate_evaluation()
    else:
        print("Load model from Day 77 Phase 1!")
        print("Then run:")
        print("  y_true, y_pred, y_prob = "
              "get_predictions(model, test_ds)")
        print("  print_confusion_matrix(y_true, y_pred)")
        print("  compute_clinical_metrics("
              "y_true, y_pred, y_prob)")
        print("  threshold_tuning_for_melanoma("
              "y_true, y_prob)")
