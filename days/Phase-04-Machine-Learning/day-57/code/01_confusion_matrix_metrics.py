"""
Day 57 — Model Evaluation & Metrics
Topic: Confusion Matrix + Core Metrics from Scratch
Date: 14 July 2026
Author: Bala Ravi

Every metric comes from 4 numbers: TN, TP, FP, FN
Understanding these = understanding ALL evaluation!
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def confusion_matrix_from_scratch(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> dict:
    """
    Build confusion matrix from scratch.
    Shows exactly what TN/TP/FP/FN mean!

    Args:
        y_true: Actual labels
        y_pred: Predicted labels

    Returns:
        Dictionary with all metrics
    """
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    n = len(y_true)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall /
           (precision + recall)
           if (precision + recall) > 0 else 0)
    accuracy = (TP + TN) / n
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {
        'TP': int(TP), 'TN': int(TN),
        'FP': int(FP), 'FN': int(FN),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1
    }


def demonstrate_confusion_matrix() -> None:
    """Show confusion matrix with real examples."""
    print("=== Confusion Matrix Deep Dive ===\n")

    # Simulated student pass/fail predictions
    y_true = np.array([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([
        1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0])

    metrics = confusion_matrix_from_scratch(
        y_true, y_pred)

    print("Confusion Matrix:")
    print(f"┌──────────────────────────────┐")
    print(f"│          Predicted           │")
    print(f"│         FAIL    PASS         │")
    print(f"│ Actual                       │")
    print(f"│  FAIL    TN={metrics['TN']:>3}   FP={metrics['FP']:>3} │")
    print(f"│  PASS    FN={metrics['FN']:>3}   TP={metrics['TP']:>3} │")
    print(f"└──────────────────────────────┘")

    print(f"\nWhat each cell means:")
    print(f"  TN={metrics['TN']} → correctly predicted FAIL ✅")
    print(f"  TP={metrics['TP']} → correctly predicted PASS ✅")
    print(f"  FP={metrics['FP']} → predicted PASS but actually FAIL ⚠️")
    print(f"      (false alarm — unnecessary intervention)")
    print(f"  FN={metrics['FN']} → predicted FAIL but actually PASS ❌")
    print(f"      (missed at-risk students — most dangerous!)")

    print(f"\nDerived Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} "
          f"({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision:   {metrics['precision']:.4f} "
          f"(of predicted PASS, {metrics['precision']*100:.1f}% actually passed)")
    print(f"  Recall:      {metrics['recall']:.4f} "
          f"(caught {metrics['recall']*100:.1f}% of actual PASS students)")
    print(f"  Specificity: {metrics['specificity']:.4f} "
          f"(correctly identified {metrics['specificity']*100:.1f}% of FAIL)")
    print(f"  F1 Score:    {metrics['f1']:.4f}")


def precision_recall_tradeoff() -> None:
    """
    Show precision-recall tradeoff.
    Changing threshold moves along the curve!
    """
    print("\n=== Precision-Recall Tradeoff ===\n")

    np.random.seed(42)
    n = 200

    # Generate realistic probability scores
    y_true = np.array([1] * 80 + [0] * 120)
    y_prob = np.concatenate([
        np.random.beta(8, 3, 80),   # pass probs (high)
        np.random.beta(2, 6, 120)   # fail probs (low)
    ])

    print(f"{'Threshold':>10} | {'Precision':>10} | "
          f"{'Recall':>8} | {'F1':>7} | "
          f"{'Predicted+':>11} | {'Use Case'}")
    print("-" * 80)

    thresholds = [0.2, 0.3, 0.4, 0.5,
                  0.6, 0.7, 0.8]
    use_cases = [
        "Catch EVERYONE (many false alarms)",
        "Medical screening",
        "High recall focus",
        "Default balanced",
        "High precision focus",
        "Strict — low false alarms",
        "Very strict — many misses"
    ]

    best_f1 = 0
    best_threshold = 0.5

    for threshold, use_case in zip(
            thresholds, use_cases):
        y_pred = (y_prob >= threshold).astype(int)
        prec = precision_score(
            y_true, y_pred, zero_division=0)
        rec = recall_score(
            y_true, y_pred, zero_division=0)
        f1 = f1_score(
            y_true, y_pred, zero_division=0)
        n_pos = y_pred.sum()

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        print(f"{threshold:>10.1f} | {prec:>10.4f} | "
              f"{rec:>8.4f} | {f1:>7.4f} | "
              f"{n_pos:>11} | {use_case}")

    print(f"\n✅ Best F1 at threshold={best_threshold}")
    print(f"   But 'best' depends on your problem!")
    print(f"   Medical: use threshold=0.3 (high recall)")
    print(f"   Spam:    use threshold=0.7 (high precision)")


def accuracy_trap_demo() -> None:
    """Show why accuracy is misleading on imbalanced data."""
    print("\n=== The Accuracy Trap ===\n")

    # Highly imbalanced — 95% pass, 5% fail
    y_true = np.array([1] * 190 + [0] * 10)

    # Dumb model: always predict PASS
    y_pred_dumb = np.ones(200, dtype=int)

    # Real model: actually tries
    np.random.seed(42)
    y_pred_real = y_true.copy()
    # Miss 30% of fails and 5% of passes
    fail_idx = np.where(y_true == 0)[0]
    pass_idx = np.where(y_true == 1)[0]
    y_pred_real[np.random.choice(
        fail_idx, 3)] = 1
    y_pred_real[np.random.choice(
        pass_idx, 10)] = 0

    print("Dataset: 190 PASS, 10 FAIL (imbalanced!)\n")
    print(f"{'Metric':<15} | {'Dumb Model':>12} | "
          f"{'Real Model':>12}")
    print("-" * 45)

    metrics_fns = [
        ('Accuracy', accuracy_score),
        ('Precision',
         lambda t, p: precision_score(
             t, p, zero_division=0)),
        ('Recall',
         lambda t, p: recall_score(
             t, p, zero_division=0)),
        ('F1',
         lambda t, p: f1_score(
             t, p, zero_division=0))
    ]

    for name, fn in metrics_fns:
        dumb_score = fn(y_true, y_pred_dumb)
        real_score = fn(y_true, y_pred_real)
        flag = ("⚠️" if (name == 'Accuracy' and
                         abs(dumb_score - real_score) < 0.05)
                else "")
        print(f"{name:<15} | {dumb_score:>12.4f} | "
              f"{real_score:>12.4f} {flag}")

    print(f"\n💡 Dumb model has 95% accuracy!")
    print(f"   But F1=0 — it catches NO fail students!")
    print(f"   Accuracy alone is DANGEROUS here!")
    print(f"\n   Always report F1/Precision/Recall")
    print(f"   on imbalanced datasets! 🔥")


def create_student_dataset(n: int = 1000):
    """Create student dataset."""
    np.random.seed(42)

    study = np.random.uniform(1, 10, n)
    attend = np.random.uniform(40, 100, n)
    prev = np.random.uniform(30, 95, n)
    assign = np.random.randint(0, 10, n)
    sleep = np.random.uniform(4, 9, n)

    score = (study * 5 + attend * 0.3 +
              prev * 0.4 + assign * 1.5 +
              sleep * 1.0 +
              np.random.normal(0, 5, n))

    X = np.column_stack([study, attend,
                          prev, assign, sleep])
    y = (score > 70).astype(int)
    return X, y


if __name__ == "__main__":
    demonstrate_confusion_matrix()
    precision_recall_tradeoff()
    accuracy_trap_demo()
