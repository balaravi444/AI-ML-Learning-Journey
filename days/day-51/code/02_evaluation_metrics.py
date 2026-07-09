"""
Day 51 — ML Fundamentals + Scikit-learn
Topic: Model Evaluation Metrics
Date: 08 July 2026
Author: Bala Ravi

Right metric = right conclusions!
Wrong metric = misleading results!
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    # Regression
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression, LogisticRegression)
from sklearn.preprocessing import StandardScaler


def regression_metrics_demo() -> None:
    """Demonstrate all regression metrics."""
    print("=== Regression Metrics ===\n")

    np.random.seed(42)
    n = 100

    y_true = np.random.normal(50, 15, n)
    y_pred_good = y_true + np.random.normal(0, 5, n)
    y_pred_bad = y_true + np.random.normal(0, 20, n)

    print(f"{'Metric':<15} | {'Good Model':>12} | "
          f"{'Bad Model':>12} | {'Better':>8}")
    print("-" * 55)

    # R² Score
    r2_good = r2_score(y_true, y_pred_good)
    r2_bad = r2_score(y_true, y_pred_bad)
    print(f"{'R² Score':<15} | {r2_good:>12.4f} | "
          f"{r2_bad:>12.4f} | {'Higher ↑':>8}")

    # MAE
    mae_good = mean_absolute_error(
        y_true, y_pred_good)
    mae_bad = mean_absolute_error(
        y_true, y_pred_bad)
    print(f"{'MAE':<15} | {mae_good:>12.4f} | "
          f"{mae_bad:>12.4f} | {'Lower ↓':>8}")

    # RMSE
    rmse_good = np.sqrt(mean_squared_error(
        y_true, y_pred_good))
    rmse_bad = np.sqrt(mean_squared_error(
        y_true, y_pred_bad))
    print(f"{'RMSE':<15} | {rmse_good:>12.4f} | "
          f"{rmse_bad:>12.4f} | {'Lower ↓':>8}")

    print("\n📌 When to use which:")
    print("  R²   → How much variance explained?")
    print("         1.0 = perfect, 0.0 = mean baseline")
    print("  MAE  → Average error (same unit as target)")
    print("         Robust to outliers")
    print("  RMSE → Penalizes large errors more")
    print("         Use when large errors are costly!")


def classification_metrics_demo() -> None:
    """Demonstrate all classification metrics."""
    print("\n=== Classification Metrics ===\n")

    np.random.seed(42)

    # Simulated student pass/fail
    y_true = np.array(
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
         0, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array(
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
         0, 0, 0, 1, 1, 0, 1, 1, 0, 0])

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print("\n📌 Intuition:")
    print("  Accuracy:  Overall correct predictions")
    print("  Precision: Of predicted Pass, how many "
          "actually passed?")
    print("  Recall:    Of actual Pass students, "
          "how many did we catch?")
    print("  F1:        Balance between precision "
          "and recall!")

    print("\n⚕️ Medical diagnosis example:")
    print("  Disease detection: HIGH RECALL needed!")
    print("  (Better to have false alarm than "
          "miss real disease)")
    print("\n📧 Spam detection example:")
    print("  PRECISION more important!")
    print("  (Better to miss spam than block "
          "real email)")


def imbalanced_data_demo() -> None:
    """Show why accuracy fails on imbalanced data."""
    print("\n=== Imbalanced Data Problem ===\n")

    np.random.seed(42)

    # 95% not fraud, 5% fraud (imbalanced!)
    n = 1000
    y_true = np.array(
        [0] * 950 + [1] * 50)

    # Dummy model: always predict "not fraud"
    y_pred_dummy = np.zeros(n, dtype=int)

    # Real model: actually detects some fraud
    y_pred_real = y_true.copy()
    # Miss 20 fraud cases
    fraud_idx = np.where(y_true == 1)[0]
    y_pred_real[fraud_idx[:20]] = 0

    print("Fraud Detection (5% fraud rate)")
    print("-" * 45)

    for name, y_pred in [
        ("Dummy (always 'not fraud')",
         y_pred_dummy),
        ("Real model", y_pred_real)
    ]:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(
            y_true, y_pred, zero_division=0)
        rec = recall_score(
            y_true, y_pred, zero_division=0)
        f1 = f1_score(
            y_true, y_pred, zero_division=0)
        print(f"\n{name}:")
        print(f"  Accuracy:  {acc:.1%} "
              f"{'← Misleading!' if acc > 0.9 and name.startswith('D') else ''}")
        print(f"  Precision: {prec:.1%}")
        print(f"  Recall:    {rec:.1%}")
        print(f"  F1 Score:  {f1:.1%}")

    print("\n💡 Lesson:")
    print("  Dummy model: 95% accuracy but USELESS!")
    print("  Real model:  Lower accuracy but "
          "ACTUALLY catches fraud!")
    print("  ✅ Always use F1/Recall for "
          "imbalanced data!")


def cross_validation_demo() -> None:
    """Show why cross-validation is better."""
    print("\n=== Cross-Validation ===\n")

    from sklearn.model_selection import (
        cross_val_score, KFold)
    from sklearn.ensemble import (
        RandomForestRegressor)

    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (3 * X[:, 0] + 2 * X[:, 1] +
         np.random.randn(200))

    model = RandomForestRegressor(
        n_estimators=50, random_state=42)

    # Single train/test split
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))
    model.fit(X_train, y_train)
    single_score = r2_score(
        y_test, model.predict(X_test))

    # 5-fold cross-validation
    cv_scores = cross_val_score(
        RandomForestRegressor(
            n_estimators=50, random_state=42),
        X, y, cv=5, scoring='r2')

    print("Single Train/Test Split:")
    print(f"  R² = {single_score:.4f}")
    print(f"  ⚠️ Could be lucky or unlucky split!")

    print("\n5-Fold Cross-Validation:")
    print(f"  Scores: {cv_scores.round(3)}")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std:  {cv_scores.std():.4f}")
    print(f"  ✅ Much more reliable estimate!")

    print("\n💡 Always use CV for:")
    print("   Model selection, hyperparameter tuning")
    print("   Reporting final model performance")


if __name__ == "__main__":
    regression_metrics_demo()
    classification_metrics_demo()
    imbalanced_data_demo()
    cross_validation_demo()
