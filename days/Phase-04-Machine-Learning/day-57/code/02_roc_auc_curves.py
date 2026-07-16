"""
Day 57 — Model Evaluation & Metrics
Topic: ROC Curve, AUC, PR Curve
Date: 14 July 2026
Author: Bala Ravi

ROC Curve = how well model RANKS predictions
AUC = area under ROC = single number summary

PR Curve = better for IMBALANCED data!
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 1000) -> tuple:
    """Create student pass/fail dataset."""
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

    return train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y)


def roc_auc_comparison() -> None:
    """Compare all classifiers using ROC AUC."""
    print("=== ROC AUC Comparison ===\n")
    print("AUC = how well model RANKS predictions")
    print("AUC = 1.0 → perfect | 0.5 → random\n")

    X_train, X_test, y_train, y_test = (
        create_student_dataset())

    classifiers = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=42, max_iter=1000))
        ]),
        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(
                max_depth=5, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=100,
                random_state=42, n_jobs=-1))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(
                kernel='rbf', C=10,
                probability=True,
                random_state=42))
        ]),
        'KNN (k=7)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(
                n_neighbors=7))
        ])
    }

    print(f"{'Classifier':<25} | {'AUC':>7} | "
          f"{'AP Score':>9} | {'F1':>7} | "
          f"{'Bar (AUC)'}")
    print("-" * 75)

    results = {}
    for name, pipeline in classifiers.items():
        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(
            X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        roc_auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'auc': roc_auc,
            'ap': ap,
            'probs': y_prob
        }

        bar = '█' * int(roc_auc * 40)
        print(f"{name:<25} | {roc_auc:>7.4f} | "
              f"{ap:>9.4f} | {f1:>7.4f} | {bar}")

    best = max(results, key=lambda x: results[x]['auc'])
    print(f"\n🏆 Best AUC: {best} "
          f"({results[best]['auc']:.4f})")


def roc_curve_analysis() -> None:
    """
    Explain ROC curve in detail.
    Show how threshold changes FPR/TPR!
    """
    print("\n=== ROC Curve Analysis ===\n")

    X_train, X_test, y_train, y_test = (
        create_student_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(
        X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"Random Forest AUC: {roc_auc:.4f}\n")

    print("Selected threshold points on ROC curve:")
    print(f"{'Threshold':>10} | {'FPR (FP Rate)':>14} | "
          f"{'TPR (Recall)':>13} | {'Interpretation'}")
    print("-" * 70)

    key_indices = [0, len(thresholds)//5,
                   len(thresholds)*2//5,
                   len(thresholds)*3//5,
                   len(thresholds)*4//5,
                   len(thresholds)-1]

    for idx in key_indices:
        idx = min(idx, len(thresholds)-1)
        t = thresholds[idx]
        f = fpr[idx]
        tp_rate = tpr[idx]

        if t > 0.8:
            interp = "Very strict — few positives"
        elif t > 0.6:
            interp = "Strict — high precision"
        elif t > 0.5:
            interp = "Above default"
        elif t > 0.3:
            interp = "Default zone"
        else:
            interp = "Loose — high recall"

        print(f"{t:>10.3f} | {f:>14.4f} | "
              f"{tp_rate:>13.4f} | {interp}")

    print(f"\n💡 Reading the ROC curve:")
    print(f"   Top-left corner = perfect model")
    print(f"   Diagonal line   = random guessing")
    print(f"   AUC closer to 1 = better model!")
    print(f"\n   FPR = FP/(FP+TN) = false alarm rate")
    print(f"   TPR = TP/(TP+FN) = recall")


def pr_curve_vs_roc() -> None:
    """
    Show when PR curve is better than ROC.
    Imbalanced data → use PR curve!
    """
    print("\n=== PR Curve vs ROC for Imbalanced Data ===\n")

    np.random.seed(42)
    n = 1000

    # Highly imbalanced: 95% negative, 5% positive
    X = np.random.randn(n, 5)
    y = (np.random.random(n) < 0.05).astype(int)

    print(f"Dataset: {y.sum()} positives out of {n}")
    print(f"Imbalance ratio: {y.mean()*100:.1f}% positive\n")

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=42,
            n_jobs=-1),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=42))
        ])
    }

    print(f"{'Model':<25} | {'ROC AUC':>8} | "
          f"{'Avg Precision':>14} | {'Note'}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(
            X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        note = ("ROC looks OK!" if roc_auc > 0.7
                else "ROC looks bad")

        print(f"{name:<25} | {roc_auc:>8.4f} | "
              f"{ap:>14.4f} | {note}")

    print(f"\n💡 For imbalanced data:")
    print(f"   ROC AUC can look decent (~0.7-0.8)")
    print(f"   But Average Precision reveals the truth!")
    print(f"   A baseline guesser gets AP = ~0.05")
    print(f"   Any model above that is actually learning!")
    print(f"\n   Rule: Imbalanced? → Use PR curve + AP! 🔥")


def optimal_threshold_finding() -> None:
    """Find optimal threshold using F1 score."""
    print("\n=== Finding Optimal Threshold ===\n")

    X_train, X_test, y_train, y_test = (
        create_student_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(
        X_test)[:, 1]

    # Try all thresholds and find best F1
    precision_arr, recall_arr, thresholds_pr = (
        precision_recall_curve(y_test, y_prob))

    from sklearn.metrics import f1_score as f1_fn

    best_f1 = 0
    best_threshold = 0.5
    best_prec = 0
    best_rec = 0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_fn(y_test, y_pred,
                    zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            from sklearn.metrics import (
                precision_score, recall_score)
            best_prec = precision_score(
                y_test, y_pred, zero_division=0)
            best_rec = recall_score(
                y_test, y_pred, zero_division=0)

    print(f"Default threshold (0.5):")
    y_default = (y_prob >= 0.5).astype(int)
    from sklearn.metrics import (
        precision_score, recall_score)
    print(f"  Precision: {precision_score(y_test, y_default):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_default):.4f}")
    print(f"  F1:        {f1_fn(y_test, y_default):.4f}")

    print(f"\nOptimal threshold ({best_threshold:.2f}):")
    print(f"  Precision: {best_prec:.4f}")
    print(f"  Recall:    {best_rec:.4f}")
    print(f"  F1:        {best_f1:.4f}")

    improvement = best_f1 - f1_fn(y_test, y_default)
    print(f"\n✅ F1 improvement: +{improvement:.4f}")
    print(f"   Just by tuning the threshold!")
    print(f"   No model change needed! 🔥")


if __name__ == "__main__":
    roc_auc_comparison()
    roc_curve_analysis()
    pr_curve_vs_roc()
    optimal_threshold_finding()
