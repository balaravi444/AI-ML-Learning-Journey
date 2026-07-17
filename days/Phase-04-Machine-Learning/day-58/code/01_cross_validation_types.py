"""
Day 58 — Cross Validation + Hyperparameter Tuning
Topic: Cross Validation Types
Date: 15 July 2026
Author: Bala Ravi

Single split = unreliable
Cross-validation = stable, trustworthy estimates!
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    cross_val_score,
    cross_validate)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 500) -> tuple:
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


def single_split_unreliability() -> None:
    """Show why single split is unreliable."""
    print("=== Single Split Unreliability ===\n")

    X, y = create_student_dataset()

    from sklearn.model_selection import train_test_split

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=42, max_iter=1000))
    ])

    print("Accuracy with different random splits:")
    print(f"{'Seed':>6} | {'Train Acc':>10} | "
          f"{'Test Acc':>9} | {'Diff':>7}")
    print("-" * 40)

    scores = []
    for seed in range(10):
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X, y, test_size=0.2,
                random_state=seed))
        pipeline.fit(X_train, y_train)
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        scores.append(test_acc)
        print(f"{seed:>6} | {train_acc:>10.4f} | "
              f"{test_acc:>9.4f} | "
              f"{test_acc-train_acc:>7.4f}")

    print(f"\nTest accuracy range:")
    print(f"  Min:  {min(scores):.4f}")
    print(f"  Max:  {max(scores):.4f}")
    print(f"  Diff: {max(scores)-min(scores):.4f}")
    print(f"\n⚠️  Same model, same data — {(max(scores)-min(scores))*100:.1f}% variation!")
    print(f"   Which split do you trust? None!")
    print(f"\n✅ Solution: Cross-Validation!")


def kfold_vs_stratified() -> None:
    """Show why Stratified is better for classification."""
    print("\n=== KFold vs Stratified KFold ===\n")

    X, y = create_student_dataset(n=300)

    print(f"Overall class distribution:")
    print(f"  Pass: {y.mean()*100:.1f}%")
    print(f"  Fail: {(1-y).mean()*100:.1f}%\n")

    kf = KFold(n_splits=5, shuffle=True,
                random_state=42)
    skf = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=42, max_iter=1000))
    ])

    print("KFold — class distribution per fold:")
    print(f"{'Fold':>5} | {'Pass%':>6} | "
          f"{'Fail%':>6} | {'Test Score':>11}")
    print("-" * 35)

    for fold, (train_idx, test_idx) in enumerate(
            kf.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipeline.fit(X_tr, y_tr)
        score = pipeline.score(X_te, y_te)
        pass_pct = y_te.mean() * 100
        print(f"{fold:>5} | {pass_pct:>6.1f} | "
              f"{100-pass_pct:>6.1f} | "
              f"{score:>11.4f}")

    print(f"\nStratified KFold — each fold respects ratio:")
    print(f"{'Fold':>5} | {'Pass%':>6} | "
          f"{'Fail%':>6} | {'Test Score':>11}")
    print("-" * 35)

    for fold, (train_idx, test_idx) in enumerate(
            skf.split(X, y), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipeline.fit(X_tr, y_tr)
        score = pipeline.score(X_te, y_te)
        pass_pct = y_te.mean() * 100
        print(f"{fold:>5} | {pass_pct:>6.1f} | "
              f"{100-pass_pct:>6.1f} | "
              f"{score:>11.4f}")

    print(f"\n✅ Stratified keeps ~same ratio every fold!")
    print(f"   ALWAYS use Stratified for classification!")


def cv_stability_comparison() -> None:
    """Compare model stability using CV std."""
    print("\n=== CV Stability Comparison ===\n")

    X, y = create_student_dataset()

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=42, max_iter=1000))
        ]),
        'Decision Tree (d=3)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', __import__(
                'sklearn.tree',
                fromlist=['DecisionTreeClassifier']
            ).DecisionTreeClassifier(
                max_depth=3, random_state=42))
        ]),
        'Decision Tree (d=None)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', __import__(
                'sklearn.tree',
                fromlist=['DecisionTreeClassifier']
            ).DecisionTreeClassifier(
                random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=100,
                random_state=42, n_jobs=-1))
        ])
    }

    cv = StratifiedKFold(
        n_splits=10, shuffle=True,
        random_state=42)

    print(f"{'Model':<28} | {'Mean':>7} | "
          f"{'Std':>7} | {'Min':>7} | "
          f"{'Max':>7} | {'Stable?'}")
    print("-" * 75)

    for name, model in models.items():
        scores = cross_val_score(
            model, X, y, cv=cv,
            scoring='f1', n_jobs=-1)
        stable = "✅ Yes" if scores.std() < 0.05 else "⚠️ No"
        print(f"{name:<28} | {scores.mean():>7.4f} | "
              f"{scores.std():>7.4f} | "
              f"{scores.min():>7.4f} | "
              f"{scores.max():>7.4f} | {stable}")

    print(f"\n💡 Low std = stable model")
    print(f"   High std = model is sensitive to data!")
    print(f"   Random Forest: usually most stable! 🔥")


def cross_validate_multiple_metrics() -> None:
    """Evaluate multiple metrics simultaneously."""
    print("\n=== Multi-metric Cross-Validation ===\n")

    X, y = create_student_dataset()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=42, n_jobs=-1))
    ])

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=['accuracy', 'f1',
                 'precision', 'recall',
                 'roc_auc'],
        n_jobs=-1)

    print("5-Fold CV Results (Random Forest):\n")
    metrics_map = {
        'test_accuracy': 'Accuracy',
        'test_f1': 'F1 Score',
        'test_precision': 'Precision',
        'test_recall': 'Recall',
        'test_roc_auc': 'ROC AUC'
    }

    for key, display in metrics_map.items():
        scores = results[key]
        print(f"  {display:<12}: "
              f"{scores.mean():.4f} ± "
              f"{scores.std():.4f}  "
              f"[{scores.min():.4f} - "
              f"{scores.max():.4f}]")

    print(f"\n  Fit time:  "
          f"{results['fit_time'].mean():.3f}s per fold")
    print(f"  Score time: "
          f"{results['score_time'].mean():.3f}s per fold")


if __name__ == "__main__":
    single_split_unreliability()
    kfold_vs_stratified()
    cv_stability_comparison()
    cross_validate_multiple_metrics()
