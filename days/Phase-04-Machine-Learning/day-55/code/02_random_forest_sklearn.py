"""
Day 55 — Random Forest & Ensemble Methods
Topic: Sklearn Random Forest — Full Usage
Date: 12 July 2026
Author: Bala Ravi

Random Forest = Bagging + Random Feature Selection
The most important addition: sqrt(features) per split!
This decorrelates the trees — they make different errors!
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report)
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 1000) -> pd.DataFrame:
    """Create student performance dataset."""
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

    return pd.DataFrame({
        'study_hours': study.round(1),
        'attendance_pct': attend.round(1),
        'prev_score': prev.round(1),
        'assignments_done': assign,
        'sleep_hours': sleep.round(1),
        'passed': (score > 70).astype(int)
    })


def dt_vs_rf_comparison(
        df: pd.DataFrame) -> None:
    """Complete Decision Tree vs Random Forest comparison."""
    print("=== Decision Tree vs Random Forest ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    models = {
        'Decision Tree (no limit)': (
            DecisionTreeClassifier(random_state=42)),
        'Decision Tree (depth=5)': (
            DecisionTreeClassifier(
                max_depth=5, random_state=42)),
        'Random Forest (10 trees)': (
            RandomForestClassifier(
                n_estimators=10, oob_score=True,
                random_state=42, n_jobs=-1)),
        'Random Forest (100 trees)': (
            RandomForestClassifier(
                n_estimators=100, oob_score=True,
                random_state=42, n_jobs=-1)),
        'Random Forest (200 trees)': (
            RandomForestClassifier(
                n_estimators=200, oob_score=True,
                random_state=42, n_jobs=-1))
    }

    print(f"{'Model':<30} | {'Train':>7} | "
          f"{'Test':>7} | {'Gap':>6} | "
          f"{'OOB':>7} | {'CV':>7}")
    print("-" * 75)

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc

        oob = getattr(model, 'oob_score_', None)
        oob_str = f"{oob:.4f}" if oob else "N/A"

        cv_scores = cross_val_score(
            model, X, y, cv=5,
            scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()

        flag = "⚠️" if gap > 0.1 else "✅"
        print(f"{name:<30} | {train_acc:>7.4f} | "
              f"{test_acc:>7.4f} | {gap:>6.4f} | "
              f"{oob_str:>7} | {cv_mean:>7.4f} {flag}")

    print(f"\n💡 Key Observations:")
    print(f"   DT no limit: 100% train → OVERFIT badly!")
    print(f"   RF 10 trees: unstable (too few)")
    print(f"   RF 100 trees: stable + accurate ✅")
    print(f"   RF 200 trees: marginal improvement")
    print(f"\n   OOB ≈ CV score — free validation! 🔥")


def n_estimators_study(
        df: pd.DataFrame) -> None:
    """Show how accuracy stabilizes with more trees."""
    print("\n=== N Estimators Study ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"{'N Trees':>8} | {'Test Acc':>9} | "
          f"{'OOB Score':>10} | {'Change':>8}")
    print("-" * 45)

    prev_acc = 0
    for n in [5, 10, 20, 50, 100, 200, 500]:
        model = RandomForestClassifier(
            n_estimators=n,
            oob_score=True,
            random_state=42,
            n_jobs=-1)
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        oob = model.oob_score_
        change = test_acc - prev_acc

        change_str = (f"+{change:.4f}"
                      if change > 0 else
                      f"{change:.4f}")
        print(f"{n:>8} | {test_acc:>9.4f} | "
              f"{oob:>10.4f} | {change_str:>8}")
        prev_acc = test_acc

    print(f"\n💡 After 100 trees — diminishing returns!")
    print(f"   100 trees → almost same as 500 trees")
    print(f"   But 100 trees trains 5x faster! 🚀")


if __name__ == "__main__":
    df = create_student_dataset()
    dt_vs_rf_comparison(df)
    n_estimators_study(df)
