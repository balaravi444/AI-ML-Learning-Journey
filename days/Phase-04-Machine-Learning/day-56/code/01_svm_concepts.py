"""
Day 56 — SVM & KNN
Topic: SVM — Maximum Margin Classifier
Date: 13 July 2026
Author: Bala Ravi

SVM finds the hyperplane that MAXIMIZES
the margin between classes!

The key insight: points closest to boundary
(support vectors) define the margin.
Maximize that margin = best generalization!
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV)
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 800) -> pd.DataFrame:
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

    return pd.DataFrame({
        'study_hours': study.round(1),
        'attendance_pct': attend.round(1),
        'prev_score': prev.round(1),
        'assignments_done': assign,
        'sleep_hours': sleep.round(1),
        'passed': (score > 70).astype(int)
    })


def kernel_comparison(df: pd.DataFrame) -> None:
    """Compare different SVM kernels."""
    print("=== SVM Kernel Comparison ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    kernels = {
        'Linear': SVC(kernel='linear',
                      C=1.0, random_state=42),
        'RBF (default)': SVC(kernel='rbf',
                              C=1.0, gamma='scale',
                              random_state=42),
        'Polynomial (d=3)': SVC(kernel='poly',
                                 degree=3, C=1.0,
                                 random_state=42),
        'Sigmoid': SVC(kernel='sigmoid',
                        C=1.0, random_state=42)
    }

    print(f"{'Kernel':<22} | {'Train Acc':>10} | "
          f"{'Test Acc':>9} | {'F1':>7} | "
          f"{'CV':>7}")
    print("-" * 65)

    for name, model in kernels.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)

        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        f1 = f1_score(y_test,
                       pipeline.predict(X_test))
        cv = cross_val_score(
            pipeline, X, y, cv=5,
            scoring='accuracy',
            n_jobs=-1).mean()

        print(f"{name:<22} | {train_acc:>10.4f} | "
              f"{test_acc:>9.4f} | {f1:>7.4f} | "
              f"{cv:>7.4f}")

    print(f"\n💡 RBF kernel usually performs best!")
    print(f"   It maps to infinite dimensions!")
    print(f"   Most flexible boundary shape! 🔥")


def c_parameter_study(
        df: pd.DataFrame) -> None:
    """Show effect of C on SVM."""
    print("\n=== C Parameter Study ===\n")
    print("High C → strict margin (may overfit)")
    print("Low C  → soft margin (may underfit)\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"{'C Value':>10} | {'Train Acc':>10} | "
          f"{'Test Acc':>9} | {'Gap':>7} | {'Notes'}")
    print("-" * 65)

    c_values = [0.001, 0.01, 0.1, 1, 10,
                100, 1000]
    notes = ['← too small (underfit)',
             '← small', '', '← default',
             '', '← large',
             '← too large (overfit)']

    for C, note in zip(c_values, notes):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', C=C,
                          gamma='scale',
                          random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        gap = train_acc - test_acc

        flag = "⚠️" if gap > 0.1 else "✅"
        print(f"{C:>10} | {train_acc:>10.4f} | "
              f"{test_acc:>9.4f} | {gap:>7.4f} | "
              f"{flag} {note}")


def svm_grid_search(
        df: pd.DataFrame) -> None:
    """GridSearchCV for SVM hyperparameters."""
    print("\n=== SVM GridSearchCV ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(random_state=42))
    ])

    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['rbf', 'linear'],
        'model__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=5, scoring='f1',
        n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best parameters:")
    for param, val in grid.best_params_.items():
        print(f"  {param.replace('model__', '')}: {val}")

    print(f"\nBest CV F1:  {grid.best_score_:.4f}")

    y_pred = grid.best_estimator_.predict(X_test)
    print(f"Test F1:     {f1_score(y_test, y_pred):.4f}")
    print(f"Test Acc:    "
          f"{accuracy_score(y_test, y_pred):.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Fail', 'Pass']))


if __name__ == "__main__":
    df = create_student_dataset()
    print(f"Dataset: {df.shape}\n")

    kernel_comparison(df)
    c_parameter_study(df)
    svm_grid_search(df)
