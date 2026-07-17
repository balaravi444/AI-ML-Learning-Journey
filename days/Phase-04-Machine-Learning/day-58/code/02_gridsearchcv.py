"""
Day 58 — Cross Validation + Hyperparameter Tuning
Topic: GridSearchCV — Exhaustive Hyperparameter Search
Date: 15 July 2026
Author: Bala Ravi

GridSearchCV tries EVERY combination!
Finds the best hyperparameters systematically.
Use when param grid is small (<100 combinations).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import time


def create_student_dataset(
        n: int = 800) -> tuple:
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

    return train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y)


def gridsearch_random_forest() -> None:
    """GridSearchCV for Random Forest."""
    print("=== GridSearchCV — Random Forest ===\n")

    X_train, X_test, y_train, y_test = (
        create_student_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_leaf': [1, 5, 10],
        'model__max_features': ['sqrt', 'log2']
    }

    n_combos = (len(param_grid['model__n_estimators']) *
                len(param_grid['model__max_depth']) *
                len(param_grid['model__min_samples_leaf']) *
                len(param_grid['model__max_features']))
    cv_folds = 5

    print(f"Parameter grid:")
    for param, values in param_grid.items():
        clean = param.replace('model__', '')
        print(f"  {clean}: {values}")

    print(f"\nTotal combinations: {n_combos}")
    print(f"With {cv_folds}-fold CV: "
          f"{n_combos * cv_folds} model fits")
    print(f"\nRunning GridSearchCV...")

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42)

    start = time.time()
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        refit=True,
        return_train_score=True)

    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s\n")

    print(f"Best Parameters:")
    for param, value in grid.best_params_.items():
        clean = param.replace('model__', '')
        print(f"  {clean}: {value}")

    print(f"\nBest CV F1:  {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    print(f"Test F1:     {test_f1:.4f}")

    # Top 5 parameter combinations
    results_df = pd.DataFrame(
        grid.cv_results_)
    top5 = results_df.nlargest(
        5, 'mean_test_score')[
        ['params', 'mean_test_score',
         'std_test_score',
         'mean_train_score']]

    print(f"\nTop 5 Parameter Combinations:")
    print(f"{'#':>2} | {'CV F1':>7} | "
          f"{'Std':>6} | {'Train F1':>9} | Params")
    print("-" * 75)

    for i, (_, row) in enumerate(
            top5.iterrows(), 1):
        params_str = str({
            k.replace('model__', ''): v
            for k, v in row['params'].items()})
        print(f"{i:>2} | "
              f"{row['mean_test_score']:>7.4f} | "
              f"{row['std_test_score']:>6.4f} | "
              f"{row['mean_train_score']:>9.4f} | "
              f"{params_str[:45]}...")


def gridsearch_svm() -> None:
    """GridSearchCV for SVM."""
    print("\n=== GridSearchCV — SVM ===\n")

    X_train, X_test, y_train, y_test = (
        create_student_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(
            random_state=42,
            probability=True))
    ])

    param_grid = {
        'model__C': [0.1, 1, 10, 100],
        'model__kernel': ['rbf', 'linear'],
        'model__gamma': ['scale', 'auto', 0.01, 0.1]
    }

    n_combos = (len(param_grid['model__C']) *
                len(param_grid['model__kernel']) *
                len(param_grid['model__gamma']))

    print(f"Total combinations: {n_combos}")
    print(f"Running GridSearchCV...")

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring='f1',
        n_jobs=-1)

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s\n")
    print(f"Best Parameters:")
    for param, value in grid.best_params_.items():
        clean = param.replace('model__', '')
        print(f"  {clean}: {value}")

    print(f"\nBest CV F1:  {grid.best_score_:.4f}")
    test_f1 = f1_score(
        y_test, grid.predict(X_test))
    print(f"Test F1:     {test_f1:.4f}")


def analyze_param_importance() -> None:
    """Analyze which params matter most."""
    print("\n=== Parameter Importance Analysis ===\n")

    X_train, X_test, y_train, y_test = (
        create_student_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_leaf': [1, 5, 10, 20]
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    grid = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring='f1',
        n_jobs=-1)

    grid.fit(X_train, y_train)
    results = pd.DataFrame(grid.cv_results_)

    print("Effect of each hyperparameter on F1:\n")

    for param in ['model__n_estimators',
                   'model__max_depth',
                   'model__min_samples_leaf']:
        clean = param.replace('model__', '')
        print(f"  {clean}:")

        param_df = results.groupby(
            f'param_{param}'
        )['mean_test_score'].mean()

        for val, score in param_df.items():
            bar = '█' * int(score * 30)
            print(f"    {str(val):>8}: "
                  f"{score:.4f} {bar}")
        print()

    print(f"Best: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.4f}")


if __name__ == "__main__":
    gridsearch_random_forest()
    gridsearch_svm()
    analyze_param_importance()
