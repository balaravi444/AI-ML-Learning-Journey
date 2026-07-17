"""
Day 58 — Cross Validation + Hyperparameter Tuning
Topic: RandomizedSearchCV — Efficient Search
Date: 15 July 2026
Author: Bala Ravi

RandomizedSearchCV samples RANDOMLY from param space.
Often finds equally good results much faster!
Use when param grid is large (100+ combinations).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
    train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')
import time


def create_dataset(n: int = 800) -> tuple:
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


def grid_vs_random_search() -> None:
    """Compare GridSearch vs RandomizedSearch."""
    print("=== Grid vs Randomized Search ===\n")

    X_train, X_test, y_train, y_test = (
        create_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            random_state=42, n_jobs=-1))
    ])

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    # Grid search — smaller grid
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [5, 10, None],
        'model__min_samples_leaf': [1, 5, 10]
    }

    grid_combos = (
        len(param_grid['model__n_estimators']) *
        len(param_grid['model__max_depth']) *
        len(param_grid['model__min_samples_leaf']))

    print(f"Grid Search: {grid_combos} combinations × 5 folds "
          f"= {grid_combos*5} fits")

    start = time.time()
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=cv, scoring='f1',
        n_jobs=-1)
    grid.fit(X_train, y_train)
    grid_time = time.time() - start
    grid_f1 = f1_score(
        y_test, grid.predict(X_test))

    # Randomized search — larger space, fewer trials
    param_dist = {
        'model__n_estimators': randint(50, 500),
        'model__max_depth': randint(3, 20),
        'model__min_samples_leaf': randint(1, 20),
        'model__max_features': uniform(0.3, 0.6),
        'model__min_samples_split': randint(2, 20)
    }

    n_iter = 30
    print(f"Randomized Search: {n_iter} random trials × 5 folds "
          f"= {n_iter*5} fits\n")

    start = time.time()
    rand_search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=n_iter,
        cv=cv, scoring='f1',
        n_jobs=-1,
        random_state=42)
    rand_search.fit(X_train, y_train)
    rand_time = time.time() - start
    rand_f1 = f1_score(
        y_test, rand_search.predict(X_test))

    print(f"{'Method':<22} | {'CV F1':>7} | "
          f"{'Test F1':>8} | {'Time':>8}")
    print("-" * 50)
    print(f"{'GridSearchCV':<22} | "
          f"{grid.best_score_:>7.4f} | "
          f"{grid_f1:>8.4f} | "
          f"{grid_time:>7.1f}s")
    print(f"{'RandomizedSearchCV':<22} | "
          f"{rand_search.best_score_:>7.4f} | "
          f"{rand_f1:>8.4f} | "
          f"{rand_time:>7.1f}s")

    print(f"\n💡 Key Insight:")
    print(f"   Randomized found ~same quality params")
    print(f"   in fraction of the time!")
    print(f"   For large param spaces → always use Random! 🔥")

    print(f"\nBest params (Randomized):")
    for param, value in rand_search.best_params_.items():
        clean = param.replace('model__', '')
        val_str = (f"{value:.3f}"
                   if isinstance(value, float)
                   else str(value))
        print(f"  {clean}: {val_str}")


def n_iter_effect() -> None:
    """Show how n_iter affects quality vs speed."""
    print("\n=== n_iter Effect on Quality ===\n")

    X_train, X_test, y_train, y_test = (
        create_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            random_state=42, n_jobs=-1))
    ])

    param_dist = {
        'model__n_estimators': randint(50, 300),
        'model__max_depth': randint(3, 20),
        'model__min_samples_leaf': randint(1, 15),
        'model__max_features': uniform(0.3, 0.5)
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    print(f"{'n_iter':>7} | {'CV F1':>7} | "
          f"{'Test F1':>8} | {'Time':>7} | "
          f"{'Notes'}")
    print("-" * 55)

    prev_f1 = 0
    for n_iter in [5, 10, 20, 50, 100]:
        start = time.time()
        search = RandomizedSearchCV(
            pipeline, param_dist,
            n_iter=n_iter, cv=cv,
            scoring='f1', n_jobs=-1,
            random_state=42)
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        test_f1 = f1_score(
            y_test, search.predict(X_test))
        improvement = test_f1 - prev_f1
        note = (f"+{improvement:.4f}"
                if improvement > 0.001
                else "~same")
        prev_f1 = test_f1

        print(f"{n_iter:>7} | "
              f"{search.best_score_:>7.4f} | "
              f"{test_f1:>8.4f} | "
              f"{elapsed:>6.1f}s | {note}")

    print(f"\n💡 Diminishing returns after n_iter=30-50!")
    print(f"   n_iter=50 usually sweet spot!")


def scipy_distributions_guide() -> None:
    """Show how to use scipy distributions."""
    print("\n=== Scipy Distributions for RandomSearch ===\n")

    print("Integer parameters → scipy.stats.randint")
    dist = randint(50, 500)
    samples = dist.rvs(size=10, random_state=42)
    print(f"  randint(50, 500) samples: {sorted(samples)}")

    print("\nContinuous parameters → scipy.stats.uniform")
    dist = uniform(0.1, 0.9)
    samples = dist.rvs(size=10, random_state=42)
    print(f"  uniform(0.1, 1.0) samples: "
          f"{sorted(samples.round(3))}")

    print("\nLog-scale parameters → scipy.stats.loguniform")
    from scipy.stats import loguniform
    dist = loguniform(1e-4, 1e2)
    samples = dist.rvs(size=10, random_state=42)
    print(f"  loguniform(1e-4, 1e2) samples: "
          f"{sorted(samples.round(4))}")
    print(f"  → Great for learning_rate, C in SVM!")

    print("\nCommon RandomSearch param_dist template:")
    template = {
        'RF': {
            'n_estimators': 'randint(50, 500)',
            'max_depth': 'randint(3, 20)',
            'min_samples_leaf': 'randint(1, 20)',
            'max_features': 'uniform(0.3, 0.5)'
        },
        'SVM': {
            'C': 'loguniform(1e-3, 1e3)',
            'gamma': 'loguniform(1e-4, 1e1)',
            'kernel': "['rbf', 'linear']"
        },
        'LogReg': {
            'C': 'loguniform(1e-3, 1e3)',
            'max_iter': 'randint(100, 1000)'
        }
    }

    for model, params in template.items():
        print(f"\n  {model}:")
        for param, dist_str in params.items():
            print(f"    '{param}': {dist_str}")


if __name__ == "__main__":
    grid_vs_random_search()
    n_iter_effect()
    scipy_distributions_guide()
