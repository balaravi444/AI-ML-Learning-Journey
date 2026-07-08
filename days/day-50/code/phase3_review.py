"""
Day 50 — Phase 3 Complete Review
Topic: Everything learned in Phase 3
Date: 07 July 2026
Author: Bala Ravi

This file reviews ALL Phase 3 concepts
in one runnable script!
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import r2_score


def numpy_review() -> None:
    """Day 36-37 — NumPy review."""
    print("=== NumPy (Day 36-37) ===\n")

    # Arrays
    arr = np.array([1, 2, 3, 4, 5])
    matrix = np.random.randn(3, 4)

    print(f"Array: {arr}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix mean per col: "
          f"{matrix.mean(axis=0).round(2)}")

    # Linear algebra
    A = np.random.randn(3, 3)
    eigenvalues, _ = np.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues.round(2)}")

    # Vectorization speed
    n = 1_000_000
    python_list = list(range(n))
    numpy_arr = np.arange(n)

    import time
    start = time.time()
    _ = [x * 2 for x in python_list]
    py_time = time.time() - start

    start = time.time()
    _ = numpy_arr * 2
    np_time = time.time() - start

    print(f"NumPy speedup: {py_time/np_time:.0f}x faster!")


def pandas_review() -> None:
    """Day 38-40 — Pandas review."""
    print("\n=== Pandas (Day 38-40) ===\n")

    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'city': np.random.choice(
            ['Bangalore', 'Mumbai', 'Delhi'], n),
        'salary': np.random.normal(20, 5, n).round(1),
        'experience': np.random.randint(0, 10, n),
        'skills': np.random.randint(2, 10, n)
    })

    # Add missing values
    df.loc[np.random.choice(n, 10), 'salary'] = np.nan

    print(f"Shape: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()}")

    # Clean
    df['salary'].fillna(
        df['salary'].median(), inplace=True)
    print(f"After cleaning: "
          f"{df.isnull().sum().sum()} missing")

    # GroupBy
    city_avg = (df.groupby('city')['salary']
                .mean().round(1))
    print(f"\nAvg salary by city:")
    for city, sal in city_avg.items():
        print(f"  {city}: ₹{sal}L")

    # Transform
    df['city_avg'] = (df.groupby('city')['salary']
                      .transform('mean').round(1))
    print(f"\nTransform added city_avg column ✅")


def statistics_review() -> None:
    """Day 45 — Statistics review."""
    print("\n=== Statistics (Day 45) ===\n")

    np.random.seed(42)

    # T-test
    remote = np.random.normal(24, 4, 50)
    onsite = np.random.normal(21, 4, 50)

    t, p = stats.ttest_ind(remote, onsite)
    print(f"Remote vs Onsite T-test:")
    print(f"  t={t:.2f}, p={p:.4f}")
    print(f"  Significant: {p < 0.05} ✅")

    # Correlation
    exp = np.random.randint(0, 12, 100)
    sal = 10 + exp * 1.8 + np.random.normal(0, 2, 100)
    corr, p_corr = stats.pearsonr(exp, sal)
    print(f"\nCorrelation (exp vs salary):")
    print(f"  r={corr:.3f}, p={p_corr:.6f}")

    # Normal test
    stat, p_norm = stats.normaltest(sal)
    print(f"\nNormality test:")
    print(f"  Normal: {p_norm > 0.05}")


def pipeline_review() -> None:
    """Day 46 — Pipeline review."""
    print("\n=== Pipeline (Day 46) ===\n")

    np.random.seed(42)
    n = 300

    exp = np.random.randint(0, 12, n).astype(float)
    skills = np.random.randint(2, 10, n).astype(float)
    sal = np.clip(
        10 + exp * 1.8 + skills * 0.5 +
        np.random.normal(0, 2, n), 5, 55)

    # Add missing
    exp[np.random.choice(n, 15)] = np.nan

    X = np.column_stack([exp, skills])
    y = sal

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    pipeline = Pipeline([
        ('imputer', SimpleImputer(
            strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=50, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    r2 = r2_score(y_test,
                   pipeline.predict(X_test))

    cv = cross_val_score(
        pipeline, X, y, cv=5,
        scoring='r2').mean()

    print(f"Pipeline: 3 steps")
    print(f"Test R²: {r2:.4f}")
    print(f"CV R²:   {cv:.4f}")
    print(f"No data leakage! ✅")


def phase3_summary() -> None:
    """Print Phase 3 complete summary."""
    print("\n" + "=" * 55)
    print("  🏆 PHASE 3 COMPLETE SUMMARY")
    print("=" * 55)

    topics = {
        "Day 36-37": "NumPy + Linear Algebra",
        "Day 38-39": "Pandas + Data Cleaning",
        "Day 40":    "GroupBy & Aggregations",
        "Day 41-42": "Matplotlib + Seaborn",
        "Day 43":    "EDA Complete Workflow",
        "Day 44":    "Feature Engineering",
        "Day 45":    "Statistics for ML",
        "Day 46":    "Preprocessing Pipeline",
        "Day 47-50": "Indian Job Market Analyzer"
    }

    for days, topic in topics.items():
        print(f"  {days:<12} → {topic}")

    stats_summary = {
        "Days completed": "15 (Day 36-50)",
        "Libraries mastered": "NumPy, Pandas, "
                              "Matplotlib, Seaborn, "
                              "Scipy, Sklearn",
        "Projects built": "1 (Indian Job Market Analyzer)",
        "Live deployments": "1 (Render)",
        "ML model R²": "0.89",
        "Charts created": "9",
        "API endpoints": "11",
        "Days missed": "0 🔥"
    }

    print()
    for key, val in stats_summary.items():
        print(f"  {key:<22} → {val}")

    print("\n  Phase 4 starts tomorrow!")
    print("  Machine Learning — Real algorithms!")
    print("=" * 55)


if __name__ == "__main__":
    print("🎓 Phase 3 Complete Review\n")

    numpy_review()
    pandas_review()
    statistics_review()
    pipeline_review()
    phase3_summary()
