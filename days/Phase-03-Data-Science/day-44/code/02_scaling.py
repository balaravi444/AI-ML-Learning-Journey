"""
Day 44 — Feature Engineering
Topic: Feature Scaling
Date: 01 July 2026
Author: Bala Ravi

Different ML models need different scaling!
StandardScaler, MinMaxScaler, RobustScaler.

Why scale?
KNN uses distance — unscaled salary (₹25000)
dominates age (25) completely!
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler)


def create_dataset() -> pd.DataFrame:
    """Create dataset with different scale features."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        'age': np.random.randint(22, 45, n),
        'salary_lpa': np.clip(
            np.random.normal(20, 8, n), 5, 55),
        'experience': np.random.randint(0, 20, n),
        'skills_count': np.random.randint(2, 10, n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1)
    })


def compare_scalers(df: pd.DataFrame) -> None:
    """
    Compare all three scalers side by side.
    Shows when to use which scaler!
    """
    print("=== Feature Scaling Comparison ===\n")

    features = ['age', 'salary_lpa',
                'experience', 'skills_count']
    X = df[features].values

    print("Before scaling:")
    print(f"  salary range: "
          f"{df['salary_lpa'].min():.0f} to "
          f"{df['salary_lpa'].max():.0f}")
    print(f"  age range:    "
          f"{df['age'].min()} to "
          f"{df['age'].max()}")
    print(f"  → Different scales = "
          f"unfair ML model! ❌\n")

    scalers = {
        'StandardScaler\n(mean=0, std=1)': StandardScaler(),
        'MinMaxScaler\n(0 to 1)': MinMaxScaler(),
        'RobustScaler\n(median-based)': RobustScaler()
    }

    for name, scaler in scalers.items():
        X_scaled = scaler.fit_transform(X)
        df_scaled = pd.DataFrame(
            X_scaled, columns=features)

        print(f"{name}:")
        for feat in features:
            col = df_scaled[feat]
            print(f"  {feat:<15}: "
                  f"min={col.min():>6.2f}, "
                  f"max={col.max():>6.2f}, "
                  f"mean={col.mean():>6.2f}")
        print()

    print("📌 When to use which:")
    print("  StandardScaler → Linear models, SVM, NN")
    print("  MinMaxScaler   → Neural Networks, KNN")
    print("  RobustScaler   → When data has outliers")
    print("  No scaling     → Tree models "
          "(Random Forest, XGBoost)")


def train_test_scaling_demo(
        df: pd.DataFrame) -> None:
    """
    CRITICAL: Only fit scaler on training data!
    Fitting on test = data leakage!
    """
    print("\n=== Train/Test Scaling (Critical!) ===\n")

    from sklearn.model_selection import train_test_split

    features = ['age', 'salary_lpa',
                'experience', 'skills_count']
    X = df[features]
    y = df['rating']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")

    scaler = StandardScaler()

    print("\n✅ CORRECT approach:")
    print("  scaler.fit(X_train)           "
          "← fit ONLY on train!")
    print("  X_train = scaler.transform(X_train)")
    print("  X_test  = scaler.transform(X_test) "
          "← transform, not fit!")

    # Correct way
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  Train salary mean after scaling: "
          f"{X_train_scaled[:, 1].mean():.4f}")
    print(f"  Train salary std after scaling:  "
          f"{X_train_scaled[:, 1].std():.4f}")

    print("\n❌ WRONG approach (data leakage):")
    print("  scaler.fit_transform(X_train)")
    print("  scaler.fit_transform(X_test)  "
          "← NEVER! Uses test data to fit!")
    print("\n  This makes your model seem better")
    print("  than it really is in production! 🚨")


if __name__ == "__main__":
    df = create_dataset()
    compare_scalers(df)
    train_test_scaling_demo(df)
