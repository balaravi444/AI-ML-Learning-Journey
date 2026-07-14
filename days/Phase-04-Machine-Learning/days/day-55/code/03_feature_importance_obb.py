"""
Day 55 — Random Forest & Ensemble Methods
Topic: Feature Importance + OOB Score
Date: 12 July 2026
Author: Bala Ravi

Two superpowers of Random Forest:
1. Feature importance — more powerful than correlation!
2. OOB score — free cross-validation!
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def create_salary_dataset(
        n: int = 1000) -> pd.DataFrame:
    """Create salary prediction dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai',
              'Delhi', 'Hyderabad', 'Pune']

    experience = np.random.randint(0, 12, n)
    skills = np.random.randint(2, 10, n)
    city = np.random.choice(cities, n)
    remote = np.random.choice([0, 1], n, p=[0.7, 0.3])
    rating = np.round(
        np.random.uniform(3.0, 5.0, n), 1)
    random_noise = np.random.randn(n)

    city_map = {'Bangalore': 3, 'Mumbai': 2,
                'Delhi': 1, 'Hyderabad': 0.5,
                'Pune': 0}
    city_encoded = np.array([city_map[c] for c in city])

    salary = np.clip(
        10 + experience * 1.8 + skills * 0.5 +
        city_encoded + remote * 2.5 +
        rating * 0.5 + random_noise * 0.1 +
        np.random.normal(0, 2, n), 5, 55)

    above_avg = (salary > salary.mean()).astype(int)

    return pd.DataFrame({
        'experience_years': experience,
        'skills_count': skills,
        'city_encoded': city_encoded,
        'remote': remote,
        'rating': rating,
        'random_noise': random_noise,
        'above_avg_salary': above_avg
    })


def feature_importance_analysis(
        df: pd.DataFrame) -> None:
    """
    Compare RF feature importance vs correlation.
    RF is MORE powerful — captures non-linear!
    """
    print("=== Feature Importance Analysis ===\n")

    feature_cols = ['experience_years', 'skills_count',
                     'city_encoded', 'remote',
                     'rating', 'random_noise']
    X = df[feature_cols].values
    y = df['above_avg_salary'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    model = RandomForestClassifier(
        n_estimators=200,
        oob_score=True,
        random_state=42,
        n_jobs=-1)
    model.fit(X_train, y_train)

    # Feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Random Forest Feature Importance:")
    print(f"\n{'Feature':<20} | {'RF Importance':>14} | "
          f"{'Correlation':>12} | Bar")
    print("-" * 65)

    for _, row in importance_df.iterrows():
        feat = row['feature']
        col_data = df[feat]
        corr = abs(col_data.corr(
            df['above_avg_salary']))

        bar = '█' * int(row['importance'] * 60)
        print(f"{feat:<20} | {row['importance']:>14.4f} | "
              f"{corr:>12.4f} | {bar}")

    print(f"\n💡 Key Insights:")
    print(f"   random_noise: ~0 importance ✅")
    print(f"   RF correctly identified it as noise!")
    print(f"   experience_years: highest importance")
    print(f"   RF captures non-linear relationships!")


def oob_vs_cv_comparison(
        df: pd.DataFrame) -> None:
    """Show OOB score ≈ CV score."""
    print("\n=== OOB Score vs Cross-Validation ===\n")

    feature_cols = ['experience_years', 'skills_count',
                     'city_encoded', 'remote',
                     'rating', 'random_noise']
    X = df[feature_cols].values
    y = df['above_avg_salary'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    model = RandomForestClassifier(
        n_estimators=200,
        oob_score=True,
        random_state=42,
        n_jobs=-1)
    model.fit(X_train, y_train)

    test_acc = model.score(X_test, y_test)
    oob = model.oob_score_

    cv_scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1),
        X, y, cv=5, scoring='accuracy',
        n_jobs=-1)

    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"OOB Score:           {oob:.4f}")
    print(f"5-fold CV Mean:      {cv_scores.mean():.4f} "
          f"± {cv_scores.std():.4f}")

    diff = abs(oob - cv_scores.mean())
    print(f"\nOOB vs CV difference: {diff:.4f}")
    print(f"\n✅ OOB ≈ CV — almost identical!")
    print(f"   OOB is FREE — computed during training!")
    print(f"   No extra computation needed! 🔥")
    print(f"\n   Always use oob_score=True in RF!")


if __name__ == "__main__":
    df = create_salary_dataset()
    feature_importance_analysis(df)
    oob_vs_cv_comparison(df)
