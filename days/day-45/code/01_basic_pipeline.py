"""
Day 46 — Data Preprocessing Pipeline
Topic: Basic Sklearn Pipeline
Date: 03 July 2026
Author: Bala Ravi

Pipeline = chain preprocessing steps + model!
Prevents data leakage. Production-ready code!
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    r2_score, mean_absolute_error)


def create_dataset(n: int = 400) -> pd.DataFrame:
    """Create job dataset with missing values."""
    np.random.seed(42)

    experience = np.random.randint(0, 12, n).astype(float)
    skills = np.random.randint(2, 10, n).astype(float)
    rating = np.round(
        np.random.uniform(3.0, 5.0, n), 1)

    salary = np.clip(
        10 + experience * 1.8 + skills * 0.5 +
        np.random.normal(0, 2, n), 5, 55).round(1)

    # Add missing values (real world!)
    missing_idx = np.random.choice(
        n, int(n * 0.1), replace=False)
    experience[missing_idx[:n//20]] = np.nan
    skills[missing_idx[n//20:]] = np.nan

    return pd.DataFrame({
        'experience_years': experience,
        'skills_count': skills,
        'rating': rating,
        'salary_lpa': salary
    })


def without_pipeline_demo(df: pd.DataFrame) -> None:
    """Show problems WITHOUT using pipeline."""
    print("=== WITHOUT Pipeline (DANGEROUS!) ===\n")

    X = df.drop('salary_lpa', axis=1)
    y = df['salary_lpa']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Manual preprocessing — LEAKAGE RISK!
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    # Correctly done here — but easy to mess up!
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    model = RandomForestRegressor(
        n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: ₹{mean_absolute_error(y_test, y_pred):.2f} LPA")
    print("\n⚠️ Manual steps = easy to make mistakes!")
    print("⚠️ Cross-validation with manual steps = LEAKAGE!")


def with_pipeline_demo(df: pd.DataFrame) -> None:
    """Show benefits WITH pipeline."""
    print("\n=== WITH Pipeline (SAFE & CLEAN!) ===\n")

    X = df.drop('salary_lpa', axis=1)
    y = df['salary_lpa']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Pipeline chains steps — NO leakage possible!
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=50, random_state=42))
    ])

    # ONE call trains everything!
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: ₹{mae:.2f} LPA")

    # Cross-validation with pipeline — SAFE!
    cv_scores = cross_val_score(
        pipeline, X, y, cv=5, scoring='r2')
    print(f"\nCross-val R² scores: "
          f"{cv_scores.round(3)}")
    print(f"Mean: {cv_scores.mean():.4f} "
          f"± {cv_scores.std():.4f}")

    print("\n✅ Pipeline = clean, safe, production-ready!")
    print("✅ Cross-validation works correctly!")
    print("✅ No data leakage possible!")

    return pipeline


def pipeline_inspection(pipeline: Pipeline) -> None:
    """Inspect pipeline steps."""
    print("\n=== Pipeline Inspection ===\n")

    print("Pipeline steps:")
    for name, step in pipeline.steps:
        print(f"  {name}: {type(step).__name__}")

    # Access individual steps
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']

    print(f"\nImputer strategy: {imputer.strategy}")
    print(f"Scaler mean: "
          f"{scaler.mean_.round(2)}")
    print(f"Model n_estimators: "
          f"{model.n_estimators}")


if __name__ == "__main__":
    df = create_dataset()
    print(f"Dataset: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()}\n")

    without_pipeline_demo(df)
    pipeline = with_pipeline_demo(df)
    pipeline_inspection(pipeline)
