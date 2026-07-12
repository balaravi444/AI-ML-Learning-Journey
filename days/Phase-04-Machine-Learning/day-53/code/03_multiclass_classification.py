"""
Day 53 — Logistic Regression
Topic: Multi-class Classification
Date: 10 July 2026
Author: Bala Ravi

OvR vs Softmax (Multinomial) strategies!
Predicting multiple categories, not just binary.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder)
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_job_level_dataset(
        n: int = 800) -> pd.DataFrame:
    """
    Create job level classification dataset.
    Predict: Junior, Mid, Senior, Expert
    """
    np.random.seed(42)

    experience = np.random.randint(0, 15, n)
    skills = np.random.randint(2, 10, n)
    rating = np.round(
        np.random.uniform(3.0, 5.0, n), 1)
    salary = np.clip(
        10 + experience * 2 + skills * 0.5 +
        np.random.normal(0, 3, n), 5, 55)

    # Job level based on experience
    level = np.where(
        experience < 2, 'Junior',
        np.where(experience < 5, 'Mid',
                 np.where(experience < 9,
                          'Senior', 'Expert')))

    return pd.DataFrame({
        'experience_years': experience,
        'skills_count': skills,
        'rating': rating,
        'salary_lpa': salary.round(1),
        'level': level
    })


def ovr_vs_softmax_demo(
        df: pd.DataFrame) -> None:
    """Compare OvR and Softmax strategies."""
    print("=== OvR vs Softmax Multi-class ===\n")

    le = LabelEncoder()
    X = df.drop('level', axis=1)
    y = le.fit_transform(df['level'])

    print(f"Classes: {le.classes_}")
    print(f"Dataset: {len(df)} samples\n")

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    strategies = {
        'OvR (One vs Rest)': LogisticRegression(
            multi_class='ovr',
            random_state=42, max_iter=1000),
        'Softmax (Multinomial)': LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42, max_iter=1000)
    }

    print(f"{'Strategy':<25} | {'Accuracy':>9} | "
          f"{'CV Mean':>8}")
    print("-" * 48)

    for name, model in strategies.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        cv = cross_val_score(
            pipe, X, y, cv=5,
            scoring='accuracy').mean()
        print(f"{name:<25} | {acc:>9.4f} | "
              f"{cv:>8.4f}")

    # Detailed report for softmax
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42, max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"\n📊 Softmax Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_))


def predict_job_level(
        df: pd.DataFrame) -> None:
    """Predict job levels for new candidates."""
    print("\n=== Predict Job Level ===\n")

    le = LabelEncoder()
    X = df.drop('level', axis=1)
    y = le.fit_transform(df['level'])

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42, max_iter=1000))
    ])
    pipe.fit(X, y)

    candidates = pd.DataFrame({
        'experience_years': [1, 3, 6, 10, 12],
        'skills_count': [3, 5, 7, 9, 10],
        'rating': [3.5, 4.0, 4.2, 4.5, 4.8],
        'salary_lpa': [10, 18, 26, 35, 45]
    })

    predictions = le.inverse_transform(
        pipe.predict(candidates))
    probabilities = pipe.predict_proba(candidates)

    print(f"{'Exp':>4} | {'Skills':>6} | "
          f"{'Predicted Level':>16} | "
          f"Confidence")
    print("-" * 65)

    for i, (_, row) in enumerate(
            candidates.iterrows()):
        pred = predictions[i]
        conf = probabilities[i].max()
        pred_idx = pipe.predict([candidates.iloc[i]])[0]
        print(f"{int(row['experience_years']):>4}yr | "
              f"{int(row['skills_count']):>6} | "
              f"{pred:>16} | "
              f"{conf:.1%}")


if __name__ == "__main__":
    df = create_job_level_dataset()
    print(f"Dataset: {df.shape}\n")
    print("Class distribution:")
    print(df['level'].value_counts())
    print()

    ovr_vs_softmax_demo(df)
    predict_job_level(df)
