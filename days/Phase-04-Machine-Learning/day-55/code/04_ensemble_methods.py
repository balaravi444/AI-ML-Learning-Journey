"""
Day 55 — Random Forest & Ensemble Methods
Topic: Ensemble Methods — Voting + Stacking
Date: 12 July 2026
Author: Bala Ravi

Beyond Random Forest:
Voting Classifier — combine different algorithms!
Stacking — train a meta-model on predictions!

These techniques win Kaggle competitions! 🔥
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_dataset(n: int = 800) -> tuple:
    """Create classification dataset."""
    np.random.seed(42)
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        random_state=42)

    return train_test_split(
        X, y, test_size=0.2, random_state=42)


def voting_classifier_demo() -> None:
    """
    Voting Classifier — combine different algorithms!
    Hard voting: majority vote
    Soft voting: average probabilities (better!)
    """
    print("=== Voting Classifier ===\n")

    X_train, X_test, y_train, y_test = create_dataset()

    # Individual models
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    dt = DecisionTreeClassifier(
        max_depth=5, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        n_jobs=-1)

    # Voting classifiers
    hard_voter = VotingClassifier(
        estimators=[('lr', lr), ('dt', dt),
                    ('rf', rf)],
        voting='hard')

    soft_voter = VotingClassifier(
        estimators=[('lr', lr), ('dt', dt),
                    ('rf', rf)],
        voting='soft')  # uses probabilities!

    models = {
        'Logistic Regression': lr,
        'Decision Tree': dt,
        'Random Forest': rf,
        'Hard Voting': hard_voter,
        'Soft Voting': soft_voter
    }

    print(f"{'Model':<25} | {'Test Acc':>9} | "
          f"{'F1':>7} | {'CV':>7}")
    print("-" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv = cross_val_score(
            model,
            np.vstack([X_train, X_test]),
            np.concatenate([y_train, y_test]),
            cv=5, scoring='accuracy',
            n_jobs=-1).mean()

        print(f"{name:<25} | {acc:>9.4f} | "
              f"{f1:>7.4f} | {cv:>7.4f}")

    print(f"\n💡 Soft Voting usually beats Hard Voting!")
    print(f"   Uses confidence scores not just votes!")


def bagging_vs_boosting() -> None:
    """Compare Bagging vs Boosting strategies."""
    print("\n=== Bagging vs Boosting ===\n")

    X_train, X_test, y_train, y_test = create_dataset()

    models = {
        'Single Decision Tree': (
            DecisionTreeClassifier(
                random_state=42)),
        'Random Forest (Bagging)': (
            RandomForestClassifier(
                n_estimators=100,
                random_state=42, n_jobs=-1)),
        'AdaBoost (Boosting)': (
            AdaBoostClassifier(
                n_estimators=100,
                random_state=42,
                algorithm='SAMME')),
        'Gradient Boosting': (
            GradientBoostingClassifier(
                n_estimators=100,
                random_state=42))
    }

    print(f"{'Strategy':<30} | {'Train':>7} | "
          f"{'Test':>7} | {'Gap':>7} | {'Note'}")
    print("-" * 70)

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc

        if 'Single' in name:
            note = "← baseline"
        elif 'Bagging' in name:
            note = "← reduces variance"
        elif 'Ada' in name:
            note = "← reduces bias"
        else:
            note = "← often highest acc"

        print(f"{name:<30} | {train_acc:>7.4f} | "
              f"{test_acc:>7.4f} | {gap:>7.4f} | "
              f"{note}")

    print(f"\n💡 Bagging (RF): reduces VARIANCE → "
          f"hard to overfit")
    print(f"   Boosting (GB): reduces BIAS → "
          f"can overfit if not tuned")
    print(f"   Both beat single tree! 🔥")


def stacking_demo() -> None:
    """
    Stacking — train a meta-model on predictions!
    Level 1: base models make predictions
    Level 2: meta-model learns from those predictions
    """
    print("\n=== Stacking Classifier ===\n")

    X_train, X_test, y_train, y_test = create_dataset()

    # Level 1 — base models
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=50, random_state=42,
            n_jobs=-1)),
        ('gb', GradientBoostingClassifier(
            n_estimators=50, random_state=42)),
        ('dt', DecisionTreeClassifier(
            max_depth=5, random_state=42))
    ]

    # Level 2 — meta-model
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5)

    # Compare
    rf_simple = RandomForestClassifier(
        n_estimators=100, random_state=42,
        n_jobs=-1)

    for name, model in [
        ('Random Forest', rf_simple),
        ('Stacking Ensemble', stacking)
    ]:
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        print(f"{name:<25}: {acc:.4f}")

    print(f"\n💡 Stacking often wins Kaggle competitions!")
    print(f"   But slower and harder to tune than RF!")
    print(f"   For production: start with RF,")
    print(f"   use stacking when you need that extra 1%!")


if __name__ == "__main__":
    voting_classifier_demo()
    bagging_vs_boosting()
    stacking_demo()
