"""
Day 71 — Neural Networks from Scratch
Topic: NN vs Traditional ML Comparison
Date: 28 July 2026
Author: Bala Ravi

When does deep learning beat shallow ML?
Tabular data vs complex patterns.
"""
import numpy as np
from sklearn.datasets import (
    make_classification, make_moons,
    make_circles)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def compare_on_datasets() -> None:
    """Compare NN vs ML on different dataset types."""
    print("=== NN vs Traditional ML ===\n")
    print("When does Deep Learning win?\n")

    datasets = {
        'Linear (simple)': make_classification(
            n_samples=600, n_features=10,
            n_informative=5, n_redundant=2,
            random_state=42),
        'Non-linear Moons': make_moons(
            n_samples=600, noise=0.2,
            random_state=42),
        'Non-linear Circles': make_circles(
            n_samples=600, noise=0.1,
            factor=0.5, random_state=42),
        'Complex (15 features)': make_classification(
            n_samples=600, n_features=15,
            n_informative=10, n_redundant=3,
            n_clusters_per_class=2,
            random_state=42)
    }

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=42, max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=100,
                random_state=42, n_jobs=-1))
        ]),
        'Neural Network (2 layers)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=1000,
                random_state=42))
        ]),
        'Deep NN (4 layers)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32, 16),
                activation='relu',
                max_iter=1000,
                random_state=42))
        ])
    }

    print(f"{'Dataset':<25}", end='')
    for name in models:
        print(f"{name[:12]:>14}", end='')
    print()
    print("-" * 82)

    for ds_name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X, y, test_size=0.2,
                random_state=42))

        print(f"{ds_name:<25}", end='')
        scores = []
        for model in models.values():
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            scores.append(acc)
            print(f"{acc:>14.4f}", end='')
        print(f"  ← {'NN wins' if scores[-1] == max(scores) else 'RF wins'}")

    print(f"\n💡 Key takeaways:")
    print(f"   Tabular data (structured CSV):")
    print(f"   → Random Forest usually wins or ties")
    print(f"   → Less data needed, faster training")
    print(f"\n   Images / Text / Audio:")
    print(f"   → Deep Learning always wins")
    print(f"   → Learns features automatically!")
    print(f"\n   Rule: Try RF first on tabular data.")
    print(f"   Use NN when RF hits a ceiling!")


def data_size_effect() -> None:
    """Show how NN improves with more data."""
    print("\n=== Data Size Effect ===\n")
    print("NN needs more data than RF to shine!\n")

    print(f"{'N Samples':>10} | "
          f"{'Random Forest':>14} | "
          f"{'Neural Network':>15}")
    print("-" * 45)

    for n_samples in [100, 200, 500, 1000, 2000]:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=20,
            n_informative=12,
            random_state=42)

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X, y, test_size=0.2,
                random_state=42))

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        rf = RandomForestClassifier(
            n_estimators=100, random_state=42,
            n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_acc = rf.score(X_test, y_test)

        nn = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1000, random_state=42)
        nn.fit(X_tr, y_train)
        nn_acc = nn.score(X_te, y_test)

        winner = ("← NN" if nn_acc > rf_acc + 0.01
                  else "← RF" if rf_acc > nn_acc + 0.01
                  else "← tie")
        print(f"{n_samples:>10} | "
              f"{rf_acc:>14.4f} | "
              f"{nn_acc:>15.4f} {winner}")

    print(f"\n💡 Small data: RF wins!")
    print(f"   Large data: NN catches up or wins!")
    print(f"   With images: NN wins from day 1!")


if __name__ == "__main__":
    compare_on_datasets()
    data_size_effect()
