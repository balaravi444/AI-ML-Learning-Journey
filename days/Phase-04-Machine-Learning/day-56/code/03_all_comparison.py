"""
Day 56 — SVM & KNN
Topic: Complete Classifier Comparison
Date: 13 July 2026
Author: Bala Ravi

All classifiers from Phase 4 side by side!
Logistic Regression, Decision Tree,
Random Forest, SVM, KNN.

Which to use when?
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold)
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score,
    classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import time


def create_rich_dataset(
        n: int = 1000) -> pd.DataFrame:
    """Create rich student dataset."""
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


def compare_all_classifiers(
        df: pd.DataFrame) -> None:
    """Compare ALL classifiers from Phase 4."""
    print("=== All Phase 4 Classifiers Comparison ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    # All classifiers in pipelines
    classifiers = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                C=1.0, random_state=42,
                max_iter=1000))
        ]),
        'Decision Tree (d=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', DecisionTreeClassifier(
                max_depth=5, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=100,
                random_state=42, n_jobs=-1))
        ]),
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(
                kernel='rbf', C=10,
                gamma='scale',
                probability=True,
                random_state=42))
        ]),
        'KNN (k=7)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(
                n_neighbors=7))
        ])
    }

    print(f"{'Classifier':<25} | {'Acc':>7} | "
          f"{'F1':>7} | {'AUC':>7} | "
          f"{'CV F1':>7} | {'Time':>8}")
    print("-" * 75)

    results = {}
    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    for name, pipeline in classifiers.items():
        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(
            X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cv_f1 = cross_val_score(
            pipeline, X, y, cv=cv,
            scoring='f1',
            n_jobs=-1).mean()

        results[name] = {
            'acc': acc, 'f1': f1,
            'auc': auc, 'cv_f1': cv_f1}

        print(f"{name:<25} | {acc:>7.4f} | "
              f"{f1:>7.4f} | {auc:>7.4f} | "
              f"{cv_f1:>7.4f} | "
              f"{train_time:>7.3f}s")

    # Find winners
    best_acc = max(results,
                    key=lambda x: results[x]['acc'])
    best_f1 = max(results,
                   key=lambda x: results[x]['cv_f1'])

    print(f"\n🏆 Best Accuracy: {best_acc}")
    print(f"🏆 Best CV F1:    {best_f1}")


def when_to_use_which() -> None:
    """Print decision guide for choosing algorithms."""
    print("\n=== When to Use Which Classifier ===\n")

    guide = {
        'Logistic Regression': {
            'use_when': [
                'Need interpretable coefficients',
                'Binary or multi-class classification',
                'Quick baseline model',
                'Linearly separable data'
            ],
            'avoid_when': [
                'Complex non-linear relationships',
                'High dimensional sparse data (use SVM)'
            ]
        },
        'Decision Tree': {
            'use_when': [
                'Need human-readable rules',
                'Stakeholders need explanations',
                'Mixed feature types (no scaling needed)',
                'Quick visual interpretation'
            ],
            'avoid_when': [
                'Need high accuracy (use RF instead)',
                'Small datasets (overfits easily)'
            ]
        },
        'Random Forest': {
            'use_when': [
                'General purpose — almost always works!',
                'Feature importance needed',
                'Tabular data with mixed types',
                'Need robust predictions'
            ],
            'avoid_when': [
                'Very large datasets (slow)',
                'Need full interpretability'
            ]
        },
        'SVM': {
            'use_when': [
                'Small to medium dataset',
                'High-dimensional data (text!)',
                'Clear margin of separation',
                'Need maximum margin boundary'
            ],
            'avoid_when': [
                'Large datasets (training slow)',
                'Need probability estimates (tricky)'
            ]
        },
        'KNN': {
            'use_when': [
                'Small dataset',
                'Non-linear decision boundary',
                'Recommendation systems',
                'Anomaly detection'
            ],
            'avoid_when': [
                'Large datasets (prediction slow)',
                'High-dimensional data (curse of dim)'
            ]
        }
    }

    for model, info in guide.items():
        print(f"🔷 {model}")
        print(f"   USE WHEN:")
        for tip in info['use_when']:
            print(f"     ✅ {tip}")
        print(f"   AVOID WHEN:")
        for tip in info['avoid_when']:
            print(f"     ⚠️  {tip}")
        print()


def scaling_importance_demo() -> None:
    """Show why scaling matters for SVM and KNN."""
    print("=== Why Scaling Matters for SVM/KNN ===\n")

    np.random.seed(42)
    n = 300

    # Unscaled features — very different ranges!
    salary = np.random.normal(30000, 8000, n)
    age = np.random.normal(28, 4, n)
    skills = np.random.randint(2, 10, n)

    y = ((salary > 30000) &
         (age < 35)).astype(int)

    X = np.column_stack([salary, age, skills])

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"Feature ranges:")
    print(f"  salary: {salary.min():.0f} to "
          f"{salary.max():.0f} (range: "
          f"{salary.max()-salary.min():.0f})")
    print(f"  age:    {age.min():.0f} to "
          f"{age.max():.0f} (range: "
          f"{age.max()-age.min():.0f})")
    print(f"  skills: {skills.min()} to "
          f"{skills.max()} (range: "
          f"{skills.max()-skills.min()})\n")

    print(f"{'Model':<20} | {'No Scale':>9} | "
          f"{'Scaled':>9} | {'Difference':>11}")
    print("-" * 55)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    for name, model_class, kwargs in [
        ('SVM (RBF)', SVC, {'kernel': 'rbf',
                             'random_state': 42}),
        ('KNN (k=5)', KNeighborsClassifier,
         {'n_neighbors': 5})
    ]:
        # Without scaling
        m1 = model_class(**kwargs)
        m1.fit(X_train, y_train)
        acc_raw = m1.score(X_test, y_test)

        # With scaling
        m2 = model_class(**kwargs)
        m2.fit(X_train_s, y_train)
        acc_scaled = m2.score(X_test_s, y_test)

        diff = acc_scaled - acc_raw
        print(f"{name:<20} | {acc_raw:>9.4f} | "
              f"{acc_scaled:>9.4f} | "
              f"+{diff:>10.4f}")

    print(f"\n✅ Scaling dramatically improves")
    print(f"   SVM and KNN performance!")
    print(f"   salary (30000) was dominating")
    print(f"   age (28) and skills (5)!")
    print(f"\n   Random Forest doesn't care — trees")
    print(f"   use thresholds, not distances! 🔥")


if __name__ == "__main__":
    df = create_rich_dataset()
    print(f"Dataset: {df.shape}\n")

    compare_all_classifiers(df)
    when_to_use_which()
    scaling_importance_demo()
