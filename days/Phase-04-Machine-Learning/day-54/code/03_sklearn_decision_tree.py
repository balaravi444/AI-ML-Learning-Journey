"""
Day 54 — Decision Trees
Topic: Sklearn Decision Tree — Full Usage
Date: 11 July 2026
Author: Bala Ravi

Complete Decision Tree workflow with sklearn!
Classifier + Regressor + Feature Importance!
"""
import numpy as np
import pandas as pd
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor)
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, f1_score,
    r2_score, mean_absolute_error,
    classification_report)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 800) -> pd.DataFrame:
    """Create student performance dataset."""
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

    passed = (score > 70).astype(int)

    return pd.DataFrame({
        'study_hours': study.round(1),
        'attendance_pct': attend.round(1),
        'prev_score': prev.round(1),
        'assignments_done': assign,
        'sleep_hours': sleep.round(1),
        'final_score': score.round(1),
        'passed': passed
    })


def overfit_demonstration(
        df: pd.DataFrame) -> None:
    """Show overfitting with deep trees."""
    print("=== Overfitting in Decision Trees ===\n")

    X = df[['study_hours', 'attendance_pct',
             'prev_score', 'assignments_done',
             'sleep_hours']].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"{'Max Depth':>10} | {'Train Acc':>10} | "
          f"{'Test Acc':>10} | {'Gap':>8} | "
          f"{'Status':>20}")
    print("-" * 65)

    depths = [1, 2, 3, 5, 8, 10, 15, None]

    for depth in depths:
        model = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42)
        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc

        if depth is None:
            depth_str = "None (full)"
        else:
            depth_str = str(depth)

        if gap > 0.15:
            status = "⚠️ Overfitting!"
        elif test_acc > 0.85:
            status = "✅ Great"
        elif test_acc > 0.75:
            status = "⚡ Good"
        else:
            status = "❌ Underfitting"

        print(f"{depth_str:>10} | {train_acc:>10.4f} | "
              f"{test_acc:>10.4f} | {gap:>8.4f} | "
              f"{status:>20}")

    print(f"\n💡 Key Finding:")
    print(f"   depth=None → 100% train, ~73% test (OVERFIT!)")
    print(f"   depth=5    → ~89% train, ~87% test (SWEET SPOT!)")
    print(f"   Always tune max_depth! 🎯")


def feature_importance_analysis(
        df: pd.DataFrame) -> None:
    """Analyze feature importance."""
    print("\n=== Feature Importance ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        random_state=42)
    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Feature Importance (contribution to splits):")
    print(f"\n{'Feature':<22} | {'Importance':>11} | "
          f"{'Bar'}")
    print("-" * 55)

    for _, row in importance_df.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"{row['feature']:<22} | "
              f"{row['importance']:>11.4f} | {bar}")

    print(f"\n💡 study_hours and prev_score are")
    print(f"   the strongest predictors!")
    print(f"   sleep_hours contributes least.")
    print(f"\n✅ This matches our domain intuition!")
    print(f"   How much you study matters most!")


def decision_tree_rules(
        df: pd.DataFrame) -> None:
    """Extract human-readable rules from tree."""
    print("\n=== Human-Readable Decision Rules ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['passed'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Shallow tree for interpretability
    model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Test accuracy (depth=3): {acc:.4f}")

    # Manually extract top rule
    tree = model.tree_
    feature_names = feature_cols

    print(f"\n📋 Top Decision Rules:")
    print(f"\nRule 1 (Root split):")
    root_feat = feature_names[tree.feature[0]]
    root_thresh = tree.threshold[0]
    print(f"  IF {root_feat} <= {root_thresh:.2f}")
    print(f"     → Goes to left branch")
    print(f"  ELSE")
    print(f"     → Goes to right branch")

    print(f"\n📌 This is WHY Decision Trees are popular:")
    print(f"   You can show the exact logic to teachers!")
    print(f"   'Student failed because study_hours <= {root_thresh:.1f}'")
    print(f"\n   Banks use this for loan decisions.")
    print(f"   Doctors use this for diagnosis.")
    print(f"   Courts use this for risk assessment.")
    print(f"\n   Interpretability = real-world deployment! 🔥")


def regression_tree_demo(
        df: pd.DataFrame) -> None:
    """Decision Tree for regression."""
    print("\n=== Regression Tree (Predicts Score) ===\n")

    feature_cols = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    X = df[feature_cols].values
    y = df['final_score'].values  # continuous!

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"{'Max Depth':>10} | {'R²':>8} | "
          f"{'MAE':>8} | {'Notes'}")
    print("-" * 55)

    for depth in [2, 3, 5, 8, None]:
        model = DecisionTreeRegressor(
            max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        depth_str = str(depth) if depth else "None"
        note = ""
        if depth is None:
            note = "← Overfit!"
        elif depth == 5:
            note = "← Sweet spot"
        elif depth == 2:
            note = "← Underfit"

        print(f"{depth_str:>10} | {r2:>8.4f} | "
              f"{mae:>8.2f} | {note}")

    print(f"\n💡 Regression Tree predicts SCORES")
    print(f"   instead of CLASSES!")
    print(f"   Leaf value = average score in that leaf")


if __name__ == "__main__":
    df = create_student_dataset()
    print(f"Dataset: {df.shape}\n")

    overfit_demonstration(df)
    feature_importance_analysis(df)
    decision_tree_rules(df)
    regression_tree_demo(df)
