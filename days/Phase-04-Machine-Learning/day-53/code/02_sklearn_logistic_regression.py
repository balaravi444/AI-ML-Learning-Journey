"""
Day 53 — Logistic Regression
Topic: Sklearn Logistic Regression — Full Usage
Date: 10 July 2026
Author: Bala Ravi

Complete classification workflow with sklearn!
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report,
    confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 1000) -> pd.DataFrame:
    """Create student pass/fail dataset."""
    np.random.seed(42)

    study_hours = np.random.uniform(1, 10, n)
    attendance = np.random.uniform(40, 100, n)
    prev_score = np.random.uniform(30, 95, n)
    assignments = np.random.randint(0, 10, n)
    sleep_hours = np.random.uniform(4, 9, n)

    # Pass if weighted score > 50
    score = (study_hours * 5 +
              attendance * 0.3 +
              prev_score * 0.4 +
              assignments * 1.5 +
              sleep_hours * 1.0 +
              np.random.normal(0, 5, n))

    passed = (score > 70).astype(int)

    return pd.DataFrame({
        'study_hours': study_hours.round(1),
        'attendance_pct': attendance.round(1),
        'prev_score': prev_score.round(1),
        'assignments_done': assignments,
        'sleep_hours': sleep_hours.round(1),
        'passed': passed
    })


def binary_classification_demo(
        df: pd.DataFrame) -> None:
    """Complete binary classification workflow."""
    print("=== Binary Classification: "
          "Student Pass/Fail ===\n")

    X = df.drop('passed', axis=1)
    y = df['passed']

    print(f"Dataset: {len(df)} students")
    print(f"Pass rate: {y.mean()*100:.1f}%")
    print(f"Fail rate: {(1-y).mean()*100:.1f}%\n")

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            C=1.0, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(
        X_test)[:, 1]

    # Metrics
    print("📊 Model Performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(f"  TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"  FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")
    print(f"\n  TN = correctly predicted FAIL")
    print(f"  TP = correctly predicted PASS")
    print(f"  FP = predicted PASS but actually FAIL")
    print(f"  FN = predicted FAIL but actually PASS")

    # Classification report
    print(f"\n📈 Full Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Fail', 'Pass']))

    # Feature importance
    model = pipeline.named_steps['model']
    scaler = pipeline.named_steps['scaler']
    features = X.columns.tolist()

    print("🎯 Feature Importance (coefficients):")
    coef_df = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient',
                   key=abs,
                   ascending=False)

    for _, row in coef_df.iterrows():
        direction = "↑ increases" if row['coefficient'] > 0 else "↓ decreases"
        print(f"  {row['feature']:<20}: "
              f"{row['coefficient']:>7.3f} "
              f"({direction} pass probability)")

    return pipeline


def threshold_tuning_demo(
        df: pd.DataFrame) -> None:
    """Show power of threshold tuning."""
    print("\n=== Threshold Tuning ===\n")
    print("Default threshold = 0.5")
    print("But we can tune it for our use case!\n")

    X = df.drop('passed', axis=1)
    y = df['passed']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(
        X_test)[:, 1]

    print(f"{'Threshold':>10} | {'Precision':>10} | "
          f"{'Recall':>8} | {'F1':>6} | "
          f"{'Use Case':>30}")
    print("-" * 75)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    use_cases = [
        "Catch all at-risk students",
        "Balanced early intervention",
        "Standard default",
        "High confidence predictions",
        "Only very confident fails"
    ]

    for threshold, use_case in zip(
            thresholds, use_cases):
        y_pred = (y_prob >= threshold).astype(int)
        prec = precision_score(
            y_test, y_pred, zero_division=0)
        rec = recall_score(
            y_test, y_pred, zero_division=0)
        f1 = f1_score(
            y_test, y_pred, zero_division=0)
        print(f"{threshold:>10.1f} | {prec:>10.3f} | "
              f"{rec:>8.3f} | {f1:>6.3f} | "
              f"{use_case:>30}")

    print(f"\n💡 For student intervention:")
    print(f"   Use threshold=0.3 to catch MORE")
    print(f"   at-risk students (higher recall)!")
    print(f"   Better to intervene unnecessarily")
    print(f"   than miss a struggling student! 🎓")


def predict_new_students(
        pipeline: Pipeline) -> None:
    """Predict outcomes for new students."""
    print("\n=== Predict New Students ===\n")

    new_students = pd.DataFrame({
        'study_hours': [8.5, 2.0, 5.5, 1.0, 7.0],
        'attendance_pct': [95, 45, 75, 30, 90],
        'prev_score': [85, 40, 65, 35, 80],
        'assignments_done': [9, 2, 6, 1, 8],
        'sleep_hours': [7.5, 5.0, 6.5, 4.0, 7.0]
    })

    predictions = pipeline.predict(new_students)
    probabilities = pipeline.predict_proba(
        new_students)[:, 1]

    print(f"{'Student':>8} | {'Study':>6} | "
          f"{'Attend':>7} | {'P(Pass)':>8} | "
          f"{'Prediction':>12}")
    print("-" * 55)

    for i, (_, row) in enumerate(
            new_students.iterrows()):
        pred = "✅ PASS" if predictions[i] == 1 else "❌ FAIL"
        print(f"{i+1:>8} | {row['study_hours']:>6.1f} | "
              f"{row['attendance_pct']:>7.0f} | "
              f"{probabilities[i]:>8.2%} | "
              f"{pred:>12}")


if __name__ == "__main__":
    df = create_student_dataset()

    pipeline = binary_classification_demo(df)
    threshold_tuning_demo(df)
    predict_new_students(pipeline)
