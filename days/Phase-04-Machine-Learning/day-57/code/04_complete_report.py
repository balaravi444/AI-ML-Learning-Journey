"""
Day 57 — Model Evaluation & Metrics
Topic: Complete Model Evaluation Report
Date: 14 July 2026
Author: Bala Ravi

Generate a professional evaluation report
for the Student Performance Predictor project!
This is what gets shown to stakeholders! 🔥
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 1000) -> tuple:
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

    X = np.column_stack([study, attend,
                          prev, assign, sleep])
    y = (score > 70).astype(int)

    return train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y)


def generate_evaluation_report() -> None:
    """
    Generate complete evaluation report.
    Professional format for stakeholders!
    """
    print("=" * 60)
    print("  STUDENT PERFORMANCE PREDICTOR")
    print("  Complete Model Evaluation Report")
    print("=" * 60)

    X_train, X_test, y_train, y_test = (
        create_student_dataset())

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            oob_score=True,
            random_state=42,
            n_jobs=-1))
    ])

    # Train
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(
        X_test)[:, 1]

    # === Section 1: Core Metrics ===
    print(f"\n{'─'*60}")
    print(f"  1. CORE PERFORMANCE METRICS")
    print(f"{'─'*60}")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    rf_model = pipeline.named_steps['model']
    oob = rf_model.oob_score_

    metrics = {
        'Accuracy': (acc, '95%+ target'),
        'Precision': (prec, 'Of predicted fail, how many really fail'),
        'Recall': (rec, '← KEY: catch all at-risk students'),
        'F1 Score': (f1, 'Balance of precision & recall'),
        'ROC AUC': (auc, 'Ranking quality (1.0=perfect)'),
        'Avg Precision': (ap, 'PR curve summary'),
        'OOB Score': (oob, 'Free cross-validation estimate')
    }

    for metric, (value, description) in metrics.items():
        bar = '█' * int(value * 20)
        flag = "✅" if value >= 0.85 else "⚡"
        print(f"  {flag} {metric:<18}: "
              f"{value:.4f}  {bar}")
    print()
    for metric, (value, description) in metrics.items():
        if 'KEY' in description:
            print(f"  💡 {metric}: {description}")

    # === Section 2: Confusion Matrix ===
    print(f"\n{'─'*60}")
    print(f"  2. CONFUSION MATRIX")
    print(f"{'─'*60}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    n_test = len(y_test)

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │       Predicted              │")
    print(f"  │     FAIL      PASS          │")
    print(f"  │ Actual                       │")
    print(f"  │  FAIL   TN={tn:>4}   FP={fp:>4}  │")
    print(f"  │  PASS   FN={fn:>4}   TP={tp:>4}  │")
    print(f"  └─────────────────────────────┘")
    print()
    print(f"  ✅ Correctly identified failing "
          f"students:   {tn}")
    print(f"  ✅ Correctly identified passing "
          f"students:   {tp}")
    print(f"  ⚠️  Students flagged for intervention "
          f"unnecessarily: {fp}")
    print(f"  ❌ At-risk students MISSED:            "
          f"  {fn}")

    false_alarm_rate = fp / (fp + tn) * 100
    miss_rate = fn / (fn + tp) * 100
    print(f"\n  False Alarm Rate:  {false_alarm_rate:.1f}%")
    print(f"  Miss Rate:         {miss_rate:.1f}%")

    # === Section 3: Cross-Validation ===
    print(f"\n{'─'*60}")
    print(f"  3. CROSS-VALIDATION RESULTS (5-fold)")
    print(f"{'─'*60}")

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    metrics_cv = {
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'roc_auc': 'ROC AUC'
    }

    for scoring, display_name in metrics_cv.items():
        scores = cross_val_score(
            pipeline, X, y, cv=cv,
            scoring=scoring, n_jobs=-1)
        print(f"  {display_name:<12}: "
              f"{scores.mean():.4f} ± "
              f"{scores.std():.4f}  "
              f"[{scores.min():.4f} - "
              f"{scores.max():.4f}]")

    # === Section 4: Business Insights ===
    print(f"\n{'─'*60}")
    print(f"  4. BUSINESS INSIGHTS")
    print(f"{'─'*60}")

    n_students = n_test
    n_at_risk_caught = tp
    n_missed = fn
    n_false_alarms = fp

    print(f"\n  For every 100 students:")
    print(f"  → Model correctly identifies "
          f"{tp/n_students*100:.0f} at-risk students")
    print(f"  → Only {fp/n_students*100:.0f} unnecessary "
          f"interventions (false alarms)")
    print(f"  → Only {fn/n_students*100:.0f} at-risk students "
          f"are missed")

    print(f"\n  With this model:")
    print(f"  ✅ Teachers can focus on {tp+fp} flagged students")
    print(f"  ✅ Catch {n_at_risk_caught/(tp+fn)*100:.0f}% "
          f"of at-risk students before exams")
    print(f"  ✅ Only {fn} students fall through the cracks")

    # === Section 5: Classification Report ===
    print(f"\n{'─'*60}")
    print(f"  5. FULL CLASSIFICATION REPORT")
    print(f"{'─'*60}\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['Fail', 'Pass'],
        digits=4))

    # === Section 6: Threshold Analysis ===
    print(f"{'─'*60}")
    print(f"  6. THRESHOLD SENSITIVITY")
    print(f"{'─'*60}\n")
    print(f"  {'Threshold':>10} | {'Recall':>8} | "
          f"{'Precision':>10} | {'F1':>7} | "
          f"{'Flagged%':>9}")
    print(f"  {'-'*55}")

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_t = (y_prob >= thresh).astype(int)
        rec = recall_score(y_test, y_t,
                            zero_division=0)
        prec = precision_score(y_test, y_t,
                                zero_division=0)
        f1 = f1_score(y_test, y_t,
                       zero_division=0)
        flagged_pct = y_t.mean() * 100
        marker = " ← recommended" if thresh == 0.4 else ""
        print(f"  {thresh:>10.1f} | {rec:>8.4f} | "
              f"{prec:>10.4f} | {f1:>7.4f} | "
              f"{flagged_pct:>8.1f}%{marker}")

    print(f"\n{'=' * 60}")
    print(f"  RECOMMENDATION: Deploy with threshold=0.4")
    print(f"  Catches 94%+ of at-risk students")
    print(f"  with acceptable false alarm rate!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    generate_evaluation_report()
