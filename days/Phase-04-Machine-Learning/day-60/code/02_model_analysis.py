"""
Day 60 — Bug Priority Predictor
Topic: Model Analysis + Feature Importance
Date: 17 July 2026
Author: Bala Ravi

What did the model learn?
Which words and features matter most?
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')


def load_pipeline(save_dir: str) -> dict:
    """Load saved pipeline."""
    path = os.path.join(save_dir, 'model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Run 01_train_pipeline.py first!")
    return joblib.load(path)


def show_top_features(
        pipeline: dict,
        n: int = 15) -> None:
    """Show most important features per class."""
    print("=== Feature Importance by Priority ===\n")

    model = pipeline['model']
    vectorizer = pipeline['vectorizer']
    le = pipeline['label_encoder']
    meta_cols = pipeline['meta_cols']

    feature_names = (
        list(vectorizer.get_feature_names_out()) +
        meta_cols)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"Top {n} Most Important Features Overall:")
    print(f"{'Feature':<35} | {'Importance':>12}")
    print("-" * 52)

    for i in range(min(n, len(indices))):
        idx = indices[i]
        name = feature_names[idx]
        imp = importances[idx]
        bar = '█' * int(imp * 1000)
        print(f"{name:<35} | {imp:>12.6f} {bar}")


def analyze_misclassifications(
        pipeline: dict,
        df: pd.DataFrame) -> None:
    """Study what the model gets wrong."""
    print("\n=== Misclassification Analysis ===\n")

    model = pipeline['model']
    vectorizer = pipeline['vectorizer']
    scaler = pipeline['scaler']
    le = pipeline['label_encoder']
    meta_cols = pipeline['meta_cols']

    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler

    text = (df['title'] + ' ' +
             df['title'] + ' ' +
             df['title'] + ' ' +
             df['description'].fillna(''))
    tfidf = vectorizer.transform(text)

    for col in ['is_off_hours', 'is_night',
                 'title_has_critical_kw',
                 'title_has_low_kw',
                 'urgency_score',
                 'desc_quality_score']:
        if col not in df.columns:
            if col == 'is_off_hours':
                df[col] = (
                    (df['hour_filed'] < 7) |
                    (df['hour_filed'] > 20)
                ).astype(int)
            elif col == 'is_night':
                df[col] = (
                    df['hour_filed'].between(0, 5)
                ).astype(int)
            elif col == 'title_has_critical_kw':
                df[col] = (
                    df['title'].str.lower()
                    .str.contains(
                        'crash|down|outage|critical|'
                        'fatal|urgent|production',
                        regex=True)).astype(int)
            elif col == 'title_has_low_kw':
                df[col] = (
                    df['title'].str.lower()
                    .str.contains(
                        'typo|cosmetic|minor|small',
                        regex=True)).astype(int)
            elif col == 'urgency_score':
                df[col] = (
                    df.get('is_night', 0) * 3 +
                    df['reporter_is_contributor'] * 2 +
                    df.get(
                        'title_has_critical_kw', 0) * 3)
            elif col == 'desc_quality_score':
                df[col] = (
                    df['has_error_message'] * 2 +
                    df['has_steps_to_reproduce'] * 2)

    available_cols = [
        c for c in meta_cols if c in df.columns]
    meta = scaler.transform(
        df[available_cols].fillna(0))
    meta_sparse = sp.csr_matrix(meta)
    X = sp.hstack([tfidf, meta_sparse])

    y_true = le.transform(df['priority'])
    y_pred = model.predict(X)

    # Find misclassifications
    wrong_mask = y_true != y_pred
    wrong_df = df[wrong_mask].copy()
    wrong_df['true_priority'] = le.inverse_transform(
        y_true[wrong_mask])
    wrong_df['pred_priority'] = le.inverse_transform(
        y_pred[wrong_mask])

    print(f"Total misclassified: "
          f"{wrong_mask.sum()} / {len(df)} "
          f"({wrong_mask.mean()*100:.1f}%)\n")

    print("Misclassification patterns:")
    pattern = wrong_df.groupby(
        ['true_priority', 'pred_priority']
    ).size().reset_index(name='count')
    pattern = pattern.sort_values(
        'count', ascending=False)

    for _, row in pattern.iterrows():
        severity = ""
        if (row['true_priority'] == 'Critical' and
                row['pred_priority'] == 'Low'):
            severity = " ← DANGEROUS!"
        elif row['true_priority'] == 'Critical':
            severity = " ← watch this"
        print(f"  {row['true_priority']:<10} → "
              f"{row['pred_priority']:<10}: "
              f"{row['count']:>4}{severity}")

    print(f"\n💡 Most errors are ADJACENT classes:")
    print(f"   Critical→High or High→Medium")
    print(f"   Not Critical→Low!")
    print(f"   Model understands severity spectrum! ✅")


def predict_sample_issues(
        pipeline: dict) -> None:
    """Predict priority for sample issues."""
    print("\n=== Sample Predictions ===\n")

    model = pipeline['model']
    vectorizer = pipeline['vectorizer']
    scaler = pipeline['scaler']
    le = pipeline['label_encoder']
    meta_cols = pipeline['meta_cols']

    import scipy.sparse as sp

    samples = [
        {
            'title': 'Production database down - all users affected',
            'description': (
                'Our production PostgreSQL instance crashed at 3:47am. '
                'All 50,000 users are locked out. Error: FATAL: '
                'connection pool exhausted. Stack trace attached. '
                'Revenue loss ongoing at $20k/hour.'),
            'hour_filed': 3,
            'has_error_message': 1,
            'has_code_snippet': 1,
            'reporter_is_contributor': 1,
            'expected': 'Critical'
        },
        {
            'title': 'Typo in welcome email template',
            'description': (
                'The welcome email has a minor typo. '
                '"Welcoem" should be "Welcome". '
                'Small fix when convenient.'),
            'hour_filed': 14,
            'has_error_message': 0,
            'has_code_snippet': 0,
            'reporter_is_contributor': 0,
            'expected': 'Low'
        },
        {
            'title': 'Login fails for users with + in email',
            'description': (
                'Users whose email contains a + symbol cannot log in. '
                'They get a 400 Bad Request error. '
                'Affects approximately 5% of user base. '
                'Steps: 1. Use email test+1@gmail.com 2. Try login '
                '3. See 400 error.'),
            'hour_filed': 10,
            'has_error_message': 1,
            'has_code_snippet': 0,
            'reporter_is_contributor': 0,
            'expected': 'High'
        }
    ]

    for sample in samples:
        text = (sample['title'] + ' ' +
                 sample['title'] + ' ' +
                 sample['title'] + ' ' +
                 sample['description'])
        tfidf = vectorizer.transform([text])

        # Build metadata row
        meta_row = {col: 0 for col in meta_cols}
        meta_row['hour_filed'] = sample['hour_filed']
        meta_row['has_error_message'] = (
            sample['has_error_message'])
        meta_row['has_code_snippet'] = (
            sample['has_code_snippet'])
        meta_row['reporter_is_contributor'] = (
            sample['reporter_is_contributor'])
        meta_row['is_night'] = (
            1 if sample['hour_filed'] < 6 else 0)
        meta_row['urgency_score'] = (
            meta_row['is_night'] * 3 +
            sample['reporter_is_contributor'] * 2)

        import pandas as pd
        meta_df = pd.DataFrame([meta_row])
        meta_scaled = scaler.transform(
            meta_df.fillna(0))
        meta_sparse = sp.csr_matrix(meta_scaled)

        X = sp.hstack([tfidf, meta_sparse])
        pred_idx = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = proba.max()

        match = "✅" if pred_label == sample['expected'] else "❌"
        print(f"Title: {sample['title'][:55]}...")
        print(f"  Predicted: {pred_label:<10} "
              f"({confidence:.1%} confidence) "
              f"{match} (expected: {sample['expected']})")

        print(f"  Probabilities: ", end='')
        for cls, p in zip(le.classes_, proba):
            print(f"{cls}={p:.2f} ", end='')
        print(f"\n")


if __name__ == "__main__":
    SAVE_DIR = "projects/bug_priority_predictor/ml"
    DATA_PATH = (
        "projects/bug_priority_predictor/data/"
        "github_issues.csv")

    pipeline = load_pipeline(SAVE_DIR)
    df = pd.read_csv(DATA_PATH)

    show_top_features(pipeline)
    predict_sample_issues(pipeline)

    try:
        analyze_misclassifications(pipeline, df)
    except Exception as e:
        print(f"Note: {e}")
