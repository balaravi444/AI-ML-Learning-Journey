"""
Day 60 — Bug Priority Predictor
Topic: Full ML Training Pipeline
Date: 17 July 2026
Author: Bala Ravi

TF-IDF + Random Forest + SMOTE = production ML!
Handles class imbalance, saves full pipeline.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler)
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠️  imbalanced-learn not installed.")
    print("    Run: pip install imbalanced-learn")
    print("    Continuing without SMOTE...\n")


def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} issues")
    print(f"   Columns: {list(df.columns)}\n")
    return df


def build_features(
        df: pd.DataFrame,
        max_tfidf: int = 500) -> tuple:
    """
    Build complete feature matrix.
    TF-IDF (text) + metadata (numerical) combined.

    Args:
        df: Raw issues DataFrame
        max_tfidf: Max TF-IDF vocabulary size

    Returns:
        X (combined features), y (labels),
        vectorizer, scaler, label_encoder
    """
    # Text: title weighted 3x + description
    text = (df['title'] + ' ' +
             df['title'] + ' ' +
             df['title'] + ' ' +
             df['description'].fillna(''))

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_tfidf,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        sublinear_tf=True)
    tfidf = vectorizer.fit_transform(text)

    # Metadata features
    df['is_off_hours'] = (
        (df['hour_filed'] < 7) |
        (df['hour_filed'] > 20)).astype(int)
    df['is_night'] = (
        df['hour_filed'].between(0, 5)).astype(int)
    df['title_has_critical_kw'] = (
        df['title'].str.lower().str.contains(
            'crash|down|outage|critical|fatal|'
            'urgent|production|security|data loss',
            regex=True)).astype(int)
    df['title_has_low_kw'] = (
        df['title'].str.lower().str.contains(
            'typo|cosmetic|minor|small|spacing|'
            'grammatical|colour|color',
            regex=True)).astype(int)
    df['urgency_score'] = (
        df['is_night'] * 3 +
        df['reporter_is_contributor'] * 2 +
        df['title_has_critical_kw'] * 3 +
        df['exclamation_count'] * 0.5)
    df['desc_quality_score'] = (
        df['has_error_message'] * 2 +
        df['has_steps_to_reproduce'] * 2 +
        df['has_code_snippet'] * 1 +
        df['has_version_number'] * 1)

    meta_cols = [
        'label_count', 'comment_count',
        'reporter_is_contributor',
        'description_length', 'word_count',
        'has_error_message',
        'has_steps_to_reproduce',
        'has_code_snippet', 'has_version_number',
        'exclamation_count', 'hour_filed',
        'is_off_hours', 'is_night',
        'title_has_critical_kw',
        'title_has_low_kw',
        'urgency_score', 'desc_quality_score'
    ]

    scaler = StandardScaler()
    meta_scaled = scaler.fit_transform(
        df[meta_cols].fillna(0))
    meta_sparse = sp.csr_matrix(meta_scaled)

    # Combine
    X = sp.hstack([tfidf, meta_sparse])

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df['priority'])

    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {le.classes_}\n")

    return X, y, vectorizer, scaler, le, meta_cols


def handle_class_imbalance(
        X: sp.csr_matrix,
        y: np.ndarray,
        label_encoder: LabelEncoder) -> tuple:
    """
    Apply SMOTE to balance classes.

    Args:
        X: Feature matrix (sparse)
        y: Labels
        label_encoder: Fitted label encoder

    Returns:
        X_resampled, y_resampled
    """
    print("Class distribution BEFORE SMOTE:")
    for label, count in zip(
            *np.unique(y, return_counts=True)):
        cls = label_encoder.classes_[label]
        pct = count / len(y) * 100
        print(f"  {cls:<10}: {count:>4} ({pct:.1f}%)")

    if not SMOTE_AVAILABLE:
        print("\n⚠️  Skipping SMOTE — not installed")
        return X, y

    smote = SMOTE(
        random_state=42,
        k_neighbors=3)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"\nClass distribution AFTER SMOTE:")
    for label, count in zip(
            *np.unique(y_res, return_counts=True)):
        cls = label_encoder.classes_[label]
        pct = count / len(y_res) * 100
        print(f"  {cls:<10}: {count:>4} ({pct:.1f}%)")

    print(f"\n✅ Dataset size: {len(y)} → {len(y_res)}")
    return X_res, y_res


def train_model(
        X_train: sp.csr_matrix,
        y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train Random Forest on bug features.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained model
    """
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        oob_score=True,
        random_state=42,
        n_jobs=-1)

    model.fit(X_train, y_train)
    print(f"✅ OOB Score: {model.oob_score_:.4f}")
    return model


def evaluate_model(
        model: RandomForestClassifier,
        X_test: sp.csr_matrix,
        y_test: np.ndarray,
        label_encoder: LabelEncoder) -> None:
    """Full evaluation with multi-class metrics."""
    print("\n=== Model Evaluation ===\n")

    y_pred = model.predict(X_test)
    classes = label_encoder.classes_

    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(
        y_test, y_pred, average='weighted')
    f1_macro = f1_score(
        y_test, y_pred, average='macro')

    print(f"Overall Metrics:")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 Weighted: {f1_weighted:.4f}")
    print(f"  F1 Macro:    {f1_macro:.4f}")

    print(f"\nPer-class Performance:")
    print(classification_report(
        y_test, y_pred,
        target_names=classes,
        digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':>12}", end='')
    for c in classes:
        print(f"{c[:6]:>8}", end='')
    print(f"\n{'─'*55}")

    for i, actual in enumerate(classes):
        print(f"{actual:<12}", end='')
        for j in range(len(classes)):
            val = cm[i][j]
            marker = ' ←' if i == j else ''
            print(f"{val:>8}{marker[:1]}", end='')
        print()

    # Critical recall — most important!
    crit_idx = list(classes).index('Critical')
    crit_recall = (cm[crit_idx][crit_idx] /
                   cm[crit_idx].sum())
    print(f"\n🚨 Critical Bug Recall: "
          f"{crit_recall:.4f}")
    if crit_recall >= 0.85:
        print(f"   ✅ Excellent! Catching "
              f"{crit_recall*100:.0f}% of critical bugs!")
    else:
        print(f"   ⚠️  Missing too many critical bugs!")


def save_pipeline(
        model, vectorizer, scaler,
        label_encoder, meta_cols,
        save_dir: str) -> None:
    """Save complete pipeline for deployment."""
    os.makedirs(save_dir, exist_ok=True)

    pipeline = {
        'model': model,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'meta_cols': meta_cols
    }

    path = os.path.join(save_dir, 'model.pkl')
    joblib.dump(pipeline, path)
    size = os.path.getsize(path) / (1024 * 1024)
    print(f"\n✅ Pipeline saved: {path}")
    print(f"   Size: {size:.1f} MB")


if __name__ == "__main__":
    DATA_PATH = (
        "projects/bug_priority_predictor/data/"
        "github_issues.csv")
    SAVE_DIR = "projects/bug_priority_predictor/ml"

    df = load_data(DATA_PATH)

    X, y, vectorizer, scaler, le, meta_cols = (
        build_features(df))

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42,
            stratify=y))

    X_train_bal, y_train_bal = (
        handle_class_imbalance(
            X_train, y_train, le))

    model = train_model(X_train_bal, y_train_bal)

    evaluate_model(model, X_test, y_test, le)

    save_pipeline(
        model, vectorizer, scaler,
        le, meta_cols, SAVE_DIR)
