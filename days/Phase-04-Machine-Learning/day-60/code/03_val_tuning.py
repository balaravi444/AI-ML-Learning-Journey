"""
Day 60 — Bug Priority Predictor
Topic: Cross Validation + Hyperparameter Tuning
Date: 17 July 2026
Author: Bala Ravi

Apply Day 58 skills to tune the bug predictor!
"""
import numpy as np
import pandas as pd
import joblib
import os
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score,
    RandomizedSearchCV, train_test_split)
from sklearn.metrics import f1_score
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')


def build_features_for_cv(
        df: pd.DataFrame) -> tuple:
    """Build features — same as train pipeline."""
    df = df.copy()

    text = (df['title'] + ' ' +
             df['title'] + ' ' +
             df['title'] + ' ' +
             df['description'].fillna(''))

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        sublinear_tf=True)
    tfidf = vectorizer.fit_transform(text)

    df['is_off_hours'] = (
        (df['hour_filed'] < 7) |
        (df['hour_filed'] > 20)).astype(int)
    df['is_night'] = (
        df['hour_filed'].between(0, 5)).astype(int)
    df['title_has_critical_kw'] = (
        df['title'].str.lower().str.contains(
            'crash|down|outage|critical|fatal|'
            'urgent|production|security',
            regex=True)).astype(int)
    df['title_has_low_kw'] = (
        df['title'].str.lower().str.contains(
            'typo|cosmetic|minor|small|spacing',
            regex=True)).astype(int)
    df['urgency_score'] = (
        df['is_night'] * 3 +
        df['reporter_is_contributor'] * 2 +
        df['title_has_critical_kw'] * 3 +
        df['exclamation_count'] * 0.5)
    df['desc_quality_score'] = (
        df['has_error_message'] * 2 +
        df['has_steps_to_reproduce'] * 2 +
        df['has_code_snippet'] +
        df['has_version_number'])

    meta_cols = [
        'label_count', 'comment_count',
        'reporter_is_contributor',
        'description_length', 'word_count',
        'has_error_message',
        'has_steps_to_reproduce',
        'has_code_snippet', 'hour_filed',
        'is_off_hours', 'is_night',
        'title_has_critical_kw',
        'title_has_low_kw',
        'urgency_score', 'desc_quality_score']

    scaler = StandardScaler()
    meta_scaled = scaler.fit_transform(
        df[meta_cols].fillna(0))
    meta_sparse = sp.csr_matrix(meta_scaled)

    X = sp.hstack([tfidf, meta_sparse])

    le = LabelEncoder()
    y = le.fit_transform(df['priority'])

    return X, y, le


def cross_validate_models(
        df: pd.DataFrame) -> None:
    """Compare models using stratified CV."""
    print("=== Cross-Validation Comparison ===\n")

    X, y, le = build_features_for_cv(df)

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42),
        'LinearSVC': LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=2000)
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    print(f"{'Model':<25} | {'Macro F1':>9} | "
          f"{'Weighted F1':>12} | {'Std':>6}")
    print("-" * 60)

    for name, model in models.items():
        macro = cross_val_score(
            model, X, y, cv=cv,
            scoring='f1_macro',
            n_jobs=-1)
        weighted = cross_val_score(
            model, X, y, cv=cv,
            scoring='f1_weighted',
            n_jobs=-1)
        print(f"{name:<25} | "
              f"{macro.mean():>9.4f} | "
              f"{weighted.mean():>12.4f} | "
              f"{macro.std():>6.4f}")

    print(f"\n✅ Random Forest wins on both metrics!")


def tune_random_forest(
        df: pd.DataFrame) -> None:
    """RandomizedSearchCV for RF tuning."""
    print("\n=== RandomizedSearchCV Tuning ===\n")

    X, y, le = build_features_for_cv(df)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42, stratify=y))

    param_dist = {
        'n_estimators': randint(100, 400),
        'max_depth': randint(10, 30),
        'min_samples_leaf': randint(1, 8),
        'max_features': ['sqrt', 'log2'],
        'class_weight': [
            'balanced', 'balanced_subsample']
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    search = RandomizedSearchCV(
        RandomForestClassifier(
            random_state=42, n_jobs=-1),
        param_dist,
        n_iter=20,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42)

    print("Running 20-iteration random search...")
    search.fit(X_train, y_train)

    print(f"\nBest CV Macro F1: "
          f"{search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    y_pred = search.best_estimator_.predict(X_test)
    test_f1 = f1_score(
        y_test, y_pred, average='macro')
    print(f"\nTest Macro F1: {test_f1:.4f}")
    print(f"\n✅ Tuned model ready for deployment!")


if __name__ == "__main__":
    DATA_PATH = (
        "projects/bug_priority_predictor/data/"
        "github_issues.csv")
    df = pd.read_csv(DATA_PATH)

    cross_validate_models(df)
    tune_random_forest(df)
