"""
Day 59 — Software Bug Priority Predictor
Topic: Feature Engineering for Bug Reports
Date: 16 July 2026
Author: Bala Ravi

Converting raw GitHub issue text + metadata
into ML-ready feature matrix.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler)
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')


# Priority keywords — high TF-IDF weight expected
CRITICAL_KEYWORDS = [
    'production', 'down', 'outage', 'crash',
    'fatal', 'critical', 'urgent', 'data loss',
    'corruption', 'security', 'breach', 'all users',
    'revenue', 'p0', 'site down', 'broken',
    'cannot access', 'not working', '500', 'timeout'
]

HIGH_KEYWORDS = [
    'fails', 'broken', 'incorrect', 'wrong',
    'not working', 'error', 'failure', 'blocked',
    'affecting', 'customers', 'enterprise',
    'intermittent', 'significant'
]

MEDIUM_KEYWORDS = [
    'incorrect', 'confusing', 'annoying',
    'inconsistent', 'resets', 'missing',
    'overlaps', 'sometimes', 'occasionally'
]

LOW_KEYWORDS = [
    'typo', 'cosmetic', 'minor', 'small',
    'grammatical', 'spacing', 'colour',
    'slightly', 'outdated', 'label'
]


def engineer_text_features(
        df: pd.DataFrame,
        max_features: int = 500,
        fit: bool = True,
        vectorizer=None) -> tuple:
    """
    Convert title + description text to TF-IDF features.

    Combines title (weighted 3x) + description for TF-IDF.
    Title words matter MORE than description words!

    Args:
        df: DataFrame with title and description columns
        max_features: Max TF-IDF features
        fit: Whether to fit or just transform
        vectorizer: Pre-fitted vectorizer (for inference)

    Returns:
        (sparse matrix, fitted vectorizer)
    """
    # Combine title (weighted 3x) + description
    text_combined = (
        df['title'] + ' ' +
        df['title'] + ' ' +
        df['title'] + ' ' +
        df['description'].fillna(''))

    if fit:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),     # unigrams + bigrams!
            stop_words='english',
            min_df=2,               # ignore rare terms
            sublinear_tf=True       # log(tf) scaling
        )
        tfidf_matrix = vectorizer.fit_transform(
            text_combined)
    else:
        tfidf_matrix = vectorizer.transform(
            text_combined)

    return tfidf_matrix, vectorizer


def engineer_metadata_features(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and engineer metadata features.

    These amplify the text signals:
    Critical text + filed at 2am + reporter is contributor
    = almost certainly Critical!

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame of engineered numerical features
    """
    features = pd.DataFrame()

    # Direct metadata
    features['label_count'] = df['label_count']
    features['comment_count'] = df['comment_count']
    features['reporter_is_contributor'] = (
        df['reporter_is_contributor'])
    features['description_length'] = (
        df['description_length'])
    features['word_count'] = df['word_count']
    features['has_error_message'] = (
        df['has_error_message'])
    features['has_steps_to_reproduce'] = (
        df['has_steps_to_reproduce'])
    features['has_code_snippet'] = df['has_code_snippet']
    features['has_version_number'] = (
        df['has_version_number'])
    features['exclamation_count'] = (
        df['exclamation_count'])

    # Time-based features
    features['hour_filed'] = df['hour_filed']
    features['is_off_hours'] = (
        (df['hour_filed'] < 7) |
        (df['hour_filed'] > 20)).astype(int)
    features['is_night'] = (
        (df['hour_filed'] >= 0) &
        (df['hour_filed'] <= 5)).astype(int)

    # Keyword signals in title
    title_lower = df['title'].str.lower()
    features['title_has_critical_kw'] = (
        title_lower.str.contains(
            '|'.join(['crash', 'down', 'outage',
                      'critical', 'fatal', 'urgent',
                      'production', 'security',
                      'data loss']),
            regex=True)).astype(int)
    features['title_has_low_kw'] = (
        title_lower.str.contains(
            '|'.join(['typo', 'cosmetic', 'minor',
                      'small', 'spacing', 'grammatical',
                      'colour', 'color']),
            regex=True)).astype(int)

    # Quality signals
    features['desc_quality_score'] = (
        features['has_error_message'] * 2 +
        features['has_steps_to_reproduce'] * 2 +
        features['has_code_snippet'] * 1 +
        features['has_version_number'] * 1)

    # Urgency score
    features['urgency_score'] = (
        features['is_night'] * 3 +
        features['reporter_is_contributor'] * 2 +
        features['title_has_critical_kw'] * 3 +
        features['exclamation_count'] * 0.5)

    return features


def combine_features(
        tfidf_matrix: sp.csr_matrix,
        metadata_df: pd.DataFrame) -> sp.csr_matrix:
    """
    Combine TF-IDF sparse matrix + metadata features.

    Args:
        tfidf_matrix: Sparse TF-IDF matrix
        metadata_df: Dense metadata features

    Returns:
        Combined sparse feature matrix
    """
    # Scale metadata before combining
    scaler = StandardScaler()
    metadata_scaled = scaler.fit_transform(metadata_df)
    metadata_sparse = sp.csr_matrix(metadata_scaled)

    # Combine horizontally
    combined = sp.hstack(
        [tfidf_matrix, metadata_sparse])

    return combined, scaler


def demonstrate_features(df: pd.DataFrame) -> None:
    """Show feature engineering results."""
    print("=== Feature Engineering Demo ===\n")

    # Sample one issue per priority
    for priority in ['Critical', 'High',
                      'Medium', 'Low']:
        sample = df[df['priority'] == priority].iloc[0]
        meta = engineer_metadata_features(
            df[df['priority'] == priority].head(1))

        print(f"{'='*50}")
        print(f"Priority: {priority}")
        print(f"Title: {sample['title'][:60]}...")
        print(f"\nEngineered Features:")
        print(f"  Urgency score:      "
              f"{meta['urgency_score'].iloc[0]:.1f}")
        print(f"  Desc quality:       "
              f"{meta['desc_quality_score'].iloc[0]:.1f}")
        print(f"  Has error msg:      "
              f"{bool(meta['has_error_message'].iloc[0])}")
        print(f"  Filed at night:     "
              f"{bool(meta['is_night'].iloc[0])}")
        print(f"  Critical keywords:  "
              f"{bool(meta['title_has_critical_kw'].iloc[0])}")
        print(f"  Comment count:      "
              f"{sample['comment_count']}")
        print()


def prepare_ml_data(df: pd.DataFrame) -> dict:
    """
    Full feature engineering pipeline.
    Returns everything needed for ML training.

    Args:
        df: Raw issues DataFrame

    Returns:
        Dictionary with X, y, vectorizer, scaler
    """
    print("Preparing ML features...")

    # Text features
    tfidf_matrix, vectorizer = engineer_text_features(df)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    # Metadata features
    metadata = engineer_metadata_features(df)
    print(f"  Metadata features: {metadata.shape[1]}")

    # Combine
    X, scaler = combine_features(tfidf_matrix, metadata)
    print(f"  Combined matrix: {X.shape}")

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df['priority'])
    print(f"  Label classes: {le.classes_}")

    print(f"\n✅ Feature matrix ready: {X.shape}")
    print(f"   {X.shape[1]} features per issue")

    return {
        'X': X,
        'y': y,
        'label_encoder': le,
        'vectorizer': vectorizer,
        'scaler': scaler,
        'feature_names': (
            vectorizer.get_feature_names_out().tolist() +
            list(metadata.columns))
    }


if __name__ == "__main__":
    df = pd.read_csv(
        "projects/bug_priority_predictor/data/"
        "github_issues.csv")

    print(f"Loaded {len(df)} issues\n")

    demonstrate_features(df)
    ml_data = prepare_ml_data(df)

    print(f"\n💡 Feature Engineering Key Points:")
    print(f"   TF-IDF: words → numbers (text meaning)")
    print(f"   Title weighted 3x (most important!)")
    print(f"   Urgency score: night filing + "
          f"keywords + contributor")
    print(f"   Quality score: error msg + "
          f"steps + code snippet")
    print(f"   All combined → one feature matrix!")
