"""
Day 65 — Sentiment Analysis
Topic: Custom ML Sentiment Classifier
Date: 22 July 2026
Author: Bala Ravi

Train a sentiment classifier on bug reports!
Domain-specific → better than VADER for tech text.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# Labeled sentiment dataset for bug reports
SENTIMENT_DATA = [
    # Highly negative (frustrated/urgent)
    ("PRODUCTION IS COMPLETELY DOWN ALL USERS AFFECTED THIS IS A DISASTER", "negative"),
    ("Nothing works everything is broken third time this week", "negative"),
    ("This is absolutely terrible the system keeps crashing", "negative"),
    ("Critical failure data loss occurring this is unacceptable", "negative"),
    ("Complete service outage cannot believe this happened again", "negative"),
    ("The entire platform is unusable right now revenue loss ongoing", "negative"),
    ("Terrible performance users are furious and leaving the app", "negative"),
    ("This bug has been reported multiple times still not fixed", "negative"),
    ("Catastrophic failure all enterprise customers affected immediately", "negative"),
    ("System is throwing errors everywhere nothing is working properly", "negative"),
    # Moderately negative
    ("Login is not working for some users experiencing issues", "negative"),
    ("Several users reporting problems with the export feature", "negative"),
    ("The dashboard is showing incorrect data since the update", "negative"),
    ("Email delivery has been unreliable over the past week", "negative"),
    ("Performance degradation observed in the production environment", "negative"),
    ("Multiple complaints received about the search functionality", "negative"),
    ("Authentication failures occurring intermittently for some accounts", "negative"),
    ("Data inconsistencies found between the UI and API responses", "negative"),
    ("Webhook events are being dropped occasionally in production", "negative"),
    ("Some users unable to complete checkout process with errors", "negative"),
    # Neutral / factual
    ("Found a potential issue with the date picker component timezone", "neutral"),
    ("The sort order does not persist after page refresh", "neutral"),
    ("Noticed the footer copyright year has not been updated", "neutral"),
    ("The button tooltip disappears before users can read it", "neutral"),
    ("Column widths are inconsistent across different screen sizes", "neutral"),
    ("The loading state indicator remains visible after completion", "neutral"),
    ("Navigation breadcrumb is missing on the mobile viewport", "neutral"),
    ("Form validation error message appears twice on submission", "neutral"),
    ("The print layout clips the right column on A4 paper", "neutral"),
    ("Dropdown options are not sorted in alphabetical order", "neutral"),
    ("Minor display issue on the settings page noticed today", "neutral"),
    ("The modal close button requires double click on Firefox", "neutral"),
    ("Pagination component shows incorrect total count sometimes", "neutral"),
    ("The graph legend position overlaps with data on small screens", "neutral"),
    ("Search input placeholder text does not clear on focus event", "neutral"),
    # Positive
    ("Feature is working correctly after the recent fix deployed", "positive"),
    ("This improvement would make the workflow much smoother", "positive"),
    ("Great suggestion for enhancing the user experience overall", "positive"),
    ("The new design looks excellent and users are responding well", "positive"),
    ("Performance improvements are noticeable and working as expected", "positive"),
    ("The fix resolved the issue successfully all tests passing now", "positive"),
    ("Users are happy with the new feature functionality released", "positive"),
    ("Excellent work the deployment went smoothly without issues", "positive"),
    ("The enhancement request would significantly improve productivity", "positive"),
    ("This update is a great improvement over the previous version", "positive"),
]


def train_sentiment_classifier() -> Pipeline:
    """Train custom sentiment classifier."""
    print("=== Custom Sentiment Classifier ===\n")

    texts = [item[0] for item in SENTIMENT_DATA]
    labels = [item[1] for item in SENTIMENT_DATA]

    print(f"Dataset: {len(texts)} labeled examples")
    from collections import Counter
    dist = Counter(labels)
    for label, count in dist.items():
        print(f"  {label}: {count}")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=500,
            sublinear_tf=True)),
        ('model', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = (
        train_test_split(
            texts, labels,
            test_size=0.25,
            random_state=42,
            stratify=labels))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\nModel Performance:")
    print(f"  Accuracy: "
          f"{accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Macro: "
          f"{f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, texts, labels,
        cv=5, scoring='f1_macro')
    print(f"5-fold CV F1: "
          f"{cv_scores.mean():.4f} "
          f"± {cv_scores.std():.4f}")

    return pipeline


def sentiment_with_confidence(
        pipeline: Pipeline,
        texts: list) -> list:
    """
    Predict sentiment with confidence scores.

    Args:
        pipeline: Trained sentiment pipeline
        texts: List of texts to classify

    Returns:
        List of prediction dictionaries
    """
    predictions = pipeline.predict(texts)
    probabilities = pipeline.predict_proba(texts)
    classes = pipeline.classes_

    results = []
    for text, pred, proba in zip(
            texts, predictions, probabilities):
        prob_dict = {
            cls: float(round(p, 4))
            for cls, p in zip(classes, proba)}

        results.append({
            'text': text[:60] + '...' if len(text) > 60 else text,
            'sentiment': pred,
            'confidence': float(proba.max()),
            'probabilities': prob_dict
        })

    return results


def developer_frustration_detector(
        pipeline: Pipeline) -> None:
    """
    Detect developer frustration in bug reports.
    High frustration → higher urgency!
    """
    print("\n=== Developer Frustration Detector ===\n")
    print("Frustrated developers write differently!")
    print("Detecting frustration → better urgency!\n")

    bug_reports = [
        {
            'title': 'EVERYTHING IS BROKEN AGAIN!!!',
            'desc': 'Third time this month. '
                    'Nothing ever stays fixed. '
                    'This is completely unacceptable.'
        },
        {
            'title': 'Production database down',
            'desc': 'Database connection pool exhausted. '
                    'Error: FATAL timeout after 30s. '
                    'Investigating root cause.'
        },
        {
            'title': 'Date picker timezone issue',
            'desc': 'The date picker component shows UTC '
                    'instead of local time. Minor display issue.'
        },
        {
            'title': 'I give up - login broken AGAIN',
            'desc': 'This has been reported 5 times. '
                    'Absolutely terrible. Users are furious. '
                    'I cannot believe this is still broken.'
        },
        {
            'title': 'Small typo in error message',
            'desc': 'Found a typo on the billing page. '
                    'Easy fix when convenient.'
        }
    ]

    print(f"{'Report':<35} | "
          f"{'Sentiment':>9} | "
          f"{'Confidence':>11} | "
          f"{'Urgency Signal'}")
    print("-" * 80)

    for bug in bug_reports:
        full_text = (bug['title'] + ' ' + bug['desc'])
        result = sentiment_with_confidence(
            pipeline, [full_text])[0]

        sentiment = result['sentiment']
        confidence = result['confidence']

        urgency = (
            "🚨 HIGH" if (sentiment == 'negative' and
                           confidence > 0.8) else
            "⚡ MED" if sentiment == 'negative' else
            "📝 LOW" if sentiment == 'neutral' else
            "✅ NONE")

        print(f"{bug['title'][:33]:<35} | "
              f"{sentiment:>9} | "
              f"{confidence:>10.2%} | "
              f"{urgency}")

    print(f"\n💡 High frustration + negative sentiment")
    print(f"   → flag for immediate escalation!")
    print(f"   Even if issue type seems minor —")
    print(f"   repeated frustration = systemic problem!")


def combined_urgency_scorer(
        pipeline: Pipeline) -> None:
    """
    Combine sentiment + metadata for urgency score.
    """
    print("\n=== Combined Urgency Scoring ===\n")

    def urgency_score(
            text: str,
            hour: int,
            is_contributor: bool,
            sentiment_pipeline) -> float:
        """
        Compute composite urgency score.

        Args:
            text: Issue text
            hour: Hour filed
            is_contributor: Reporter status
            sentiment_pipeline: Trained pipeline

        Returns:
            Urgency score 0-10
        """
        result = sentiment_with_confidence(
            sentiment_pipeline, [text])[0]

        # Sentiment score (0-4)
        if result['sentiment'] == 'negative':
            sentiment_pts = 3 + result['confidence']
        elif result['sentiment'] == 'neutral':
            sentiment_pts = 1.5
        else:
            sentiment_pts = 0.5

        # Time score (0-2)
        time_pts = 2.0 if hour < 6 else (
            1.5 if hour > 20 else 0.5)

        # Contributor score (0-2)
        contrib_pts = 2.0 if is_contributor else 0.5

        # Keyword score (0-2)
        kw_pts = sum([
            0.5 for kw in [
                'crash', 'down', 'outage',
                'critical', 'fatal', 'data loss']
            if kw in text.lower()])
        kw_pts = min(kw_pts, 2.0)

        total = sentiment_pts + time_pts + (
            contrib_pts + kw_pts)
        return min(total, 10.0)

    test_issues = [
        ("PRODUCTION SERVER DOWN ALL USERS AFFECTED!!!",
         3, True, "Expected: Critical"),
        ("Login not working for some users",
         10, False, "Expected: High"),
        ("Date picker shows wrong timezone",
         14, False, "Expected: Medium"),
        ("Typo in the footer copyright year",
         15, False, "Expected: Low"),
        ("This is broken AGAIN - third time!!!",
         11, True, "Expected: High+"),
    ]

    print(f"{'Issue':<45} | "
          f"{'Urgency':>8} | "
          f"{'Note'}")
    print("-" * 75)

    for text, hour, contrib, note in test_issues:
        score = urgency_score(
            text, hour, contrib, pipeline)
        bar = '█' * int(score)
        print(f"{text[:43]:<45} | "
              f"{score:>8.2f} | {note}")

    print(f"\n✅ Urgency score = sentiment + time + "
          f"contributor + keywords")
    print(f"   This is Bug Predictor v2 thinking! 🔥")


if __name__ == "__main__":
    pipeline = train_sentiment_classifier()

    # Test on new texts
    test_texts = [
        "CRITICAL: Everything is broken production down NOW",
        "Minor issue with the button alignment on mobile",
        "The fix worked perfectly tests all passing",
        "Database connection failing intermittently"
    ]

    print("\n=== New Text Predictions ===\n")
    results = sentiment_with_confidence(
        pipeline, test_texts)

    for r in results:
        print(f"Text: {r['text']}")
        print(f"  Sentiment:   {r['sentiment']}")
        print(f"  Confidence:  {r['confidence']:.2%}")
        print(f"  Probs: neg={r['probabilities'].get('negative', 0):.2f} "
              f"neu={r['probabilities'].get('neutral', 0):.2f} "
              f"pos={r['probabilities'].get('positive', 0):.2f}")
        print()

    developer_frustration_detector(pipeline)
    combined_urgency_scorer(pipeline)
