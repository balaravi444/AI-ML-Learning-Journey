"""
Day 65 — Sentiment Analysis
Topic: Aspect-Based Sentiment Analysis
Date: 22 July 2026
Author: Bala Ravi

Overall sentiment is not enough!
"Login works but dashboard is broken"
→ Overall: mixed
→ login: positive, dashboard: negative

Much more actionable for product teams!
"""
import re
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Software component aspects
SOFTWARE_ASPECTS = {
    'authentication': [
        'login', 'logout', 'signup', 'register',
        'password', 'auth', 'oauth', 'sso',
        'session', 'token', '2fa'],
    'performance': [
        'slow', 'fast', 'speed', 'latency',
        'timeout', 'loading', 'response time',
        'lag', 'freeze', 'performance'],
    'ui': [
        'button', 'dashboard', 'display',
        'layout', 'design', 'css', 'style',
        'color', 'spacing', 'font', 'ui',
        'interface', 'screen', 'modal'],
    'data': [
        'database', 'data', 'export', 'import',
        'csv', 'backup', 'sync', 'migration',
        'query', 'storage'],
    'api': [
        'api', 'endpoint', 'request', 'response',
        'webhook', 'integration', 'rest', 'json'],
    'notifications': [
        'email', 'notification', 'alert',
        'message', 'sms', 'push', 'slack']
}

# Sentiment indicators
NEGATIVE_WORDS = {
    'broken', 'crashed', 'failed', 'error',
    'wrong', 'incorrect', 'missing', 'not',
    'never', 'cannot', 'fails', 'issue',
    'problem', 'bug', 'down', 'slow',
    'terrible', 'awful', 'broken', 'unusable'
}

POSITIVE_WORDS = {
    'working', 'fixed', 'resolved', 'great',
    'excellent', 'perfect', 'smooth', 'fast',
    'correct', 'improved', 'better', 'good',
    'works', 'success', 'passed', 'correct'
}


def extract_aspect_sentiment(
        text: str) -> dict:
    """
    Extract sentiment per aspect.

    Simple rule-based approach:
    1. Find aspect keywords in text
    2. Check surrounding words for sentiment
    3. Assign positive/negative/neutral per aspect

    Args:
        text: Input text

    Returns:
        Dictionary of aspect → sentiment
    """
    text_lower = text.lower()
    words = text_lower.split()

    aspect_sentiments = {}

    for aspect, keywords in (
            SOFTWARE_ASPECTS.items()):
        # Check if aspect mentioned
        mentioned = [
            kw for kw in keywords
            if kw in text_lower]

        if not mentioned:
            continue

        # Find sentiment in context window
        # around each keyword mention
        pos_count = 0
        neg_count = 0

        for keyword in mentioned:
            kw_idx = None
            for i, word in enumerate(words):
                if keyword in word:
                    kw_idx = i
                    break

            if kw_idx is None:
                continue

            # Window of 5 words around keyword
            window_start = max(0, kw_idx - 5)
            window_end = min(
                len(words), kw_idx + 6)
            window = words[window_start:window_end]

            for w in window:
                clean_w = re.sub(
                    r'[^a-zA-Z]', '', w)
                if clean_w in POSITIVE_WORDS:
                    pos_count += 1
                if clean_w in NEGATIVE_WORDS:
                    neg_count += 1

        if neg_count > pos_count:
            sentiment = 'negative'
            confidence = neg_count / max(
                neg_count + pos_count, 1)
        elif pos_count > neg_count:
            sentiment = 'positive'
            confidence = pos_count / max(
                neg_count + pos_count, 1)
        else:
            sentiment = 'neutral'
            confidence = 0.5

        aspect_sentiments[aspect] = {
            'sentiment': sentiment,
            'confidence': confidence,
            'keywords_found': mentioned,
            'pos_signals': pos_count,
            'neg_signals': neg_count
        }

    return aspect_sentiments


def demonstrate_aspect_sentiment() -> None:
    """Show aspect-based sentiment on bug reports."""
    print("=== Aspect-Based Sentiment Analysis ===\n")

    reviews = [
        "The login is working great now but the dashboard "
        "is completely broken and very slow.",

        "Authentication failed for all users. "
        "The API is returning 500 errors. "
        "Database seems fine though.",

        "Email notifications are not being delivered. "
        "The UI looks excellent after the redesign. "
        "Performance is much faster now.",

        "Export to CSV is broken producing wrong data. "
        "Login works correctly. Notifications not sent.",
    ]

    for i, review in enumerate(reviews, 1):
        print(f"Review {i}: {review}\n")

        aspects = extract_aspect_sentiment(review)

        if not aspects:
            print("  No aspects detected.\n")
            continue

        print(f"  {'Aspect':<18} | "
              f"{'Sentiment':>9} | "
              f"{'Confidence':>11} | "
              f"{'Keywords'}")
        print(f"  {'-'*65}")

        for aspect, data in aspects.items():
            emoji = (
                '✅' if data['sentiment'] == 'positive'
                else '❌' if data['sentiment'] == 'negative'
                else '➡️')
            print(f"  {aspect:<18} | "
                  f"{emoji} {data['sentiment']:>7} | "
                  f"{data['confidence']:>10.0%} | "
                  f"{data['keywords_found']}")
        print()


def product_health_monitor() -> None:
    """
    Monitor product health from bug reports.
    Which components are most complained about?
    """
    print("=== Product Health Monitor ===\n")
    print("Aggregate sentiment across all bug reports\n")

    bug_reports = [
        "Login completely broken cannot authenticate",
        "Authentication timeout happening for all users",
        "Dashboard is very slow loading charts",
        "API returning 500 errors on all endpoints",
        "Email notifications not being delivered",
        "Password reset email never arrives",
        "Data export corrupted for large files",
        "CSV import failing for 30% of records",
        "Login now working after the hotfix",
        "Performance improved significantly after update",
        "API response time much better now",
        "Dashboard loads quickly excellent work",
        "Button alignment slightly off on mobile",
        "Minor spacing issue in the settings form",
    ]

    # Aggregate aspect sentiments
    aspect_scores = defaultdict(
        lambda: {'pos': 0, 'neg': 0, 'neu': 0})

    for report in bug_reports:
        aspects = extract_aspect_sentiment(report)
        for aspect, data in aspects.items():
            aspect_scores[aspect][
                data['sentiment'][0:3]] += 1

    print(f"{'Component':<18} | "
          f"{'✅ Pos':>7} | "
          f"{'❌ Neg':>7} | "
          f"{'➡️ Neu':>7} | "
          f"{'Health':>8}")
    print("-" * 60)

    for aspect, scores in (
            sorted(aspect_scores.items())):
        pos = scores.get('pos', 0)
        neg = scores.get('neg', 0)
        neu = scores.get('neu', 0)
        total = pos + neg + neu

        if total == 0:
            continue

        health_pct = pos / total * 100
        health = (
            "🟢 Good" if health_pct >= 60 else
            "🟡 Fair" if health_pct >= 40 else
            "🔴 Poor")

        print(f"{aspect:<18} | "
              f"{pos:>7} | "
              f"{neg:>7} | "
              f"{neu:>7} | "
              f"{health}")

    print(f"\n💡 Product managers use this!")
    print(f"   Which component needs most attention?")
    print(f"   Sentiment trend over time → is it improving?")
    print(f"\n   This is exactly what AI Engineering Copilot")
    print(f"   (Phase 6) will build at scale! 🔥")


if __name__ == "__main__":
    demonstrate_aspect_sentiment()
    product_health_monitor()
