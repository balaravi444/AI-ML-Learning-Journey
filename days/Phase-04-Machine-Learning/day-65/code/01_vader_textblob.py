"""
Day 65 — Sentiment Analysis
Topic: VADER + TextBlob Rule-Based Sentiment
Date: 22 July 2026
Author: Bala Ravi

Rule-based sentiment — no training needed!
Works out of the box. Fast and interpretable.
"""
import re
import warnings
warnings.filterwarnings('ignore')

try:
    from vaderSentiment.vaderSentiment import (
        SentimentIntensityAnalyzer)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  vaderSentiment not installed.")
    print("    Run: pip install vaderSentiment\n")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not installed.")
    print("    Run: pip install textblob\n")


def get_vader_sentiment(text: str) -> dict:
    """
    Get VADER sentiment scores.

    VADER handles:
    - Capitalization (BROKEN > broken)
    - Punctuation (broken!!! > broken)
    - Degree modifiers (very broken)
    - Negation (not working)

    Args:
        text: Input text

    Returns:
        Dictionary with sentiment scores
    """
    if not VADER_AVAILABLE:
        # Simple fallback
        neg_words = {
            'crash', 'down', 'broken', 'failed',
            'error', 'critical', 'urgent', 'terrible',
            'awful', 'horrible', 'broken', 'disaster'}
        pos_words = {
            'fixed', 'resolved', 'working', 'great',
            'excellent', 'perfect', 'awesome'}

        words = set(text.lower().split())
        neg_count = len(words & neg_words)
        pos_count = len(words & pos_words)
        total = max(neg_count + pos_count, 1)

        compound = (pos_count - neg_count) / total
        return {
            'neg': neg_count / total,
            'pos': pos_count / total,
            'neu': 1 - abs(compound),
            'compound': compound,
            'label': (
                'NEGATIVE' if compound < -0.05 else
                'POSITIVE' if compound > 0.05 else
                'NEUTRAL')
        }

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)

    if scores['compound'] >= 0.05:
        label = 'POSITIVE'
    elif scores['compound'] <= -0.05:
        label = 'NEGATIVE'
    else:
        label = 'NEUTRAL'

    scores['label'] = label
    return scores


def demonstrate_vader() -> None:
    """Show VADER on various texts."""
    print("=== VADER Sentiment Analysis ===\n")

    texts = [
        # Different ways to report same bug
        ("PRODUCTION IS COMPLETELY DOWN!!!",
         "Critical — all caps + exclamation"),
        ("Production is completely down.",
         "Critical — calm tone"),
        ("production is down",
         "Critical — minimal emphasis"),
        ("Minor typo in the footer.",
         "Low — calm factual"),
        ("This is absolutely terrible!!! "
         "Nothing works!",
         "Frustrated user"),
        ("I found a small cosmetic issue "
         "that could be improved.",
         "Polite reporter"),
        ("Server not working properly "
         "for some users.",
         "Neutral report"),
        ("Everything is broken again. "
         "Third time this week!!!",
         "Very frustrated"),
    ]

    print(f"{'Text':<45} | "
          f"{'Compound':>9} | "
          f"{'Label':>9} | "
          f"{'Note'}")
    print("-" * 95)

    for text, note in texts:
        scores = get_vader_sentiment(text)
        print(f"{text[:43]:<45} | "
              f"{scores['compound']:>9.4f} | "
              f"{scores['label']:>9} | "
              f"{note}")

    print(f"\n💡 Key VADER behaviors:")
    print(f"   CAPS → stronger sentiment")
    print(f"   !!! → stronger sentiment")
    print(f"   'completely', 'absolutely' → amplifiers")
    print(f"   'not working' → negation handled ✅")


def vader_on_bug_reports() -> None:
    """Apply VADER to real bug reports."""
    print("\n=== VADER on Bug Reports ===\n")

    bug_reports = [
        {
            'title': 'PRODUCTION DOWN - ALL USERS LOCKED OUT!!!',
            'expected_priority': 'Critical'
        },
        {
            'title': 'Database connection pool exhausted causing outage',
            'expected_priority': 'Critical'
        },
        {
            'title': 'Login fails for users with special characters',
            'expected_priority': 'High'
        },
        {
            'title': 'Date picker shows wrong timezone',
            'expected_priority': 'Medium'
        },
        {
            'title': 'Typo in footer copyright year',
            'expected_priority': 'Low'
        },
        {
            'title': 'This is absolutely broken again!!! '
                     'Nothing ever works properly!!',
            'expected_priority': 'High'
        }
    ]

    print(f"{'Title':<50} | "
          f"{'Sentiment':>9} | "
          f"{'Priority'}")
    print("-" * 75)

    for bug in bug_reports:
        scores = get_vader_sentiment(bug['title'])
        print(f"{bug['title'][:48]:<50} | "
              f"{scores['compound']:>9.4f} | "
              f"{bug['expected_priority']}")

    print(f"\n💡 Sentiment + text features = better model!")
    print(f"   Bug Predictor v2: add vader_compound")
    print(f"   as a metadata feature → better F1!")


def textblob_demo() -> None:
    """TextBlob for polarity and subjectivity."""
    if not TEXTBLOB_AVAILABLE:
        print("\n=== TextBlob (not installed) ===")
        print("Run: pip install textblob")
        print("Showing expected behavior:\n")

        examples = [
            ("Server completely crashed horrible outage",
             -0.7, 0.8),
            ("Minor cosmetic issue in footer",
             0.0, 0.2),
            ("I found a possible improvement",
             0.2, 0.4)
        ]

        print(f"{'Text':<45} | "
              f"{'Polarity':>9} | "
              f"{'Subjectivity':>13}")
        print("-" * 72)

        for text, pol, sub in examples:
            print(f"{text:<45} | "
                  f"{pol:>9.2f} | "
                  f"{sub:>13.2f}")
        return

    print("\n=== TextBlob Sentiment ===\n")
    print("Polarity:     -1 (negative) to +1 (positive)")
    print("Subjectivity: 0 (factual) to 1 (opinion)\n")

    texts = [
        "The production server completely crashed!",
        "There is a minor issue with the button.",
        "This is absolutely terrible and broken!!!",
        "I found a possible improvement suggestion.",
        "Login is not working for some users.",
        "Amazing fix! Everything works perfectly now!"
    ]

    print(f"{'Text':<45} | "
          f"{'Polarity':>9} | "
          f"{'Subjectivity':>13} | "
          f"{'Label'}")
    print("-" * 80)

    for text in texts:
        blob = TextBlob(text)
        pol = blob.sentiment.polarity
        sub = blob.sentiment.subjectivity
        label = (
            'NEGATIVE' if pol < -0.1 else
            'POSITIVE' if pol > 0.1 else
            'NEUTRAL')
        print(f"{text[:43]:<45} | "
              f"{pol:>9.3f} | "
              f"{sub:>13.3f} | "
              f"{label}")

    print(f"\n💡 Subjectivity distinguishes:")
    print(f"   Factual reports (sub ~0.2) →")
    print(f"   likely more accurate bug descriptions")
    print(f"   Opinion-based (sub ~0.8) →")
    print(f"   may be exaggerating severity!")


if __name__ == "__main__":
    demonstrate_vader()
    vader_on_bug_reports()
    textblob_demo()
