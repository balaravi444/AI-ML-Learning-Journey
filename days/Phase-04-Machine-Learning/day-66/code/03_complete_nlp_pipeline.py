"""
Day 66 — Named Entity Recognition
Topic: Complete NLP Pipeline — Days 63-66 Combined
Date: 23 July 2026
Author: Bala Ravi

Combining everything from NLP week:
Text preprocessing (Day 63) +
TF-IDF / embeddings (Day 64) +
Sentiment (Day 65) +
NER (Day 66) = complete pipeline!
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')


class CompleteBugAnalyzer:
    """
    Complete NLP pipeline for bug reports.
    Combines preprocessing + TF-IDF +
    sentiment + NER → rich feature vector.

    This is what Bug Predictor v3 would look like!
    """

    def __init__(self) -> None:
        """Initialize all NLP components."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=500,
            sublinear_tf=True)

        # Tech entity patterns
        self.tech_patterns = {
            'has_service': re.compile(
                r'\b\w+(?:-\w+)*-(?:service|api|gateway)\b',
                re.IGNORECASE),
            'has_version': re.compile(
                r'v?\d+\.\d+(?:\.\d+)?',
                re.IGNORECASE),
            'has_error_code': re.compile(
                r'(?:HTTP|Error|FATAL|Status)\s*:?\s*\d+',
                re.IGNORECASE),
            'is_production': re.compile(
                r'\b(?:production|prod)\b',
                re.IGNORECASE),
            'has_user_count': re.compile(
                r'\d+(?:,\d{3})*\s*(?:users|customers)',
                re.IGNORECASE),
            'has_revenue': re.compile(
                r'(?:\$|₹)\s*\d+',
                re.IGNORECASE)
        }

        # Sentiment word lists
        self.neg_words = {
            'crash', 'down', 'broken', 'failed',
            'error', 'fatal', 'critical', 'urgent',
            'terrible', 'awful', 'disaster', 'outage',
            'completely', 'totally', 'nothing'}
        self.pos_words = {
            'fixed', 'resolved', 'working', 'great',
            'improved', 'better', 'correct', 'minor',
            'small', 'cosmetic', 'typo', 'convenient'}

    def _sentiment_score(self, text: str) -> float:
        """Simple sentiment score -1 to +1."""
        words = set(text.lower().split())
        neg = len(words & self.neg_words)
        pos = len(words & self.pos_words)
        total = max(neg + pos, 1)
        # Amplify for caps
        if text != text.lower():
            neg *= 1.3
        return (pos - neg) / total

    def _urgency_signals(self, text: str) -> dict:
        """Extract urgency signals via NER + rules."""
        text_lower = text.lower()
        signals = {}

        # NER features
        for feat, pattern in (
                self.tech_patterns.items()):
            signals[feat] = int(
                bool(pattern.search(text)))

        # Emphasis signals
        signals['has_caps_words'] = int(
            any(w.isupper() and len(w) > 2
                for w in text.split()))
        signals['exclamation_count'] = min(
            text.count('!'), 5)
        signals['sentiment_score'] = (
            self._sentiment_score(text))

        # Time urgency
        signals['mentions_time'] = int(
            bool(re.search(
                r'\b(?:now|immediately|urgent|'
                r'asap|since|hours?\s+ago)\b',
                text_lower)))

        return signals

    def fit(self,
             texts: list,
             labels: list) -> 'CompleteBugAnalyzer':
        """Fit the complete pipeline."""
        # TF-IDF features
        tfidf = self.vectorizer.fit_transform(texts)

        # Urgency signal features
        signals = np.array([
            list(self._urgency_signals(t).values())
            for t in texts])

        self.scaler = StandardScaler()
        signals_scaled = self.scaler.fit_transform(
            signals)
        signals_sparse = sp.csr_matrix(signals_scaled)

        X = sp.hstack([tfidf, signals_sparse])

        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42)
        self.model.fit(X, labels)
        self.classes_ = self.model.classes_

        return self

    def predict(self, texts: list) -> list:
        """Predict priority for texts."""
        tfidf = self.vectorizer.transform(texts)
        signals = np.array([
            list(self._urgency_signals(t).values())
            for t in texts])
        signals_scaled = self.scaler.transform(signals)
        signals_sparse = sp.csr_matrix(signals_scaled)
        X = sp.hstack([tfidf, signals_sparse])

        return list(self.model.predict(X))

    def analyze(self, text: str) -> dict:
        """Full analysis of a single bug report."""
        prediction = self.predict([text])[0]
        signals = self._urgency_signals(text)

        # Entity extraction
        entities = {}
        for feat, pattern in (
                self.tech_patterns.items()):
            match = pattern.search(text)
            if match:
                entities[feat.replace(
                    'has_', '').replace(
                    'is_', '')] = match.group()

        return {
            'predicted_priority': prediction,
            'sentiment_score': round(
                signals['sentiment_score'], 3),
            'urgency_signals': {
                k: v for k, v in signals.items()
                if isinstance(v, int) and v > 0},
            'entities_found': entities,
            'text_preview': text[:80] + '...'
        }


def demonstrate_complete_pipeline() -> None:
    """Show complete NLP pipeline."""
    print("=== Complete NLP Pipeline (Days 63-66) ===\n")

    training_data = [
        ("PRODUCTION DOWN ALL USERS AFFECTED "
         "auth-service FATAL error PostgreSQL",
         'Critical'),
        ("Database connection pool exhausted "
         "causing complete outage v2.3.1",
         'Critical'),
        ("HTTP 503 errors api-gateway production "
         "50000 users cannot login now",
         'Critical'),
        ("Login fails special characters "
         "password authentication broken",
         'High'),
        ("Email notifications not delivered "
         "notification-service error",
         'High'),
        ("CSV export malformed output "
         "data pipeline failing",
         'High'),
        ("Date picker wrong timezone "
         "display issue booking form",
         'Medium'),
        ("Sort order resets page refresh "
         "table view inconsistent",
         'Medium'),
        ("Minor spacing issue form fields "
         "cosmetic settings page",
         'Low'),
        ("Typo footer copyright year "
         "outdated small fix",
         'Low'),
    ]

    texts = [d[0] for d in training_data]
    labels = [d[1] for d in training_data]

    analyzer = CompleteBugAnalyzer()
    analyzer.fit(texts, labels)
    print("✅ Pipeline trained!\n")

    # Test on new bugs
    test_bugs = [
        "auth-service v2.3.1 FATAL error "
        "production AWS PostgreSQL COMPLETELY DOWN!!!",
        "payment-service HTTP 503 staging "
        "users cannot checkout",
        "button tooltip disappears minor cosmetic",
        "EVERYTHING BROKEN AGAIN third time "
        "this week revenue loss ongoing"
    ]

    print("Full Analysis of New Bug Reports:\n")
    for bug in test_bugs:
        result = analyzer.analyze(bug)
        print(f"Bug: {result['text_preview']}")
        print(f"  Priority:  {result['predicted_priority']}")
        print(f"  Sentiment: {result['sentiment_score']}")
        print(f"  Urgency:   "
              f"{result['urgency_signals']}")
        print(f"  Entities:  "
              f"{result['entities_found']}")
        print()

    print("💡 NLP Week Summary (Days 63-66):")
    print("  Day 63: Text preprocessing + TF-IDF")
    print("  Day 64: Word embeddings + semantic search")
    print("  Day 65: Sentiment analysis")
    print("  Day 66: NER + entity features")
    print("\n  All combined → richer bug analysis!")
    print("  Next: Autonomous Data Scientist project! 🚀")


if __name__ == "__main__":
    demonstrate_complete_pipeline()
    print()

    # Quick NER demo without spaCy
    from days_day_66_code_02_regex_ner import (
        TechEntityExtractor,
        demonstrate_extraction)
    demonstrate_extraction()
