"""
Day 45 — Statistics for ML
Topic: Bayes Theorem and Probability
Date: 02 July 2026
Author: Bala Ravi

Bayes Theorem = updating beliefs with evidence!
Foundation of Naive Bayes, Bayesian Neural Nets!
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def bayes_theorem_demo() -> None:
    """
    Demonstrate Bayes theorem with real examples.
    """
    print("=== Bayes Theorem ===\n")
    print("P(A|B) = P(B|A) × P(A) / P(B)\n")

    # Example: Spam detection
    print("📧 Spam Detection Example:")
    print("-" * 40)

    # Prior probabilities
    p_spam = 0.30    # 30% of emails are spam
    p_not_spam = 0.70

    # Likelihood: P(contains "free money" | class)
    p_free_given_spam = 0.80
    p_free_given_not_spam = 0.05

    # Evidence: P("free money" in any email)
    p_free = (p_free_given_spam * p_spam +
              p_free_given_not_spam * p_not_spam)

    # Posterior: P(spam | "free money")
    p_spam_given_free = (p_free_given_spam *
                          p_spam / p_free)

    print(f"P(spam):                 {p_spam:.2f}")
    print(f"P('free money'|spam):    {p_free_given_spam:.2f}")
    print(f"P('free money'|not spam):{p_free_given_not_spam:.2f}")
    print(f"\nP(spam | 'free money') = {p_spam_given_free:.4f}")
    print(f"= {p_spam_given_free*100:.1f}% chance of spam!")

    print(f"\n💡 Started with 30% prior probability")
    print(f"   After seeing 'free money' → "
          f"{p_spam_given_free*100:.0f}% probability!")
    print(f"   Evidence UPDATED our belief! 🔥")


def naive_bayes_classifier() -> None:
    """
    Naive Bayes classifier — Bayes Theorem in action!
    Used for text classification, spam detection!
    """
    print("\n=== Naive Bayes Classifier ===\n")

    np.random.seed(42)
    n = 500

    # Simulate email features
    # Feature 1: count of "free money" words
    # Feature 2: email length
    # Feature 3: number of exclamation marks

    # Spam emails
    spam_features = np.column_stack([
        np.random.poisson(3, n // 2),   # more spam words
        np.random.normal(100, 30, n // 2),  # shorter
        np.random.poisson(5, n // 2)    # more !!!
    ])

    # Not spam emails
    ham_features = np.column_stack([
        np.random.poisson(0.5, n // 2),  # fewer spam words
        np.random.normal(300, 80, n // 2),  # longer
        np.random.poisson(1, n // 2)     # fewer !!!
    ])

    X = np.vstack([spam_features, ham_features])
    y = np.array([1] * (n // 2) + [0] * (n // 2))

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Naive Bayes = Bayes Theorem!
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Naive Bayes Accuracy: {acc:.4f}")
    print(f"({acc*100:.1f}% of emails correctly classified!)")

    # Show probabilities
    sample = X_test[:3]
    proba = model.predict_proba(sample)
    print(f"\nSample predictions with probabilities:")
    print(f"{'Email':>6} | {'P(not spam)':>12} | "
          f"{'P(spam)':>8} | {'Prediction':>10}")
    print("-" * 45)
    for i, (p, pred) in enumerate(
            zip(proba, model.predict(sample))):
        label = "SPAM" if pred == 1 else "Not Spam"
        print(f"{i+1:>6} | {p[0]:>12.4f} | "
              f"{p[1]:>8.4f} | {label:>10}")


def clt_demonstration() -> None:
    """
    Central Limit Theorem demonstration.
    Why ML works on large datasets!
    """
    print("\n=== Central Limit Theorem ===\n")

    np.random.seed(42)

    # Original distribution — highly skewed (exponential)
    population = np.random.exponential(scale=2, size=10000)

    print(f"Original distribution (Exponential):")
    print(f"  Mean:     {population.mean():.3f}")
    print(f"  Skewness: {pd.Series(population).skew():.3f}")
    print(f"  → Very skewed! 📉")

    print(f"\nSample means distribution:")
    for n in [5, 30, 100, 500]:
        sample_means = [
            np.random.choice(
                population, n).mean()
            for _ in range(1000)
        ]
        skew = pd.Series(sample_means).skew()
        print(f"  n={n:>4}: mean={np.mean(sample_means):.3f}, "
              f"skew={skew:.3f} "
              f"{'✅ Normal!' if abs(skew) < 0.5 else '→ Getting there'}")

    print(f"\n💡 CLT says: With n≥30,")
    print(f"   sample means are approximately normal!")
    print(f"   This is why batch gradient descent works!")
    print(f"   Each batch mean ≈ true mean (CLT!) 🔥")


if __name__ == "__main__":
    bayes_theorem_demo()
    naive_bayes_classifier()
    clt_demonstration()
