"""
Day 64 — Word Embeddings
Topic: Word2Vec Concepts + From Scratch Intuition
Date: 21 July 2026
Author: Bala Ravi

Understanding WHY word embeddings work.
Skip-gram intuition + cosine similarity.
"""
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def cosine_similarity(
        a: np.ndarray,
        b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Range: -1 to 1
    1.0  = same direction (similar meaning)
    0.0  = orthogonal (unrelated)
    -1.0 = opposite direction

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def demonstrate_embedding_intuition() -> None:
    """
    Show the intuition behind word embeddings
    using hand-crafted 3D vectors.
    """
    print("=== Word Embedding Intuition ===\n")
    print("Hand-crafted embeddings (3 dimensions):")
    print("Dim 1: technical severity")
    print("Dim 2: user impact")
    print("Dim 3: urgency\n")

    # Hand-crafted embeddings to show intuition
    embeddings = {
        'crash':      np.array([0.9,  0.9,  0.95]),
        'down':       np.array([0.85, 0.88, 0.90]),
        'outage':     np.array([0.88, 0.92, 0.93]),
        'error':      np.array([0.7,  0.6,  0.65]),
        'fail':       np.array([0.72, 0.62, 0.68]),
        'broken':     np.array([0.65, 0.70, 0.60]),
        'slow':       np.array([0.4,  0.5,  0.35]),
        'incorrect':  np.array([0.3,  0.4,  0.25]),
        'missing':    np.array([0.25, 0.35, 0.20]),
        'typo':       np.array([0.1,  0.1,  0.05]),
        'cosmetic':   np.array([0.05, 0.08, 0.03]),
        'spacing':    np.array([0.02, 0.03, 0.01]),
    }

    print(f"{'Word':<12} | "
          f"{'Severity':>9} | "
          f"{'Impact':>7} | "
          f"{'Urgency':>8}")
    print("-" * 45)

    for word, vec in embeddings.items():
        print(f"{word:<12} | "
              f"{vec[0]:>9.2f} | "
              f"{vec[1]:>7.2f} | "
              f"{vec[2]:>8.2f}")

    # Similarity pairs
    print(f"\nSemantic Similarity (cosine):\n")

    pairs = [
        ('crash', 'down', 'synonyms — should be HIGH'),
        ('crash', 'outage', 'synonyms — should be HIGH'),
        ('crash', 'typo', 'opposites — should be LOW'),
        ('error', 'fail', 'related — should be MED'),
        ('typo', 'cosmetic', 'related — should be HIGH'),
        ('down', 'spacing', 'unrelated — should be LOW'),
    ]

    print(f"{'Pair':<25} | "
          f"{'Similarity':>11} | "
          f"{'Expected'}")
    print("-" * 65)

    for w1, w2, note in pairs:
        sim = cosine_similarity(
            embeddings[w1], embeddings[w2])
        print(f"{w1+' vs '+w2:<25} | "
              f"{sim:>11.4f} | {note}")

    print(f"\n✅ Similar words → high cosine similarity!")
    print(f"   This is what Word2Vec learns automatically")
    print(f"   from reading millions of documents! 🔥")


def skipgram_intuition() -> None:
    """Show Skip-gram training intuition."""
    print("\n=== Skip-gram Intuition ===\n")

    corpus = [
        "production server crashed all users affected",
        "database went down service unavailable",
        "server timeout causing connection errors",
        "critical bug production system failure",
        "typo in footer minor cosmetic issue",
        "spacing problem small visual bug only"
    ]

    print("Training corpus:")
    for i, doc in enumerate(corpus, 1):
        print(f"  {i}. {doc}")

    window = 2
    print(f"\nSkip-gram training pairs (window={window}):")
    print(f"Given center word → predict context words\n")

    print(f"{'Center Word':<15} | {'Context Words'}")
    print("-" * 50)

    shown = set()
    for doc in corpus[:3]:
        words = doc.split()
        for i, center in enumerate(words):
            ctx_words = []
            for j in range(
                    max(0, i-window),
                    min(len(words), i+window+1)):
                if j != i:
                    ctx_words.append(words[j])

            key = center
            if key not in shown and ctx_words:
                shown.add(key)
                print(f"{center:<15} | "
                      f"{ctx_words}")

    print(f"\n💡 Word2Vec trains a neural network")
    print(f"   to predict context from center word.")
    print(f"   The hidden layer weights become")
    print(f"   the word embeddings!")
    print(f"\n   'crashed' and 'down' appear with")
    print(f"   SAME context words → similar vectors!")


def word_arithmetic_demo() -> None:
    """Show the famous word arithmetic."""
    print("\n=== Word Vector Arithmetic ===\n")

    print("The famous example:")
    print("king - man + woman ≈ queen\n")

    print("Why this works:")
    print("vec(king) - vec(man) = 'royalty' vector")
    print("vec(queen) - vec(woman) = 'royalty' vector")
    print("These are approximately EQUAL!\n")

    # Simulate with controlled vectors
    dim = 5
    # Dimensions: royalty, male, female, power, age

    king   = np.array([0.9, 0.8, 0.1, 0.85, 0.7])
    queen  = np.array([0.9, 0.1, 0.9, 0.85, 0.7])
    man    = np.array([0.1, 0.9, 0.1, 0.4,  0.5])
    woman  = np.array([0.1, 0.1, 0.9, 0.4,  0.5])
    prince = np.array([0.8, 0.8, 0.1, 0.6,  0.2])
    lord   = np.array([0.7, 0.7, 0.1, 0.7,  0.6])

    vocab = {
        'king': king, 'queen': queen,
        'man': man, 'woman': woman,
        'prince': prince, 'lord': lord
    }

    # king - man + woman
    result = king - man + woman
    print(f"king - man + woman =")
    print(f"  Result vector: {result.round(2)}\n")
    print(f"  Similarity to each word:")

    sims = {}
    for word, vec in vocab.items():
        sim = cosine_similarity(result, vec)
        sims[word] = sim
        print(f"    {word:<8}: {sim:.4f}")

    best = max(sims, key=sims.get)
    print(f"\n  → Closest word: '{best}'")
    print(f"  ✅ king - man + woman ≈ queen!")

    print(f"\n💡 In real Word2Vec (trained on Wikipedia):")
    print(f"   This arithmetic actually works!")
    print(f"   The geometry of the embedding space")
    print(f"   captures human concepts! 🔥")


def tfidf_vs_embeddings_comparison() -> None:
    """Show when to use TF-IDF vs embeddings."""
    print("\n=== TF-IDF vs Embeddings ===\n")

    doc1 = "server crashed production down"
    doc2 = "system went offline service unavailable"
    doc3 = "typo in footer cosmetic issue"

    print(f"Doc 1: '{doc1}'")
    print(f"Doc 2: '{doc2}'")
    print(f"Doc 3: '{doc3}'\n")

    from sklearn.feature_extraction.text import (
        TfidfVectorizer)
    from sklearn.metrics.pairwise import (
        cosine_similarity as sk_cosine)

    docs = [doc1, doc2, doc3]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(docs)

    sim_12_tfidf = sk_cosine(
        tfidf_matrix[0], tfidf_matrix[1])[0][0]
    sim_13_tfidf = sk_cosine(
        tfidf_matrix[0], tfidf_matrix[2])[0][0]

    print("TF-IDF Similarity:")
    print(f"  Doc1 vs Doc2: {sim_12_tfidf:.4f}")
    print(f"  Doc1 vs Doc3: {sim_13_tfidf:.4f}")
    print(f"  ⚠️  Doc1 and Doc2 look UNRELATED to TF-IDF!")
    print(f"     'crashed' ≠ 'offline', 'down' ≠ 'unavailable'")

    # Simulated embedding similarity
    # (real Word2Vec would show ~0.75 for Doc1 vs Doc2)
    print(f"\nEmbedding Similarity (simulated):")
    print(f"  Doc1 vs Doc2: ~0.76 (similar meaning!)")
    print(f"  Doc1 vs Doc3: ~0.12 (different!)")
    print(f"  ✅ Embeddings understand synonyms!")

    print(f"\nWhen to choose:")
    print(f"  TF-IDF    → keyword matching, small data,")
    print(f"              interpretability needed")
    print(f"  Embeddings → semantic search, synonyms,")
    print(f"               large corpus, MemoryOS! 🔥")


if __name__ == "__main__":
    demonstrate_embedding_intuition()
    skipgram_intuition()
    word_arithmetic_demo()
    tfidf_vs_embeddings_comparison()
