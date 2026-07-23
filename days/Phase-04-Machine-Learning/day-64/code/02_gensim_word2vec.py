"""
Day 64 — Word Embeddings
Topic: Gensim Word2Vec — Train + Use
Date: 21 July 2026
Author: Bala Ravi

Train Word2Vec on bug report corpus.
Find similar words, visualize embedding space.
"""
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️  Gensim not installed.")
    print("    Run: pip install gensim")


# Bug report corpus for training
BUG_CORPUS = [
    "production server crashed all users affected immediately",
    "database connection pool exhausted causing outage",
    "server down service unavailable all enterprise customers",
    "critical security vulnerability sql injection detected",
    "payment system broken revenue loss ongoing now",
    "memory leak causing server crashes every two hours",
    "ssl certificate expired site completely unreachable",
    "deployment pipeline broken cannot release new code",
    "race condition causing silent data loss production",
    "out of memory error crashing application servers",
    "login fails users special characters in password",
    "email notifications not being delivered to users",
    "export csv produces malformed output large datasets",
    "oauth google login failing thirty percent users",
    "mobile app crashes ios seventeen users affected",
    "api rate limiting not working correctly requests",
    "webhook delivery failing intermittently production",
    "bulk import fails silently thirty percent records",
    "search returns incorrect results unicode queries",
    "report generation timing out large datasets export",
    "date picker shows incorrect timezone booking form",
    "sort order resets after page refresh annoying",
    "copy button not working firefox browser only",
    "table column widths inconsistent display issue",
    "loading spinner stays visible after completion",
    "breadcrumb navigation missing mobile view",
    "settings page scroll resets on save button",
    "graph legend overlaps chart small screens",
    "keyboard shortcut conflicts browser default keys",
    "avatar image not updating immediately after upload",
    "typo error message settings billing page",
    "button label grammatically incorrect text issue",
    "footer copyright year outdated shows old year",
    "inconsistent capitalization navigation menu items",
    "tooltip text wraps awkwardly some resolutions",
    "empty state illustration looks outdated design",
    "minor spacing issue between form fields settings",
    "help text slightly misleading billing page users",
    "console warning deprecated prop component react",
    "color disabled button slightly off brand guidelines"
]


def preprocess_for_word2vec(
        corpus: list) -> list:
    """
    Tokenize corpus for Word2Vec training.

    Args:
        corpus: List of raw text documents

    Returns:
        List of token lists
    """
    tokenized = []
    for doc in corpus:
        # Lowercase + remove punctuation
        clean = re.sub(r'[^a-zA-Z\s]', '', doc.lower())
        tokens = [
            t for t in clean.split()
            if len(t) > 2]
        if tokens:
            tokenized.append(tokens)
    return tokenized


def train_word2vec(
        sentences: list) -> object:
    """
    Train Word2Vec model on sentences.

    Args:
        sentences: List of token lists

    Returns:
        Trained Word2Vec model
    """
    if not GENSIM_AVAILABLE:
        return None

    model = Word2Vec(
        sentences=sentences,
        vector_size=50,     # embedding dimension
        window=4,           # context window size
        min_count=1,        # min word frequency
        sg=1,               # 1=Skip-gram, 0=CBOW
        workers=4,
        epochs=100,
        seed=42)

    return model


def explore_embeddings(model) -> None:
    """Explore trained embeddings."""
    if model is None:
        print("Model not available — "
              "install gensim first!")
        return

    print("=== Word2Vec Exploration ===\n")

    wv = model.wv

    print(f"Vocabulary size: {len(wv)}")
    print(f"Embedding dim:   {wv.vector_size}\n")

    # Similar words
    test_words = [
        'crash', 'server', 'typo',
        'production', 'error']

    print("Most similar words:\n")
    for word in test_words:
        if word in wv:
            similar = wv.most_similar(
                word, topn=4)
            sim_str = ', '.join([
                f"{w}({s:.2f})"
                for w, s in similar])
            print(f"  {word:<15} → {sim_str}")

    # Similarity between pairs
    print(f"\nSimilarity between word pairs:")
    pairs = [
        ('crash', 'down'),
        ('crash', 'outage'),
        ('crash', 'typo'),
        ('error', 'fail'),
        ('typo', 'spacing'),
        ('production', 'server')
    ]

    print(f"\n{'Pair':<25} | {'Similarity':>11}")
    print("-" * 40)

    for w1, w2 in pairs:
        if w1 in wv and w2 in wv:
            sim = wv.similarity(w1, w2)
            bar = '█' * int(sim * 20)
            print(f"{w1+' vs '+w2:<25} | "
                  f"{sim:>11.4f} {bar}")


def sentence_embeddings_demo(
        model) -> None:
    """
    Create sentence embeddings by averaging
    word vectors.
    """
    print("\n=== Sentence Embeddings ===\n")
    print("Average word vectors = sentence vector")
    print("Simple but effective! 🔥\n")

    sentences = [
        "production server crashed down",
        "database went offline outage",
        "typo in footer cosmetic",
        "login fails authentication broken",
        "minor spacing visual issue"
    ]

    def sentence_vector(
            sentence: str,
            model) -> np.ndarray:
        """Average word vectors for sentence."""
        words = sentence.lower().split()
        vectors = []
        for word in words:
            if word in model.wv:
                vectors.append(
                    model.wv[word])
        if not vectors:
            return np.zeros(
                model.wv.vector_size)
        return np.mean(vectors, axis=0)

    if model is None:
        print("Gensim not available.")
        print("Showing concept only:\n")
        for sent in sentences:
            print(f"  '{sent}'")
            print(f"  → Average of word vectors")
            print(f"  → 50-dimensional vector\n")
        return

    from sklearn.metrics.pairwise import (
        cosine_similarity)

    sent_vecs = np.array([
        sentence_vector(s, model)
        for s in sentences])

    print("Similarity matrix:")
    print(f"{'':>30}", end='')
    for i in range(len(sentences)):
        print(f"S{i+1:>4}", end='')
    print()

    sim_matrix = cosine_similarity(sent_vecs)
    for i, sent in enumerate(sentences):
        print(f"{sent[:28]:>30}", end='')
        for j in range(len(sentences)):
            print(f"{sim_matrix[i][j]:>5.2f}", end='')
        print()

    print(f"\n💡 Similar bug reports → high similarity!")
    print(f"   S1 (crashed) vs S2 (offline) → ~0.75")
    print(f"   S1 (crashed) vs S3 (typo) → ~0.15")


def word2vec_for_bug_predictor() -> None:
    """
    How embeddings could improve Bug Predictor.
    """
    print("\n=== Word2Vec vs TF-IDF for Bug Predictor ===\n")

    print("Current Bug Predictor uses TF-IDF:")
    print("  'crashed' and 'went down' → unrelated ❌")
    print("  Misses synonym relationships\n")

    print("With Word2Vec embeddings:")
    print("  'crashed' and 'went down' → similar ✅")
    print("  'outage' and 'unavailable' → similar ✅")
    print("  Better generalization to new issue text!\n")

    print("Upgrade path for Bug Predictor v2:")
    print("  1. Replace TF-IDF with Word2Vec averages")
    print("  2. Or use sentence-transformers (Day 82)")
    print("  3. sentence-transformers >> Word2Vec")
    print("  4. That's what MemoryOS (Day 87) uses! 🔥")

    print("\nPerformance comparison (estimated):")
    print(f"  {'Method':<30} | {'F1 Score':>9}")
    print("-" * 45)
    print(f"  {'TF-IDF + Random Forest':<30} | {'0.8891':>9}")
    print(f"  {'Word2Vec avg + Random Forest':<30} | {'~0.91':>9}")
    print(f"  {'sentence-transformers + RF':<30} | {'~0.94':>9}")
    print(f"\n  Each generation of embeddings")
    print(f"  improves semantic understanding! 🚀")


if __name__ == "__main__":
    print("Training Word2Vec on bug corpus...\n")

    sentences = preprocess_for_word2vec(BUG_CORPUS)
    print(f"Corpus: {len(sentences)} documents")
    print(f"Unique words: "
          f"{len(set(w for s in sentences for w in s))}\n")

    model = train_word2vec(sentences)

    if model:
        print("✅ Word2Vec trained!\n")
        explore_embeddings(model)
        sentence_embeddings_demo(model)
    else:
        print("Running concept demo without Gensim:\n")
        from days.day_64.code.word2vec_concepts import (
            demonstrate_embedding_intuition)
        demonstrate_embedding_intuition()

    word2vec_for_bug_predictor()
