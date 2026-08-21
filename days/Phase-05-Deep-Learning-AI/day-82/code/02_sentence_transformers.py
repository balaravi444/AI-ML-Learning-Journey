"""
Day 82 — HuggingFace: Pretrained Models
Topic: sentence-transformers for Semantic Search
Date: 08 August 2026
Author: Bala Ravi

sentence-transformers = BERT for sentences!
Converts sentences → dense vectors.
Semantic similarity → foundation of MemoryOS!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import (
        SentenceTransformer)
    from sklearn.metrics.pairwise import (
        cosine_similarity)
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("⚠️  sentence-transformers not installed.")
    print("    Run: pip install sentence-transformers\n")


def demonstrate_embeddings() -> None:
    """Show sentence embeddings in action."""
    print("=== Sentence Embeddings Demo ===\n")

    if not ST_AVAILABLE:
        print("sentence-transformers not available.")
        print("\nExpected output:\n")

        print("Model: all-MiniLM-L6-v2")
        print("Embedding dimension: 384\n")

        sentences = [
            "server crashed in production",
            "system went offline unexpectedly",
            "typo in footer",
            "login fails for some users",
            "database connection exhausted"
        ]

        # Simulated similarities
        sim_matrix = np.array([
            [1.00, 0.87, 0.12, 0.41, 0.76],
            [0.87, 1.00, 0.09, 0.38, 0.72],
            [0.12, 0.09, 1.00, 0.15, 0.11],
            [0.41, 0.38, 0.15, 1.00, 0.44],
            [0.76, 0.72, 0.11, 0.44, 1.00]
        ])

        print("Cosine similarity matrix:\n")
        print(f"{'':>35}", end='')
        for s in sentences:
            print(f"{s[:8]:>10}", end='')
        print()

        for i, s1 in enumerate(sentences):
            print(f"{s1[:33]:>35}", end='')
            for j in range(len(sentences)):
                val = sim_matrix[i][j]
                print(f"{val:>10.3f}", end='')
            print()

        print(f"\n💡 crash ≈ offline (0.87) → same meaning!")
        print(f"   crash ≠ typo (0.12) → different!")
        print(f"   TF-IDF would give 0.0 for both!")
        return

    print("Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences = [
        "server crashed in production",
        "system went offline unexpectedly",
        "typo in the footer",
        "login fails for some users",
        "database connection exhausted"
    ]

    print(f"Model loaded!")
    embeddings = model.encode(sentences)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"(sentences × embedding_dim)\n")

    # Similarity matrix
    sims = cosine_similarity(embeddings)

    print("Cosine Similarity Matrix:\n")
    abbrevs = [s[:10] for s in sentences]
    print(f"{'':>30}", end='')
    for ab in abbrevs:
        print(f"{ab:>12}", end='')
    print()

    for i, s in enumerate(sentences):
        print(f"{s[:28]:>30}", end='')
        for j in range(len(sentences)):
            val = sims[i][j]
            print(f"{val:>12.3f}", end='')
        print()

    print(f"\n✅ 'crash' ≈ 'offline' (high similarity)")
    print(f"   'crash' ≠ 'typo' (low similarity)")
    print(f"   semantic understanding! 🔥")


def semantic_search_demo() -> None:
    """Build simple semantic search engine."""
    print("\n=== Semantic Search Engine ===\n")
    print("Same as MemoryOS core! 🔥\n")

    if not ST_AVAILABLE:
        print("sentence-transformers not available.")
        print("\nSemantic search concept:\n")

        knowledge_base = [
            "RAG stands for Retrieval Augmented Generation",
            "Vector databases store embeddings for fast search",
            "BERT is a bidirectional transformer model",
            "Melanoma is the deadliest form of skin cancer",
            "Transfer learning reuses pretrained model weights",
            "LangChain connects LLMs with external tools"
        ]

        queries = [
            "How does retrieval augmented generation work?",
            "What is the most dangerous skin disease?",
            "How to use pretrained models?"
        ]

        expected = [
            ("RAG stands for Retrieval...", 0.89),
            ("Melanoma is the deadliest...", 0.91),
            ("Transfer learning reuses...", 0.84)
        ]

        for query, (doc, score) in zip(
                queries, expected):
            print(f"Query: '{query}'")
            print(f"  → Found: '{doc[:45]}...'")
            print(f"     Score: {score:.3f}\n")

        print("💡 This is exactly what")
        print("   MemoryOS does at scale! 🔥")
        return

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Knowledge base (what user has learned)
    knowledge_base = [
        "RAG stands for Retrieval Augmented "
        "Generation. It combines retrieval with LLMs.",
        "Vector databases store embeddings for "
        "fast semantic similarity search.",
        "BERT is a bidirectional transformer model "
        "pretrained on masked language modeling.",
        "Melanoma is the deadliest form of skin "
        "cancer with 15% late-stage survival rate.",
        "Transfer learning reuses pretrained model "
        "weights for new tasks with small datasets.",
        "LangChain connects LLMs with external tools "
        "like databases, APIs, and search engines.",
        "ChromaDB is an open-source vector database "
        "that runs locally without cloud setup.",
        "MobileNetV2 is a lightweight CNN optimized "
        "for mobile deployment with 3.4M parameters."
    ]

    # Encode knowledge base
    kb_embeddings = model.encode(knowledge_base)

    queries = [
        "How does retrieval augmented generation work?",
        "What is the most dangerous skin disease?",
        "How to use pretrained models on small data?",
        "What database stores vectors?"
    ]

    print("Semantic search over knowledge base:\n")

    for query in queries:
        query_emb = model.encode([query])
        scores = cosine_similarity(
            query_emb, kb_embeddings)[0]
        top_idx = scores.argsort()[::-1][:2]

        print(f"Query: '{query}'")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  #{rank}: {knowledge_base[idx][:60]}...")
            print(f"       Score: {scores[idx]:.4f}")
        print()

    print("✅ Finds semantically similar docs!")
    print("   Even with different wording!")
    print("   This is the heart of MemoryOS! 🔥")


if __name__ == "__main__":
    demonstrate_embeddings()
    semantic_search_demo()
