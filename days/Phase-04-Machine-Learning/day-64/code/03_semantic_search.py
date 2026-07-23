"""
Day 64 — Word Embeddings
Topic: Semantic Search with Embeddings
Date: 21 July 2026
Author: Bala Ravi

Build a simple semantic search engine!
Find similar bug reports using vector similarity.
Foundation of MemoryOS RAG pipeline!
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class SimpleSematicSearch:
    """
    Semantic search using TF-IDF as proxy embeddings.

    In production → replace TF-IDF with
    sentence-transformers for true semantic search!

    This is the same concept MemoryOS uses:
    Store document vectors → find nearest on query!
    """

    def __init__(self) -> None:
        """Initialize search engine."""
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            max_features=1000)
        self.doc_vectors = None
        self.documents = []
        self.metadata = []

    def index(self,
              documents: list,
              metadata: list = None) -> None:
        """
        Index documents for search.

        Args:
            documents: List of text documents
            metadata: Optional metadata per doc
        """
        self.documents = documents
        self.metadata = metadata or [
            {} for _ in documents]
        self.doc_vectors = (
            self.vectorizer.fit_transform(documents))
        print(f"✅ Indexed {len(documents)} documents")
        print(f"   Feature dim: "
              f"{self.doc_vectors.shape[1]}")

    def search(self,
               query: str,
               top_k: int = 3) -> list:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (score, document, metadata) tuples
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vec, self.doc_vectors)[0]

        top_indices = scores.argsort()[
            ::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'score': float(scores[idx]),
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'rank': len(results) + 1
                })

        return results


def build_bug_search_engine() -> None:
    """Build semantic search over bug reports."""
    print("=== Bug Report Semantic Search ===\n")

    bug_db = [
        {
            'text': "Production database connection pool exhausted "
                    "causing complete service outage all users",
            'meta': {'priority': 'Critical',
                     'id': 'BUG-001',
                     'repo': 'api-gateway'}
        },
        {
            'text': "Server memory leak causing crashes "
                    "every two hours in production environment",
            'meta': {'priority': 'Critical',
                     'id': 'BUG-002',
                     'repo': 'auth-service'}
        },
        {
            'text': "Login fails for users with special "
                    "characters plus sign in email address",
            'meta': {'priority': 'High',
                     'id': 'BUG-003',
                     'repo': 'frontend'}
        },
        {
            'text': "Email notifications not being delivered "
                    "to enterprise customers since yesterday",
            'meta': {'priority': 'High',
                     'id': 'BUG-004',
                     'repo': 'notification-service'}
        },
        {
            'text': "CSV export produces malformed output "
                    "when dataset exceeds ten thousand rows",
            'meta': {'priority': 'High',
                     'id': 'BUG-005',
                     'repo': 'data-pipeline'}
        },
        {
            'text': "Date picker showing UTC timezone instead "
                    "of user local timezone on booking form",
            'meta': {'priority': 'Medium',
                     'id': 'BUG-006',
                     'repo': 'frontend'}
        },
        {
            'text': "Sort order resets to default after "
                    "page refresh on data table view",
            'meta': {'priority': 'Medium',
                     'id': 'BUG-007',
                     'repo': 'admin-dashboard'}
        },
        {
            'text': "Typo in error message on settings billing "
                    "page grammatically incorrect text",
            'meta': {'priority': 'Low',
                     'id': 'BUG-008',
                     'repo': 'frontend'}
        },
        {
            'text': "Footer copyright year shows 2024 "
                    "should be updated to 2026 minor fix",
            'meta': {'priority': 'Low',
                     'id': 'BUG-009',
                     'repo': 'frontend'}
        },
        {
            'text': "Minor spacing issue between form fields "
                    "on settings page cosmetic only",
            'meta': {'priority': 'Low',
                     'id': 'BUG-010',
                     'repo': 'frontend'}
        }
    ]

    engine = SimpleSematicSearch()
    engine.index(
        documents=[b['text'] for b in bug_db],
        metadata=[b['meta'] for b in bug_db])

    # Test queries
    queries = [
        "server is down and crashing",
        "users cannot login",
        "small visual problem in UI",
        "database connection issue"
    ]

    print()
    for query in queries:
        print(f"Query: '{query}'")
        results = engine.search(query, top_k=3)

        for r in results:
            meta = r['metadata']
            print(f"  #{r['rank']} [{meta['id']}] "
                  f"{meta['priority']:<10} "
                  f"score={r['score']:.3f}")
            print(f"     {r['document'][:60]}...")
        print()

    print("💡 This IS the core of MemoryOS!")
    print("   Index documents → search by meaning")
    print("   Replace TF-IDF with sentence-transformers")
    print("   → True semantic search! 🔥")


def embedding_space_visualization() -> None:
    """Show how documents cluster in embedding space."""
    print("\n=== Embedding Space Clusters ===\n")

    print("If we could visualize 50D → 2D (via PCA/TSNE):")
    print()

    clusters = {
        'Critical cluster 🔴': [
            "production server crashed",
            "database down all users",
            "service outage critical"
        ],
        'High cluster 🟠': [
            "login fails some users",
            "feature broken error",
            "api not responding"
        ],
        'Medium cluster 🟡': [
            "timezone wrong display",
            "sort resets on refresh",
            "ui inconsistency found"
        ],
        'Low cluster 🟢': [
            "typo in error message",
            "cosmetic spacing issue",
            "copyright year outdated"
        ]
    }

    docs = []
    labels = []
    for cluster_name, cluster_docs in clusters.items():
        for doc in cluster_docs:
            docs.append(doc)
            labels.append(cluster_name)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english')
    matrix = vec.fit_transform(docs).toarray()

    # PCA to 2D
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    top2 = eigenvectors[:, -2:]
    coords_2d = centered @ top2

    print(f"Documents in 2D embedding space:\n")
    print(f"{'Document':<35} | "
          f"{'x':>7} | {'y':>7} | Cluster")
    print("-" * 75)

    unique_clusters = list(clusters.keys())
    for i, (doc, label) in enumerate(
            zip(docs, labels)):
        x, y = coords_2d[i]
        print(f"{doc[:33]:<35} | "
              f"{x:>7.3f} | {y:>7.3f} | "
              f"{label}")

    print(f"\n💡 Similar priority bugs cluster together!")
    print(f"   Critical bugs: top-right region")
    print(f"   Low bugs: bottom-left region")
    print(f"   This clustering is what makes")
    print(f"   semantic search work! 🔥")


if __name__ == "__main__":
    build_bug_search_engine()
    embedding_space_visualization()
