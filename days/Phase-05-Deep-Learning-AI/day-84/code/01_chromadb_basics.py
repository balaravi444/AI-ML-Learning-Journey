"""
Day 84 — Vector Databases + Embeddings
Topic: ChromaDB — Local Vector Store
Date: 10 August 2026
Author: Bala Ravi

ChromaDB = the memory of MemoryOS!
Store embeddings → search by meaning → retrieve!
"""
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️  chromadb not installed.")
    print("    Run: pip install chromadb\n")


def demonstrate_chromadb_basics() -> None:
    """Show ChromaDB CRUD operations."""
    print("=== ChromaDB Basics ===\n")

    if not CHROMA_AVAILABLE:
        print("ChromaDB not available.")
        print("\nChromaDB concepts:\n")

        code = '''
import chromadb

# 1. Create client (in-memory)
client = chromadb.Client()

# 2. Create collection
collection = client.create_collection(
    name="my_knowledge_base")

# 3. Add documents
collection.add(
    documents=[
        "RAG combines retrieval with generation",
        "ChromaDB stores vector embeddings",
        "MobileNetV2 is lightweight CNN"
    ],
    metadatas=[
        {"source": "day-85", "topic": "AI"},
        {"source": "day-84", "topic": "vectors"},
        {"source": "day-73", "topic": "CNN"}
    ],
    ids=["doc_1", "doc_2", "doc_3"]
)

# 4. Query semantically!
results = collection.query(
    query_texts=["how does vector search work?"],
    n_results=2)

for doc, meta in zip(
        results['documents'][0],
        results['metadatas'][0]):
    print(f"Found: {doc}")
    print(f"Source: {meta['source']}")

# Output:
# Found: ChromaDB stores vector embeddings
# Source: day-84
# Found: RAG combines retrieval with generation
# Source: day-85
        '''
        print(code)
        return

    # Create in-memory client
    client = chromadb.Client()

    # Use sentence-transformers for embeddings
    try:
        ef = (embedding_functions
              .SentenceTransformerEmbeddingFunction(
                  model_name="all-MiniLM-L6-v2"))
    except Exception:
        ef = None  # use default

    # Create collection
    collection = client.create_collection(
        name="learning_journey",
        embedding_function=ef)

    # Add documents from learning journey
    documents = [
        "RAG stands for Retrieval Augmented "
        "Generation, combining retrieval with LLMs",
        "ChromaDB is an open-source vector database "
        "that runs locally without cloud setup",
        "MobileNetV2 is a lightweight CNN optimized "
        "for mobile with 3.4M parameters",
        "Transfer learning reuses pretrained model "
        "weights for new tasks with small datasets",
        "BERT is a bidirectional transformer that "
        "learns from both left and right context",
        "Attention mechanism allows transformers to "
        "attend to relevant tokens in the sequence",
        "sentence-transformers converts text to "
        "dense vectors preserving semantic meaning",
        "SMOTE oversamples minority class by creating "
        "synthetic examples between real ones"
    ]

    metadatas = [
        {"day": 85, "topic": "RAG", "phase": 5},
        {"day": 84, "topic": "VectorDB", "phase": 5},
        {"day": 73, "topic": "CNN", "phase": 5},
        {"day": 74, "topic": "TransferLearning",
         "phase": 5},
        {"day": 81, "topic": "Transformers",
         "phase": 5},
        {"day": 81, "topic": "Attention", "phase": 5},
        {"day": 82, "topic": "HuggingFace",
         "phase": 5},
        {"day": 60, "topic": "SMOTE", "phase": 4}
    ]

    ids = [f"doc_{i:03d}"
            for i in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids)

    print(f"✅ Added {len(documents)} documents")
    print(f"   Collection: {collection.name}")
    print(f"   Count: {collection.count()}\n")

    # Query
    queries = [
        "how does retrieval augmented generation work?",
        "lightweight models for mobile devices",
        "dealing with imbalanced datasets"
    ]

    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=2)

        print(f"Query: '{query}'")
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results.get(
            'distances', [[0, 0]])[0]

        for i, (doc, meta, dist) in enumerate(
                zip(docs, metas, distances)):
            similarity = 1 - dist
            print(f"  #{i+1} [Day {meta['day']}] "
                  f"sim={similarity:.3f}")
            print(f"       {doc[:60]}...")
        print()


def metadata_filtering_demo() -> None:
    """Show filtering + semantic search combo."""
    print("=== Metadata Filtering ===\n")
    print("Combine semantic search + attribute filters!\n")

    if not CHROMA_AVAILABLE:
        print("ChromaDB not available.")
        print("\nMetadata filtering concept:\n")

        code = '''
# Filter by metadata + semantic search!

# Only search Phase 5 documents:
results = collection.query(
    query_texts=["neural network architecture"],
    n_results=3,
    where={"phase": 5}  # filter!
)

# Only search specific day range:
results = collection.query(
    query_texts=["deep learning"],
    n_results=3,
    where={"$and": [
        {"day": {"$gte": 71}},
        {"day": {"$lte": 85}}
    ]}
)

# This is MemoryOS "search by date range"!
# "What did I learn last week?"
# → filter by date, then semantic search! 🔥
        '''
        print(code)
        return

    client = chromadb.Client()
    collection = client.create_collection(
        name="filtered_demo")

    documents = [
        "CNN processes images with convolution",
        "Transformers use self-attention",
        "Random Forest is ensemble learning",
        "BERT is bidirectional transformer",
        "TF-IDF represents text as sparse vectors"
    ]

    metadatas = [
        {"phase": 5, "day": 73, "type": "vision"},
        {"phase": 5, "day": 81, "type": "nlp"},
        {"phase": 4, "day": 55, "type": "ml"},
        {"phase": 5, "day": 82, "type": "nlp"},
        {"phase": 4, "day": 63, "type": "nlp"}
    ]

    ids = [f"d_{i}" for i in range(len(documents))]
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids)

    # Unfiltered search
    r1 = collection.query(
        query_texts=["language model"],
        n_results=2)
    print("Unfiltered (all phases):")
    for doc, meta in zip(
            r1['documents'][0],
            r1['metadatas'][0]):
        print(f"  [Phase {meta['phase']}, "
              f"Day {meta['day']}] {doc[:50]}")

    # Filtered: Phase 5 only
    r2 = collection.query(
        query_texts=["language model"],
        n_results=2,
        where={"phase": 5})
    print("\nFiltered (Phase 5 only):")
    for doc, meta in zip(
            r2['documents'][0],
            r2['metadatas'][0]):
        print(f"  [Phase {meta['phase']}, "
              f"Day {meta['day']}] {doc[:50]}")

    print(f"\n✅ Metadata filtering works!")
    print(f"   MemoryOS uses this for:")
    print(f"   'What did I learn in Phase 5?'")
    print(f"   'Show me AI documents from last week!'")


def persistent_storage_demo() -> None:
    """Show persistent ChromaDB."""
    print("\n=== Persistent ChromaDB ===\n")
    print("Data survives application restarts!\n")

    if not CHROMA_AVAILABLE:
        print("ChromaDB not available.")
        print("\nPersistent storage:\n")
        print("# Save to disk (not in-memory)")
        print("client = chromadb.PersistentClient(")
        print('    path="./memory_os_db")')
        print("\n# Data persists across sessions!")
        print("# Perfect for MemoryOS! 🔥")
        return

    db_path = "./temp_chroma_demo_db"

    # Write session
    client1 = chromadb.PersistentClient(
        path=db_path)
    col1 = client1.get_or_create_collection(
        "persistent_demo")
    col1.add(
        documents=["This doc survives restart!"],
        ids=["persist_001"])
    count1 = col1.count()
    print(f"Session 1: Added 1 doc. Total: {count1}")

    # New client — same database
    client2 = chromadb.PersistentClient(
        path=db_path)
    col2 = client2.get_or_create_collection(
        "persistent_demo")
    count2 = col2.count()
    print(f"Session 2: Loaded DB. Count: {count2}")
    print(f"\n✅ Data persisted across client restarts!")
    print(f"   MemoryOS uses PersistentClient!")

    # Cleanup
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


if __name__ == "__main__":
    demonstrate_chromadb_basics()
    metadata_filtering_demo()
    persistent_storage_demo()
