# Day 84 — Vector Databases + Embeddings 🚀

**Date:** 10 August 2026
**Phase:** 5 — Deep Learning + AI
**Time Spent:** (add your hours)
**Resource Used:** [ChromaDB Docs](https://docs.trychroma.com)

---

## 📚 Topics Covered

- What are vector databases
- How embedding search works
- ChromaDB — local vector store
- Creating collections + adding documents
- Semantic search with ChromaDB
- Metadata filtering
- Persistent storage
- ChromaDB vs Pinecone vs FAISS

---

## 🔑 What is a Vector Database?
Traditional database:
SELECT * FROM docs WHERE title = 'RAG'
→ Exact keyword match only!

Vector database:
query = embed("how does retrieval work?")
→ find all docs where embed(doc) ≈ query
→ Returns semantically similar documents!

How it works:

Text → embedding → dense vector

---

## 🔑 ChromaDB

```python
import chromadb
from chromadb.utils import embedding_functions

# Create client
client = chromadb.Client()  # in-memory
# OR
client = chromadb.PersistentClient(path="./db")

# Embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")

# Create collection
collection = client.create_collection(
    name="my_memory",
    embedding_function=ef)

# Add documents
collection.add(
    documents=["RAG is retrieval augmented generation"],
    metadatas=[{"source": "notes", "date": "2026-08-10"}],
    ids=["doc_001"])

# Query
results = collection.query(
    query_texts=["how does retrieval work?"],
    n_results=3)
# Returns: docs, distances, metadatas
```

---

## 🔑 HNSW — How Fast Search Works
Naive search: compare query to ALL documents
Time: O(n) — scales badly!
1M documents = 1M comparisons = slow!

HNSW (Hierarchical Navigable Small World):
Graph-based approximate nearest neighbor search
Time: O(log n) — fast!
1M documents = ~20 comparisons!

ChromaDB uses HNSW internally.
You never see it — just fast search! 🔥

---

## 💎 Important Realizations

1. **Vector DB is the memory in MemoryOS**
   Every document user saves → embed → store in ChromaDB
   Every query → embed → search ChromaDB → retrieve → answer!
   ChromaDB IS the memory layer!

2. **Metadata filtering is powerful**
   "Find AI documents from last week"
   → filter by source='ai' AND date > '2026-08-03'
   → then semantic search in filtered subset!

3. **Persistent client = data survives restarts**
   In-memory: fast but data lost on restart
   PersistentClient: data saved to disk
   MemoryOS needs persistent! Always!

4. **ChromaDB vs Pinecone**
   ChromaDB: free, local, perfect for learning
   Pinecone: cloud, scalable, production
   Start with ChromaDB, migrate if needed!

---

## 🎯 Next Goal (Day 85)

- RAG pipeline!
- Document loading → chunking → embedding
  → ChromaDB → LangChain retrieval → LLM → answer!
- The complete MemoryOS core!

---

*Day 84 complete — Vector DBs mastered! 💾🔥*


Store vector + metadata + original text
Query → embed query → find nearest vectors
Return top-k most similar documents!
