"""
Day 85 — RAG: Retrieval Augmented Generation
Topic: Complete RAG Pipeline
Date: 11 August 2026
Author: Bala Ravi

The core of MemoryOS!
Load → Split → Embed → Store → Retrieve → Generate
"""
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import (
        SentenceTransformer)
    from sklearn.metrics.pairwise import (
        cosine_similarity)
    import numpy as np
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

GEMINI_API_KEY = os.environ.get(
    'GEMINI_API_KEY', 'your-api-key-here')


class SimpleRAGPipeline:
    """
    Complete RAG pipeline from scratch.

    Load → Split → Embed → Store → Retrieve → Generate

    This IS the MemoryOS core!
    Day 87 extends this with full product features.
    """

    def __init__(self,
                  collection_name: str = "rag_demo",
                  chunk_size: int = 400,
                  chunk_overlap: int = 50,
                  n_retrieve: int = 3) -> None:
        """
        Initialize RAG pipeline.

        Args:
            collection_name: ChromaDB collection name
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            n_retrieve: Docs to retrieve per query
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_retrieve = n_retrieve
        self.chunks = []
        self.metadatas = []
        self.embeddings = None

        if ST_AVAILABLE:
            print("Loading sentence-transformers...")
            self.embedder = SentenceTransformer(
                'all-MiniLM-L6-v2')
        else:
            self.embedder = None

        if CHROMA_AVAILABLE:
            self.chroma_client = chromadb.Client()
            try:
                self.chroma_client.delete_collection(
                    collection_name)
            except Exception:
                pass
            self.collection = (
                self.chroma_client.create_collection(
                    collection_name))
        else:
            self.collection = None

        print(f"✅ RAG Pipeline initialized")
        print(f"   Chunk size: {chunk_size} chars")
        print(f"   Overlap:    {chunk_overlap} chars")
        print(f"   Retrieve:   {n_retrieve} docs\n")

    def split_text(
            self,
            text: str,
            source: str = "unknown"
            ) -> list:
        """
        Split text into overlapping chunks.

        RecursiveCharacterTextSplitter approach:
        Try paragraph splits first, then sentence,
        then character.

        Args:
            text: Input text
            source: Source metadata

        Returns:
            List of chunk dictionaries
        """
        chunks = []
        separators = ['\n\n', '\n', '. ', ' ']

        def split_with_separator(
                text: str,
                sep: str) -> list:
            if sep in text:
                return [
                    s.strip() for s in
                    text.split(sep)
                    if s.strip()]
            return [text]

        # Try to split on paragraphs
        paragraphs = split_with_separator(
            text, '\n\n')

        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= (
                    self.chunk_size):
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'source': source})
                # Start new chunk with overlap
                if len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[
                        -self.chunk_overlap:]
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'source': source})

        return chunks

    def add_document(
            self,
            text: str,
            source: str = "document",
            metadata: dict = None) -> int:
        """
        Add document to RAG index.

        Splits → embeds → stores in ChromaDB.

        Args:
            text: Document text
            source: Document source
            metadata: Additional metadata

        Returns:
            Number of chunks added
        """
        chunks = self.split_text(text, source)
        if not chunks:
            return 0

        chunk_texts = [c['text'] for c in chunks]
        n_existing = len(self.chunks)

        if self.embedder and ST_AVAILABLE:
            new_embeddings = self.embedder.encode(
                chunk_texts)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([
                    self.embeddings, new_embeddings])

        for i, (chunk, text) in enumerate(
                zip(chunks, chunk_texts)):
            meta = {
                'source': source,
                'chunk_idx': i,
                'chunk_size': len(text)}
            if metadata:
                meta.update(metadata)
            self.chunks.append(text)
            self.metadatas.append(meta)

        if self.collection is not None:
            ids = [
                f"chunk_{n_existing + i}"
                for i in range(len(chunks))]
            try:
                self.collection.add(
                    documents=chunk_texts,
                    metadatas=self.metadatas[
                        n_existing:],
                    ids=ids)
            except Exception as e:
                pass

        return len(chunks)

    def retrieve(
            self,
            query: str) -> list:
        """
        Retrieve most relevant chunks for query.

        Query → embed → cosine similarity → top-k

        Args:
            query: User's question

        Returns:
            List of relevant chunk dicts
        """
        if not self.chunks:
            return []

        if (self.embedder and ST_AVAILABLE and
                self.embeddings is not None):
            query_emb = self.embedder.encode([query])
            sims = cosine_similarity(
                query_emb, self.embeddings)[0]
            top_k = min(
                self.n_retrieve, len(self.chunks))
            top_indices = sims.argsort()[
                ::-1][:top_k]

            return [
                {
                    'text': self.chunks[i],
                    'metadata': self.metadatas[i],
                    'similarity': float(sims[i])
                }
                for i in top_indices
                if sims[i] > 0.1
            ]

        elif self.collection is not None:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(
                    self.n_retrieve,
                    len(self.chunks)))
            retrieved = []
            for doc, meta in zip(
                    results['documents'][0],
                    results['metadatas'][0]):
                retrieved.append({
                    'text': doc,
                    'metadata': meta,
                    'similarity': 0.8})
            return retrieved

        # Fallback: keyword search
        query_words = set(query.lower().split())
        scores = []
        for chunk in self.chunks:
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            scores.append(score)

        top_k = min(self.n_retrieve, len(self.chunks))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True)[:top_k]

        return [
            {
                'text': self.chunks[i],
                'metadata': self.metadatas[i],
                'similarity': scores[i] / max(
                    len(query_words), 1)
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def generate(
            self,
            query: str,
            context_docs: list) -> str:
        """
        Generate answer using retrieved context.

        Builds RAG prompt:
        Context: [retrieved docs]
        Question: [user query]
        Answer: [LLM generates this]

        Args:
            query: User's question
            context_docs: Retrieved documents

        Returns:
            Generated answer string
        """
        if not context_docs:
            return ("I couldn't find relevant "
                    "information in my knowledge base. "
                    "Please add more documents first.")

        context = "\n\n---\n\n".join([
            f"[From: {doc['metadata'].get('source', '?')}]\n"
            f"{doc['text']}"
            for doc in context_docs])

        rag_prompt = f"""You are a helpful AI assistant.
Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say so.

CONTEXT:
{context}

QUESTION: {query}

ANSWER (based on context above):"""

        # Try Gemini API
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                'gemini-1.5-flash')
            response = model.generate_content(
                rag_prompt)
            return response.text
        except Exception:
            pass

        # Fallback: return most relevant chunk
        best = context_docs[0]
        return (f"Based on documents, here is "
                f"what I found:\n\n"
                f"{best['text'][:300]}...\n\n"
                f"[Source: "
                f"{best['metadata'].get('source', '?')}]")

    def ask(self, query: str) -> dict:
        """
        Complete RAG query.

        Query → Retrieve → Generate → Return

        Args:
            query: User's question

        Returns:
            Dictionary with answer + sources
        """
        retrieved = self.retrieve(query)
        answer = self.generate(query, retrieved)

        return {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'source': doc['metadata'].get(
                        'source', '?'),
                    'similarity': round(
                        doc['similarity'], 3),
                    'excerpt': doc['text'][:100] + '...'
                }
                for doc in retrieved
            ],
            'n_docs_retrieved': len(retrieved)
        }


def demonstrate_rag() -> None:
    """Show complete RAG pipeline in action."""
    print("=== Complete RAG Pipeline Demo ===\n")
    print("This IS the MemoryOS core! 🔥\n")

    rag = SimpleRAGPipeline(
        collection_name="memoryos_demo",
        chunk_size=400,
        chunk_overlap=50,
        n_retrieve=3)

    # Add learning journey documents
    documents = [
        {
            'text': """
RAG (Retrieval Augmented Generation) is a technique
that combines information retrieval with text generation.

Instead of relying solely on the LLM's parametric memory,
RAG systems first retrieve relevant documents from an
external knowledge base, then provide these documents
as context to the LLM when generating responses.

The 5-step RAG pipeline:
1. Load documents (PDF, Markdown, Web)
2. Split into chunks (chunk_size=500, overlap=50)
3. Embed chunks using sentence-transformers
4. Store embeddings in ChromaDB
5. At query time: retrieve relevant chunks → LLM

Benefits: No hallucination, private data support,
updatable knowledge base, source attribution.
            """.strip(),
            'source': 'day-85-notes',
            'metadata': {
                'day': 85, 'topic': 'RAG',
                'phase': 5}
        },
        {
            'text': """
Vector databases store high-dimensional embeddings
and enable fast semantic similarity search.

ChromaDB is an open-source vector database that
runs locally without any cloud setup. It uses HNSW
(Hierarchical Navigable Small World) graphs for
approximate nearest neighbor search.

Key ChromaDB operations:
- client.create_collection() → new vector store
- collection.add() → store documents + embeddings
- collection.query() → semantic search
- Metadata filtering with where= parameter
- PersistentClient for data across restarts

ChromaDB vs alternatives:
- ChromaDB: local, free, easy setup (our choice!)
- Pinecone: cloud, scalable, production
- FAISS: Facebook, fast but limited features
            """.strip(),
            'source': 'day-84-notes',
            'metadata': {
                'day': 84, 'topic': 'VectorDB',
                'phase': 5}
        },
        {
            'text': """
Transformers revolutionized NLP in 2017 with the
paper "Attention Is All You Need" by Vaswani et al.

The key innovation is self-attention:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) @ V

Q (Query), K (Key), V (Value) are learned linear
projections of the input embeddings.

BERT uses bidirectional encoder (sees all tokens).
GPT uses causal decoder (sees only past tokens).

sentence-transformers uses BERT-style encoders
to convert sentences into dense 384-dim vectors
that capture semantic meaning for similarity search.
            """.strip(),
            'source': 'day-81-notes',
            'metadata': {
                'day': 81, 'topic': 'Transformers',
                'phase': 5}
        },
        {
            'text': """
MobileNetV2 is a lightweight CNN architecture
designed for mobile and edge deployment.

Key features:
- 3.4M parameters (vs 138M for VGG16)
- Inverted residuals and linear bottlenecks
- Trained on ImageNet (1.2M images, 1000 classes)
- Feature extraction: 7x7x1280 = 62,720 features

For skin disease detection we used MobileNetV2
with transfer learning:
Phase 1: Freeze base, train head (LR=1e-3)
Phase 2: Unfreeze top 54 layers (LR=1e-5)

Achieved 89% accuracy on 7 skin disease classes
with melanoma recall of 87%.
            """.strip(),
            'source': 'day-73-74-notes',
            'metadata': {
                'day': 74, 'topic': 'TransferLearning',
                'phase': 5}
        }
    ]

    # Index documents
    print("Indexing documents...")
    total_chunks = 0
    for doc in documents:
        n = rag.add_document(
            text=doc['text'],
            source=doc['source'],
            metadata=doc.get('metadata', {}))
        total_chunks += n
        print(f"  ✅ {doc['source']}: {n} chunks")

    print(f"\nTotal chunks indexed: {total_chunks}\n")

    # RAG queries
    queries = [
        "How does RAG work and what are its benefits?",
        "What is the difference between BERT and GPT?",
        "How did we train the skin disease model?",
        "What database should I use for vector search?"
    ]

    print(f"{'='*55}")
    print(f"RAG Query + Answer Demo:")
    print(f"{'='*55}\n")

    for query in queries:
        result = rag.ask(query)
        print(f"❓ {result['query']}")
        print(f"\n💬 {result['answer'][:200]}...")
        print(f"\n📚 Sources ({result['n_docs_retrieved']}):")
        for src in result['sources']:
            print(f"   [{src['source']}] "
                  f"sim={src['similarity']:.3f}")
            print(f"   {src['excerpt'][:60]}...")
        print(f"\n{'─'*55}\n")


def chunking_strategy_comparison() -> None:
    """Compare different chunking strategies."""
    print("=== Chunking Strategy Comparison ===\n")

    sample_text = """
RAG combines retrieval with generation.
It retrieves relevant documents first.

Then the LLM uses those documents as context.
This prevents hallucination significantly.

ChromaDB stores the embeddings efficiently.
sentence-transformers creates the embeddings.
The pipeline works end-to-end seamlessly.
    """.strip()

    strategies = {
        'chunk_size=200 (small)': 200,
        'chunk_size=400 (medium)': 400,
        'chunk_size=600 (large)': 600
    }

    print(f"Input text: {len(sample_text)} chars\n")

    for name, size in strategies.items():
        rag = SimpleRAGPipeline(
            chunk_size=size, chunk_overlap=50)
        chunks = rag.split_text(
            sample_text, "test")
        print(f"{name}:")
        print(f"  Chunks: {len(chunks)}")
        if chunks:
            avg_len = sum(
                len(c['text'])
                for c in chunks) / len(chunks)
            print(f"  Avg chunk size: {avg_len:.0f} chars")
        print()

    print("💡 Sweet spot: chunk_size=400-500")
    print("   Small: too fragmented, loses context")
    print("   Large: too broad, poor retrieval precision")
    print("   MemoryOS uses 500 chars + 50 overlap! 🔥")


if __name__ == "__main__":
    demonstrate_rag()
    chunking_strategy_comparison()
