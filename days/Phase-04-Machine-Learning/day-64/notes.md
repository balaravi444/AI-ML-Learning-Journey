# Day 64 — Word Embeddings 🚀

**Date:** 21 July 2026
**Time Spent:** (2 hours)
**Resource Used:** [Gensim Docs](https://radimrehurek.com/gensim/) | [Jay Alammar Blog](https://jalammar.github.io/)

---

## 📚 Topics Covered

- Why TF-IDF fails at meaning
- What are word embeddings
- Word2Vec — Skip-gram and CBOW
- GloVe embeddings
- The famous king - man + woman = queen
- Semantic similarity with embeddings
- Sentence embeddings
- When to use embeddings vs TF-IDF

---

## 🔑 Why TF-IDF Fails at Meaning
TF-IDF treats every word as independent.
No concept of meaning or relationship.

"The server crashed"
"The server went down"

TF-IDF sees these as COMPLETELY different!
"crashed" and "down" have zero connection.

But they mean the SAME THING!

Word Embeddings fix this:
"crashed" → [0.82, -0.31, 0.54, ...]
"down" → [0.79, -0.28, 0.51, ...]

Similar meaning → similar vectors!
Distance between vectors = semantic distance!
---

## 🔑 What Are Word Embeddings?
Every word → dense vector of numbers
Usually 100 to 300 dimensions

"king" → [0.50, 0.12, -0.34, 0.78, ...]
"queen" → [0.48, 0.10, -0.32, 0.91, ...]
"man" → [0.21, 0.09, -0.11, 0.32, ...]
"woman" → [0.22, 0.11, -0.09, 0.45, ...]

Similar words → close vectors in space!
Different words → far apart vectors!

The famous example:
king - man + woman ≈ queen

Vector arithmetic captures MEANING! 🔥
---

## 🔑 Word2Vec — How It Learns
Skip-gram: given a word → predict context words
Input: "crashed"
Target: ["server", "production", "down", "error"]

Train a neural network to predict context.
The HIDDEN LAYER weights = word embeddings!
Model never explicitly taught relationships —
it discovers them from co-occurrence patterns!

CBOW (Continuous Bag of Words):
Input: ["server", "production", "error"]
Target: "crashed"
Opposite of Skip-gram — context → center word

Skip-gram works better for rare words.
CBOW is faster for large corpora.
---

## 🔑 Cosine Similarity for Embeddings

```python
from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Range: -1 to 1
# 1.0  → identical direction (same meaning)
# 0.0  → orthogonal (unrelated)
# -1.0 → opposite direction (opposite meaning)

cosine_sim(vec("crash"), vec("down"))   # → 0.87
cosine_sim(vec("crash"), vec("typo"))   # → 0.12
cosine_sim(vec("crash"), vec("server")) # → 0.63
```

---

## 🔑 TF-IDF vs Embeddings
TF-IDF:
✅ Fast, interpretable
✅ Works with small datasets
✅ Great for keyword matching
❌ No semantic understanding
❌ "crash" and "down" = unrelated

Word Embeddings:
✅ Captures meaning and relationships
✅ Handles synonyms automatically
✅ Works across languages (multilingual)
❌ Needs large training corpus
❌ Less interpretable
❌ Requires more compute

When to use what:
Small dataset + keywords → TF-IDF
Large dataset + meaning → Embeddings
Production semantic search → Embeddings
Bug Predictor (Day 59) → TF-IDF ✅ (right choice!)
MemoryOS (Day 87) → Embeddings ✅
---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Word2Vec from scratch concept | Skip-gram intuition |
| 2 | Gensim Word2Vec | Train on corpus |
| 3 | Pretrained GloVe | king - man + woman |
| 4 | Sentence embeddings | Average word vectors |
| 5 | Semantic similarity | Find similar documents |
| 6 | Job description matching | Embedding-based search |

---

## 🔗 How This Connects to AI/ML

```python
# Word Embeddings → foundation of Transformers!
# BERT, GPT, all use learned embeddings
# But much more powerful (contextual!)

# "bank" in "river bank" vs "bank account"
# TF-IDF: same vector both times!
# Word2Vec: same vector both times!
# BERT: DIFFERENT vector! Context-aware! 🔥

# Day 82: HuggingFace sentence-transformers
# Returns contextual embeddings per sentence
# MUCH better than Word2Vec averages
# That's what MemoryOS will use!

# Progression:
# TF-IDF (Day 63) → Word2Vec (Day 64)
# → Transformers (Day 82) → MemoryOS (Day 87)
```

---

## 💎 Important Realizations

1. **Embeddings capture relationships TF-IDF cannot**
   "server crashed" ≈ "system went down"
   Embeddings know this. TF-IDF is blind to it.

2. **The magic is in the training objective**
   Word2Vec never sees "king is to queen as man is to woman"
   It just predicts context words.
   The geometry emerges automatically!

3. **Pretrained embeddings > training from scratch**
   GloVe trained on 840B tokens of text.
   Your dataset has 1000 issues.
   Always start with pretrained! Fine-tune if needed.

4. **Sentence embeddings = average word vectors**
   Simple but effective baseline!
   Better: use sentence-transformers (Day 82)

---

## 🎯 Next Goal (Day 65)

- Sentiment Analysis!
- Classify text as positive / negative / neutral
- Uses NLP + ML together
- Real application — review analyzer

---

*Day 64 complete — Word Embeddings mastered! 🔢🔥*


