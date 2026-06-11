"""
Day 24 — DSA: Hash Maps & Sets
Topic: Word Frequency Analyzer for NLP
Date: 11 June 2026
Author: Bala Ravi

Real World Project:
    This is a simplified version of what
    search engines and NLP models do!
    Word frequency is the foundation of:
    - TF-IDF
    - Bag of Words
    - Word2Vec training
    - Spam detection

Time Complexity: O(n)
Space Complexity: O(n)
"""
import re
from collections import Counter


def preprocess_text(text: str) -> list[str]:
    """
    Clean and tokenize text for NLP.

    Args:
        text: Raw input text

    Returns:
        List of cleaned tokens
    """
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize
    return text.split()


def build_vocabulary(corpus: list[str]) -> dict[str, int]:
    """
    Build word to index vocabulary.
    Used in every NLP model!

    Args:
        corpus: List of words

    Returns:
        Dictionary mapping word to unique index

    Time Complexity: O(n)
    """
    vocab = {}
    for word in corpus:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def tf_idf_term_frequency(document: list[str]) -> dict[str, float]:
    """
    Calculate term frequency for a document.
    First step of TF-IDF!

    TF(word) = count(word) / total_words

    Args:
        document: List of words in document

    Returns:
        Dictionary mapping word to TF score

    Time Complexity: O(n)
    """
    freq = Counter(document)
    total = len(document)
    return {word: count / total
            for word, count in freq.items()}


def analyze_corpus(texts: list[str]) -> None:
    """
    Full NLP corpus analysis pipeline.

    Args:
        texts: List of text documents
    """
    print("=== NLP Corpus Analysis ===\n")

    all_words = []
    for i, text in enumerate(texts):
        words = preprocess_text(text)
        all_words.extend(words)
        print(f"Doc {i+1}: {words}")

    print(f"\nTotal words: {len(all_words)}")
    print(f"Unique words: {len(set(all_words))}")

    # Vocabulary
    vocab = build_vocabulary(all_words)
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample vocab: {dict(list(vocab.items())[:5])}")

    # Top 5 words
    freq = Counter(all_words)
    print(f"\nTop 5 words:")
    for word, count in freq.most_common(5):
        print(f"  '{word}': {count}")

    # TF for first document
    doc1_words = preprocess_text(texts[0])
    tf = tf_idf_term_frequency(doc1_words)
    print(f"\nTF scores for Doc 1:")
    for word, score in sorted(tf.items(),
                               key=lambda x: x[1],
                               reverse=True)[:5]:
        print(f"  '{word}': {score:.3f}")


if __name__ == "__main__":
    corpus = [
        "AI and Machine Learning are transforming the world",
        "Python is the best language for AI and ML development",
        "Deep Learning and Neural Networks power modern AI",
        "Data Science uses Python Pandas and NumPy for ML"
    ]

    analyze_corpus(corpus)
