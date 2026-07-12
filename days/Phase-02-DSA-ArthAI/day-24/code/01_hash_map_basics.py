"""
Day 24 — DSA: Hash Maps & Sets
Topic: Hash Map Basics & Frequency Counter
Date: 11 June 2026
Author: Bala Ravi

Time Complexity: O(n) for all operations
Space Complexity: O(n)

Real World Connection:
    Word frequency counting is the foundation of NLP!
    TF-IDF, Bag of Words, word embeddings all
    start with counting word frequencies!
"""
from collections import defaultdict, Counter


def count_frequency(arr: list) -> dict:
    """
    Count frequency of each element.

    Args:
        arr: List of elements

    Returns:
        Dictionary with element frequencies

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq


def count_word_frequency(text: str) -> dict:
    """
    Count word frequency in text.
    Foundation of NLP processing!

    Args:
        text: Input text string

    Returns:
        Dictionary mapping words to frequencies

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    words = text.lower().split()
    return count_frequency(words)


def top_k_frequent(arr: list, k: int) -> list:
    """
    Find top k most frequent elements.

    Args:
        arr: List of elements
        k: Number of top elements to return

    Returns:
        List of k most frequent elements

    Time Complexity: O(n log k)
    Space Complexity: O(n)
    """
    freq = count_frequency(arr)
    return sorted(freq.keys(),
                  key=lambda x: freq[x],
                  reverse=True)[:k]


if __name__ == "__main__":
    print("=== Frequency Counter ===")
    nums = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4]
    freq = count_frequency(nums)
    print(f"Array: {nums}")
    print(f"Frequency: {freq}")

    print("\n=== Word Frequency (NLP Foundation) ===")
    text = "AI is amazing and Python is great for AI and ML"
    word_freq = count_word_frequency(text)
    for word, count in sorted(word_freq.items(),
                               key=lambda x: x[1],
                               reverse=True):
        print(f"  '{word}': {count}")

    print("\n=== Top K Frequent ===")
    elements = [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5]
    top3 = top_k_frequent(elements, 3)
    print(f"Top 3 frequent in {elements}: {top3}")
