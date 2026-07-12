"""
Day 08 — Collections Module
Topic: Counter for frequency counting
Date: 26 May 2026
Author: Bala Ravi

Real World Connection:
    Counter is used in NLP for word frequency analysis!
    Word frequency is the foundation of:
    - TF-IDF (search engines)
    - Bag of Words (spam detection)
    - Sentiment analysis
"""
from collections import Counter


def count_word_frequency(text: str) -> Counter:
    """
    Count frequency of each word in text.

    Args:
        text: Input string to analyze

    Returns:
        Counter object with word frequencies
    """
    words = text.lower().split()
    return Counter(words)


def count_char_frequency(text: str) -> Counter:
    """
    Count frequency of each character in text.

    Args:
        text: Input string to analyze

    Returns:
        Counter object with character frequencies
    """
    return Counter(text.lower())


def get_top_words(text: str, n: int = 5) -> list:
    """
    Get top n most common words in text.

    Args:
        text: Input string to analyze
        n: Number of top words to return. Defaults to 5

    Returns:
        List of (word, count) tuples
    """
    freq = count_word_frequency(text)
    return freq.most_common(n)


if __name__ == "__main__":
    # NLP example
    text = "AI is the future of technology. AI will change the world. Python is the language of AI."

    print("=== Word Frequency ===")
    freq = count_word_frequency(text)
    print(freq)

    print("\n=== Top 5 Words ===")
    top = get_top_words(text, 5)
    for word, count in top:
        print(f"{word}: {count}")

    print("\n=== Character Frequency ===")
    char_freq = count_char_frequency("hello")
    print(char_freq)
