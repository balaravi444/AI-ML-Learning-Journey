"""
Day 24 — DSA: Hash Maps & Sets
Topic: Group Anagrams — LeetCode #49
Date: 11 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Hash Map with sorted key

Group words that are anagrams together!

Key insight:
    Anagrams have same sorted characters!
    "eat" → sorted → "aet"
    "tea" → sorted → "aet"  ← same key!
    Use sorted string as hash map key!

Time Complexity:  O(n * k log k) where k is max word length
Space Complexity: O(n * k)

Real World Connection:
    Grouping similar words is used in:
    - NLP stemming and lemmatization
    - Search query expansion
    - Synonym detection!
"""
from collections import defaultdict


def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Group words that are anagrams of each other.

    Args:
        words: List of strings

    Returns:
        List of groups where each group contains anagrams

    Time Complexity: O(n * k log k)
    Space Complexity: O(n * k)
    """
    groups = defaultdict(list)

    for word in words:
        # Sorted string is the key!
        key = tuple(sorted(word))
        groups[key].append(word)

    return list(groups.values())


if __name__ == "__main__":
    print("=== Group Anagrams — LeetCode #49 ===")

    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    result = group_anagrams(words)
    print(f"Words: {words}")
    print(f"Groups: {result}")

    # NLP example
    print("\n=== NLP Application ===")
    nlp_words = ["listen", "silent", "enlist",
                 "hello", "world", "inlets"]
    groups = group_anagrams(nlp_words)
    for group in groups:
        print(f"  Group: {group}")
