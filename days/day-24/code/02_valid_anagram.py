"""
Day 24 — DSA: Hash Maps & Sets
Topic: Valid Anagram — LeetCode #242
Date: 11 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Hash Map frequency count

Two strings are anagrams if they have
same characters with same frequencies!

Time Complexity:  O(n)
Space Complexity: O(1) — at most 26 characters

Real World Connection:
    Anagram detection is used in:
    - Spell checkers
    - NLP preprocessing
    - Plagiarism detection!
"""
from collections import Counter


def is_anagram(s: str, t: str) -> bool:
    """
    Check if two strings are anagrams.

    Key insight:
        Anagrams have same character frequencies!
        Build frequency map for both and compare.

    Args:
        s: First string
        t: Second string

    Returns:
        True if anagrams, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(s) != len(t):
        return False

    freq_s = {}
    freq_t = {}

    for char in s:
        freq_s[char] = freq_s.get(char, 0) + 1

    for char in t:
        freq_t[char] = freq_t.get(char, 0) + 1

    return freq_s == freq_t


def is_anagram_counter(s: str, t: str) -> bool:
    """
    Check anagram using Counter — cleaner solution!

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    return Counter(s) == Counter(t)


if __name__ == "__main__":
    print("=== Valid Anagram — LeetCode #242 ===")

    test_cases = [
        ("anagram", "nagaram", True),
        ("rat", "car", False),
        ("listen", "silent", True),
        ("hello", "world", False)
    ]

    for s, t, expected in test_cases:
        result = is_anagram(s, t)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{s}' & '{t}' → {result}")
