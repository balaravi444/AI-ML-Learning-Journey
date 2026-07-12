"""
Day 24 — DSA: Hash Maps & Sets
Topic: Longest Substring Without Repeating — LeetCode #3
Date: 11 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Sliding Window + Hash Map

Classic sliding window problem!
Find longest window with all unique characters.

Time Complexity:  O(n)
Space Complexity: O(min(n, 26)) — unique chars

Real World Connection:
    Sliding window is used in:
    - Time series analysis in ML
    - Text feature extraction in NLP
    - Signal processing for audio ML models!
"""


def length_of_longest_substring(s: str) -> int:
    """
    Find length of longest substring without repeating chars.

    Key insight:
        Sliding window with hash map tracking last position!
        When duplicate found → move left pointer past it.

    Args:
        s: Input string

    Returns:
        Length of longest valid substring

    Time Complexity: O(n)
    Space Complexity: O(min(n, 26))
    """
    seen = {}           # char → last seen index
    left = 0
    max_len = 0

    for right, char in enumerate(s):
        # If char seen and within current window
        if char in seen and seen[char] >= left:
            left = seen[char] + 1   # shrink window!

        seen[char] = right          # update last position
        max_len = max(max_len, right - left + 1)

    return max_len


def get_longest_substring(s: str) -> str:
    """
    Get the actual longest substring (not just length).

    Args:
        s: Input string

    Returns:
        Longest substring without repeating characters
    """
    seen = {}
    left = 0
    max_len = 0
    max_start = 0

    for right, char in enumerate(s):
        if char in seen and seen[char] >= left:
            left = seen[char] + 1

        seen[char] = right

        if right - left + 1 > max_len:
            max_len = right - left + 1
            max_start = left

    return s[max_start:max_start + max_len]


if __name__ == "__main__":
    print("=== Longest Substring — LeetCode #3 ===")

    test_cases = [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("", 0)
    ]

    for s, expected in test_cases:
        result = length_of_longest_substring(s)
        actual = get_longest_substring(s)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{s}' → length={result}, substring='{actual}'")
