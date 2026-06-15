"""
Day 28 — DSA: Dynamic Programming
Topic: Edit Distance — LeetCode #72
Date: 15 June 2026
Author: Bala Ravi

Difficulty: Hard
Pattern: 2D String DP

Minimum operations to convert word1 to word2.
Operations: insert, delete, replace

State: dp[i][j] = edit distance of word1[:i] and word2[:j]
Recurrence:
    if chars match: dp[i][j] = dp[i-1][j-1]
    else: dp[i][j] = 1 + min(dp[i-1][j],    # delete
                              dp[i][j-1],    # insert
                              dp[i-1][j-1]) # replace

Time Complexity:  O(m * n)
Space Complexity: O(m * n)

Real World Connection:
    Edit Distance (Levenshtein Distance) used in:
    - Autocorrect in every keyboard app!
    - Spell checking in NLP models!
    - Fuzzy string matching in search engines!
    - DNA mutation analysis in bioinformatics ML!
    - Voice recognition error correction!
"""


def edit_distance(word1: str, word2: str) -> int:
    """
    Find minimum edit distance between two strings.

    Args:
        word1: Source string
        word2: Target string

    Returns:
        Minimum operations to convert word1 to word2

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # delete all chars of word1
    for j in range(n + 1):
        dp[0][j] = j  # insert all chars of word2

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # no operation
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # delete
                    dp[i][j - 1],      # insert
                    dp[i - 1][j - 1]   # replace
                )

    return dp[m][n]


def similarity_score(word1: str, word2: str) -> float:
    """
    Calculate similarity between two strings.
    Used in fuzzy matching for NLP!

    Args:
        word1: First string
        word2: Second string

    Returns:
        Similarity score 0-100%
    """
    distance = edit_distance(word1, word2)
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 100.0
    return (1 - distance / max_len) * 100


if __name__ == "__main__":
    print("=== Edit Distance — LeetCode #72 ===")

    test_cases = [
        ("horse", "ros", 3),
        ("intention", "execution", 5),
        ("", "abc", 3),
        ("abc", "abc", 0)
    ]

    for w1, w2, expected in test_cases:
        result = edit_distance(w1, w2)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{w1}' → '{w2}': {result} ops")

    print("\n=== Autocorrect Simulation ===")
    dictionary = ["python", "machine", "learning",
                  "algorithm", "dynamic", "programming"]
    typo = "pyhton"

    best_match = min(dictionary,
                     key=lambda w: edit_distance(typo, w))
    score = similarity_score(typo, best_match)
    print(f"Typo: '{typo}'")
    print(f"Best match: '{best_match}' ({score:.1f}% similar)")
    print(f"Did you mean: '{best_match}'?")
