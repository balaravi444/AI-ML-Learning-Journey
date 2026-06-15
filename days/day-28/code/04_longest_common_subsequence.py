"""
Day 28 — DSA: Dynamic Programming
Topic: Longest Common Subsequence — LeetCode #1143
Date: 15 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: 2D String DP

Find longest common subsequence of two strings!
Subsequence = characters in order but not necessarily adjacent.

State: dp[i][j] = LCS of text1[:i] and text2[:j]
Recurrence:
    if text1[i] == text2[j]: dp[i][j] = dp[i-1][j-1] + 1
    else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])

Time Complexity:  O(m * n)
Space Complexity: O(m * n)

Real World Connection:
    LCS used in:
    - DNA sequence alignment in bioinformatics ML!
    - Plagiarism detection in NLP!
    - Git diff algorithm (finding changed lines)!
    - File comparison tools!
"""


def lcs(text1: str, text2: str) -> int:
    """
    Find length of longest common subsequence.

    Args:
        text1: First string
        text2: Second string

    Returns:
        Length of LCS

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j],
                               dp[i][j - 1])

    return dp[m][n]


def lcs_string(text1: str, text2: str) -> str:
    """
    Find the actual LCS string (not just length).

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j],
                               dp[i][j - 1])

    # Backtrack to find actual LCS
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            result.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(result))


if __name__ == "__main__":
    print("=== Longest Common Subsequence — LeetCode #1143 ===")

    test_cases = [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0)
    ]

    for t1, t2, expected in test_cases:
        result = lcs(t1, t2)
        actual = lcs_string(t1, t2)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{t1}' & '{t2}' → length={result}, LCS='{actual}'")

    print("\n=== DNA Sequence Alignment (Bioinformatics ML) ===")
    dna1 = "AGGTAB"
    dna2 = "GXTXAYB"
    print(f"DNA1: {dna1}")
    print(f"DNA2: {dna2}")
    print(f"Common sequence: {lcs_string(dna1, dna2)}")
    print(f"Similarity: {lcs(dna1, dna2)/max(len(dna1), len(dna2))*100:.1f}%")
