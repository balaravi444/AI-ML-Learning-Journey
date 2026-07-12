"""
Day 25 — DSA: Recursion & Backtracking
Topic: Permutations — LeetCode #46
Date: 12 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Backtracking

Generate all possible orderings of array!

Time Complexity:  O(n! * n)
Space Complexity: O(n)

Real World Connection:
    Permutations used in:
    - Testing all possible layer orderings in neural nets
    - Finding optimal sequence in NLP
    - Route optimization in logistics ML!
"""


def permute(nums: list[int]) -> list[list[int]]:
    """
    Generate all permutations using backtracking.

    Args:
        nums: List of distinct integers

    Returns:
        All possible permutations

    Time Complexity: O(n! * n)
    Space Complexity: O(n)
    """
    result = []

    def backtrack(current: list, remaining: list) -> None:
        # Base case — used all numbers
        if not remaining:
            result.append(current.copy())
            return

        for i in range(len(remaining)):
            # Make choice
            current.append(remaining[i])
            new_remaining = remaining[:i] + remaining[i+1:]

            # Recurse
            backtrack(current, new_remaining)

            # Backtrack
            current.pop()

    backtrack([], nums)
    return result


if __name__ == "__main__":
    print("=== Permutations — LeetCode #46 ===")

    nums = [1, 2, 3]
    result = permute(nums)
    print(f"Input: {nums}")
    print(f"Total permutations: {len(result)}")
    for perm in result:
        print(f"  {perm}")
