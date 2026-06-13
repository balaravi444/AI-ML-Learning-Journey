"""
Day 25 — DSA: Recursion & Backtracking
Topic: Subsets — LeetCode #78
Date: 12 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Backtracking

Generate all possible subsets of an array!

Time Complexity:  O(2ⁿ) — 2 choices per element
Space Complexity: O(n) — recursion depth

Real World Connection:
    Subset generation is used in:
    - Feature selection in ML (which features to use?)
    - Ensemble learning (which models to combine?)
    - Generating all possible hyperparameter combos!
"""


def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets using backtracking.

    Key insight:
        For each element — two choices:
        1. Include it
        2. Exclude it
        Total subsets = 2ⁿ

    Args:
        nums: List of distinct integers

    Returns:
        All possible subsets

    Time Complexity: O(2ⁿ)
    Space Complexity: O(n)
    """
    result = []

    def backtrack(start: int, current: list) -> None:
        # Every state is a valid subset!
        result.append(current.copy())

        for i in range(start, len(nums)):
            current.append(nums[i])         # include
            backtrack(i + 1, current)       # recurse
            current.pop()                   # backtrack!

    backtrack(0, [])
    return result


if __name__ == "__main__":
    print("=== Subsets — LeetCode #78 ===")

    nums = [1, 2, 3]
    result = subsets(nums)
    print(f"Input: {nums}")
    print(f"Total subsets: {len(result)}")
    print(f"All subsets: {result}")

    print("\n=== ML Connection — Feature Selection ===")
    features = ["age", "salary", "experience"]
    all_feature_combos = subsets(features)
    print(f"All possible feature combinations:")
    for combo in all_feature_combos:
        if combo:  # skip empty
            print(f"  {combo}")
