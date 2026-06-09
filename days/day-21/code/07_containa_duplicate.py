"""
Day 21 — DSA: Arrays
Topic: Contains Duplicate — LeetCode #217
Date: 08 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Hash Set

Time Complexity:  O(n)
Space Complexity: O(n)
"""


def contains_duplicate_brute(nums: list[int]) -> bool:
    """
    Check if array contains duplicates.
    Brute force — O(n²).

    Args:
        nums: List of integers

    Returns:
        True if any duplicate exists, False otherwise
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] == nums[j]:
                return True
    return False


def contains_duplicate_optimal(nums: list[int]) -> bool:
    """
    Check if array contains duplicates.
    Hash set approach — O(n).

    Args:
        nums: List of integers

    Returns:
        True if any duplicate exists, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False


if __name__ == "__main__":
    print("=== Contains Duplicate — LeetCode #217 ===")

    test_cases = [
        [1, 2, 3, 1],           # True
        [1, 2, 3, 4],           # False
        [1, 1, 1, 3, 3, 4, 3]  # True
    ]

    for nums in test_cases:
        result = contains_duplicate_optimal(nums)
        print(f"{nums} → {result}")
