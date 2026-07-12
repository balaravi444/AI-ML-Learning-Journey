"""
Day 21 — DSA: Arrays
Topic: Two Sum — LeetCode #1
Date: 08 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Hash Map

This is the most famous interview question!
Asked at Google, Amazon, Microsoft, Meta!

Time Complexity:  O(n) — hash map solution
Space Complexity: O(n) — hash map storage
"""


def two_sum_brute(nums: list[int], target: int) -> list[int]:
    """
    Find two numbers that add up to target.
    Brute force approach — O(n²).

    Args:
        nums: List of integers
        target: Target sum

    Returns:
        Indices of two numbers that add up to target
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def two_sum_optimal(nums: list[int], target: int) -> list[int]:
    """
    Find two numbers that add up to target.
    Optimal hash map approach — O(n).

    Key insight:
        For each number x, we need (target - x)
        Store seen numbers in hash map
        Check if complement exists!

    Args:
        nums: List of integers
        target: Target sum

    Returns:
        Indices of two numbers that add up to target

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = {}   # value → index

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i

    return []


if __name__ == "__main__":
    print("=== Two Sum — LeetCode #1 ===")

    # Test 1
    nums = [2, 7, 11, 15]
    target = 9
    print(f"nums={nums}, target={target}")
    print(f"Brute:   {two_sum_brute(nums, target)}")
    print(f"Optimal: {two_sum_optimal(nums, target)}")

    # Test 2
    nums = [3, 2, 4]
    target = 6
    print(f"\nnums={nums}, target={target}")
    print(f"Brute:   {two_sum_brute(nums, target)}")
    print(f"Optimal: {two_sum_optimal(nums, target)}")

    print("\n=== Why Hash Map is Better ===")
    print("Brute force:  O(n²) — checks every pair")
    print("Hash map:     O(n)  — one pass!")
    print("For n=1000000: O(n²)=10^12 vs O(n)=10^6 ops!")
