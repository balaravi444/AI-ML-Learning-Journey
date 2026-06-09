"""
Day 21 — DSA: Arrays
Topic: Maximum Subarray — LeetCode #53
Date: 08 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Kadane's Algorithm — Dynamic Programming!

This is a VERY important algorithm used in:
- Stock price analysis
- Signal processing in ML
- Finding best time window in time series!

Time Complexity:  O(n)
Space Complexity: O(1)
"""


def max_subarray_brute(nums: list[int]) -> int:
    """
    Find maximum sum subarray using brute force.
    O(n²) — checks all subarrays.

    Args:
        nums: List of integers

    Returns:
        Maximum subarray sum
    """
    max_sum = float('-inf')
    n = len(nums)

    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += nums[j]
            max_sum = max(max_sum, current_sum)

    return max_sum


def max_subarray_kadane(nums: list[int]) -> int:
    """
    Find maximum sum subarray using Kadane's algorithm.

    Key insight:
        At each position, decide:
        1. Extend current subarray
        2. Start new subarray from current element
        Whichever gives larger sum!

    Args:
        nums: List of integers

    Returns:
        Maximum subarray sum

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    max_sum = nums[0]
    current_sum = nums[0]

    for i in range(1, len(nums)):
        # Either extend current subarray or start fresh
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)

    return max_sum


if __name__ == "__main__":
    print("=== Maximum Subarray — LeetCode #53 ===")

    test_cases = [
        [-2, 1, -3, 4, -1, 2, 1, -5, 4],  # 6
        [1],                                 # 1
        [5, 4, -1, 7, 8],                   # 23
        [-1, -2, -3, -4]                    # -1
    ]

    for nums in test_cases:
        result = max_subarray_kadane(nums)
        print(f"{nums}")
        print(f"Max subarray sum: {result}\n")

    print("=== Real World Use Case ===")
    # Best profit window in stock prices
    price_changes = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    best_profit = max_subarray_kadane(price_changes)
    print(f"Price changes: {price_changes}")
    print(f"Best profit window: {best_profit}")
