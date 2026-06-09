"""
Day 22 — DSA: Monotonic Stack
Topic: Next Greater Element — LeetCode #496
Date: 09 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Monotonic Stack

Monotonic stack = stack that maintains
elements in increasing or decreasing order!

Real World Connection:
    Used in stock price analysis!
    "For each day, find the next day with higher price"
    Used in anomaly detection in time series ML!

Time Complexity:  O(n)
Space Complexity: O(n)
"""


def next_greater_element(nums1: list[int],
                         nums2: list[int]) -> list[int]:
    """
    Find next greater element for each element in nums1
    from nums2.

    Args:
        nums1: Subset of nums2
        nums2: Main array

    Returns:
        Next greater element for each num in nums1,
        -1 if none exists

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Build next greater map using monotonic stack
    stack = []
    next_greater = {}

    for num in nums2:
        # While stack not empty and current > stack top
        while stack and num > stack[-1]:
            next_greater[stack.pop()] = num
        stack.append(num)

    # Remaining elements have no next greater
    while stack:
        next_greater[stack.pop()] = -1

    return [next_greater[num] for num in nums1]


def next_greater_stock_prices(prices: list[int]) -> list[int]:
    """
    For each day find next day with higher stock price.
    Real world time series analysis!

    Args:
        prices: Daily stock prices

    Returns:
        Next higher price for each day, -1 if none

    Time Complexity: O(n)
    """
    n = len(prices)
    result = [-1] * n
    stack = []  # monotonic decreasing stack

    for i in range(n):
        while stack and prices[i] > prices[stack[-1]]:
            idx = stack.pop()
            result[idx] = prices[i]
        stack.append(i)

    return result


if __name__ == "__main__":
    print("=== Next Greater Element — LeetCode #496 ===")

    nums1 = [4, 1, 2]
    nums2 = [1, 3, 4, 2]
    result = next_greater_element(nums1, nums2)
    print(f"nums1={nums1}")
    print(f"nums2={nums2}")
    print(f"Result={result}")

    print("\n=== Stock Price Analysis ===")
    prices = [73, 74, 75, 71, 69, 72, 76, 73]
    result = next_greater_stock_prices(prices)
    print(f"Prices:       {prices}")
    print(f"Next higher:  {result}")
