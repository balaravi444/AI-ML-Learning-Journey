"""
Day 29 — Advanced DSA Patterns
Topic: Sliding Window Pattern
Date: 16 June 2026
Author: Bala Ravi

Sliding Window = maintain a window over array
Expand right, shrink left when invalid!

Real World Connection:
    Moving averages in stock analysis (ArthAI!)
    Rolling statistics in ML feature engineering!
"""
from collections import defaultdict


def max_average_subarray(nums: list[int],
                          k: int) -> float:
    """
    Maximum average subarray of size k.
    LeetCode #643

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum / k


def longest_substring_k_distinct(s: str,
                                   k: int) -> int:
    """
    Longest substring with at most k distinct chars.

    Time Complexity: O(n)
    Space Complexity: O(k)
    """
    char_count = defaultdict(int)
    left = 0
    max_len = 0

    for right in range(len(s)):
        char_count[s[right]] += 1

        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len


def moving_average_stocks(prices: list[float],
                           window: int) -> list[float]:
    """
    Calculate moving average for stock prices.
    DIRECTLY used in ArthAI stock analysis! 🔥

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if len(prices) < window:
        return []

    window_sum = sum(prices[:window])
    averages = [round(window_sum / window, 2)]

    for i in range(window, len(prices)):
        window_sum += prices[i] - prices[i - window]
        averages.append(round(window_sum / window, 2))

    return averages


def max_profit(prices: list[int]) -> int:
    """
    Best time to buy and sell stock.
    LeetCode #121 — sliding window variant!

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    min_price = float('inf')
    max_profit_val = 0

    for price in prices:
        min_price = min(min_price, price)
        max_profit_val = max(max_profit_val,
                             price - min_price)

    return max_profit_val


if __name__ == "__main__":
    print("=== Sliding Window Patterns ===")

    nums = [1, 12, -5, -6, 50, 3]
    print(f"Max avg (k=4): {max_average_subarray(nums, 4)}")

    s = "eceba"
    print(f"Longest k=2 distinct: {longest_substring_k_distinct(s, 2)}")

    print("\n=== ArthAI — Stock Moving Average ===")
    nifty_prices = [17500, 17650, 17420, 17800,
                    17950, 18100, 17900, 18200,
                    18350, 18500]
    ma_5 = moving_average_stocks(nifty_prices, 5)
    print(f"NIFTY prices: {nifty_prices}")
    print(f"5-day MA:     {ma_5}")

    profit = max_profit(nifty_prices)
    print(f"Best profit possible: ₹{profit}")
