"""
Day 28 — DSA: Dynamic Programming
Topic: Coin Change — LeetCode #322
Date: 15 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Unbounded Knapsack DP

Find minimum coins to make amount.
Can use each coin unlimited times!

State: dp[i] = min coins to make amount i
Recurrence: dp[i] = min(dp[i-coin]+1) for each coin
Base: dp[0] = 0

Time Complexity:  O(amount * len(coins))
Space Complexity: O(amount)

Real World Connection:
    Coin Change = Portfolio Optimization!
    "Minimum number of investments to reach target"
    Used in financial planning algorithms!
    Also used in compiler optimization in ML systems!
"""


def coin_change(coins: list[int],
                amount: int) -> int:
    """
    Find minimum coins to make amount.

    Key insight:
        For each amount from 1 to target,
        try every coin and take minimum!

    Args:
        coins: Available coin denominations
        amount: Target amount

    Returns:
        Minimum coins needed, -1 if impossible

    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    """
    # Initialize with infinity (impossible)
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # base case — 0 coins for amount 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


def coin_change_ways(coins: list[int],
                     amount: int) -> int:
    """
    Count number of ways to make amount.
    (Different problem — counting not minimizing!)

    Time Complexity: O(amount * len(coins))
    Space Complexity: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1  # one way to make 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]


if __name__ == "__main__":
    print("=== Coin Change — LeetCode #322 ===")

    test_cases = [
        ([1, 5, 10, 25], 36),
        ([1, 2, 5], 11),
        ([2], 3),
        ([1], 0)
    ]

    for coins, amount in test_cases:
        result = coin_change(coins, amount)
        print(f"coins={coins}, amount={amount} → {result}")

    print("\n=== Coin Change Ways ===")
    coins = [1, 2, 5]
    amount = 5
    ways = coin_change_ways(coins, amount)
    print(f"Ways to make {amount} with {coins}: {ways}")

    print("\n=== Financial Planning Connection ===")
    # Minimum SIP plans to reach target
    sip_options = [500, 1000, 2500, 5000]
    target = 7500
    result = coin_change(sip_options, target)
    print(f"Min SIP plans for ₹{target}: {result}")
