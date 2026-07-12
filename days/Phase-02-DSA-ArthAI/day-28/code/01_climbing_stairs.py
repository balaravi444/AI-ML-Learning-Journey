"""
Day 28 — DSA: Dynamic Programming
Topic: Climbing Stairs — LeetCode #70
Date: 15 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Linear DP (exactly like Fibonacci!)

n stairs, can climb 1 or 2 steps at a time.
How many distinct ways to reach the top?

Time Complexity:  O(n)
Space Complexity: O(n) → can optimize to O(1)!

Real World Connection:
    This exact pattern appears in:
    - Counting paths in neural network architectures
    - Number of ways to reach a state in RL!
"""


def climb_stairs_recursive(n: int) -> int:
    """
    Recursive solution — O(2ⁿ) SLOW!
    Shows why we need DP.

    Time Complexity: O(2ⁿ)
    """
    if n <= 2:
        return n
    return (climb_stairs_recursive(n - 1) +
            climb_stairs_recursive(n - 2))


def climb_stairs_memo(n: int,
                      memo: dict = None) -> int:
    """
    Memoized solution — Top-Down DP.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 2:
        return n

    memo[n] = (climb_stairs_memo(n - 1, memo) +
               climb_stairs_memo(n - 2, memo))
    return memo[n]


def climb_stairs_dp(n: int) -> int:
    """
    Bottom-Up DP solution.

    State: dp[i] = ways to reach stair i
    Recurrence: dp[i] = dp[i-1] + dp[i-2]
    Base: dp[1]=1, dp[2]=2

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def climb_stairs_optimized(n: int) -> int:
    """
    Space-optimized DP — O(1) space!
    Only need last two values!

    Time Complexity: O(n)
    Space Complexity: O(1) ← best!
    """
    if n <= 2:
        return n

    prev2, prev1 = 1, 2

    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1


if __name__ == "__main__":
    print("=== Climbing Stairs — LeetCode #70 ===")

    for n in [1, 2, 3, 4, 5, 10, 20]:
        result = climb_stairs_optimized(n)
        print(f"n={n}: {result} ways")

    print("\n=== DP Table Visualization ===")
    n = 6
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    print(f"Stair: {list(range(n + 1))}")
    print(f"Ways:  {dp}")
