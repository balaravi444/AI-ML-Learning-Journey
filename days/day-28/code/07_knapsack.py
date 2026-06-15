"""
Day 28 — DSA: Dynamic Programming
Topic: 0/1 Knapsack Problem
Date: 15 June 2026
Author: Bala Ravi

Classic DP problem — foundation of many others!

State: dp[i][w] = max value using first i items with weight w
Recurrence:
    if weights[i] > w: dp[i][w] = dp[i-1][w]
    else: dp[i][w] = max(dp[i-1][w],
                         dp[i-1][w-weights[i]] + values[i])

Time Complexity:  O(n * W)
Space Complexity: O(n * W) → O(W) optimized

Real World Connection:
    Knapsack = Resource allocation in ML!
    "Maximum model performance within compute budget"
    Used in:
    - Neural Architecture Search (NAS)!
    - Feature selection with memory constraints!
    - Portfolio optimization in finance AI!
    - ArthAI project — investment portfolio optimization!
"""


def knapsack(weights: list[int],
             values: list[int],
             capacity: int) -> int:
    """
    Solve 0/1 knapsack problem.

    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity

    Returns:
        Maximum value achievable

    Time Complexity: O(n * W)
    Space Complexity: O(n * W)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1)
          for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i - 1][w]

            # Take item i (if it fits)
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - weights[i - 1]] +
                    values[i - 1]
                )

    return dp[n][capacity]


def knapsack_optimized(weights: list[int],
                       values: list[int],
                       capacity: int) -> int:
    """
    Space-optimized knapsack — O(W) space!

    Time Complexity: O(n * W)
    Space Complexity: O(W)
    """
    dp = [0] * (capacity + 1)

    for i in range(len(weights)):
        # Traverse backwards to avoid reuse!
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w],
                        dp[w - weights[i]] + values[i])

    return dp[capacity]


if __name__ == "__main__":
    print("=== 0/1 Knapsack ===")

    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8

    result = knapsack(weights, values, capacity)
    print(f"Weights: {weights}")
    print(f"Values:  {values}")
    print(f"Capacity: {capacity}")
    print(f"Max value: {result}")

    print("\n=== ArthAI Connection — Investment Portfolio ===")
    # Items = investment options
    # Weight = risk score (1-10)
    # Value = expected return %
    # Capacity = max risk tolerance

    investments = [
        ("Fixed Deposit", 1, 6),
        ("Debt Mutual Fund", 2, 8),
        ("Balanced Fund", 4, 12),
        ("Equity Fund", 6, 15),
        ("Direct Stocks", 8, 20),
        ("Crypto", 10, 30)
    ]

    names = [i[0] for i in investments]
    risks = [i[1] for i in investments]
    returns = [i[2] for i in investments]
    max_risk = 12  # moderate risk appetite

    max_return = knapsack(risks, returns, max_risk)
    print(f"Risk tolerance: {max_risk}/30")
    print(f"Max expected return: {max_return}%")
    print("\nThis is how ArthAI will optimize your portfolio! 🚀")
