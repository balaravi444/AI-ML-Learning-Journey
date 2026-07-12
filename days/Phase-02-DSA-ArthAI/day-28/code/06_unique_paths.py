"""
Day 28 — DSA: Dynamic Programming
Topic: Unique Paths — LeetCode #62
Date: 15 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: 2D Grid DP

Robot moving from top-left to bottom-right.
Can only move right or down.
How many unique paths?

State: dp[i][j] = paths to reach cell (i,j)
Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1]
Base: dp[0][j] = 1, dp[i][0] = 1

Time Complexity:  O(m * n)
Space Complexity: O(m * n) → O(n) optimized

Real World Connection:
    Grid DP used in:
    - Path planning for robotics AI!
    - Counting routes in network optimization ML!
    - Game AI pathfinding!
"""


def unique_paths(m: int, n: int) -> int:
    """
    Count unique paths from top-left to bottom-right.

    Args:
        m: Number of rows
        n: Number of columns

    Returns:
        Number of unique paths

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]


def unique_paths_obstacles(grid: list[list[int]]) -> int:
    """
    Unique paths with obstacles — LeetCode #63.
    0 = empty, 1 = obstacle.

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # Initialize first cell
    dp[0][0] = 1 if grid[0][0] == 0 else 0

    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] if grid[0][j] == 0 else 0

    # First column
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] if grid[i][0] == 0 else 0

    # Fill rest
    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 0:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]


if __name__ == "__main__":
    print("=== Unique Paths — LeetCode #62 ===")

    test_cases = [(3, 7), (3, 2), (1, 1)]
    for m, n in test_cases:
        result = unique_paths(m, n)
        print(f"Grid {m}x{n}: {result} paths")

    print("\n=== Unique Paths with Obstacles — LeetCode #63 ===")
    grids = [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1], [0, 0]]
    ]
    for grid in grids:
        result = unique_paths_obstacles(grid)
        print(f"Grid: {grid} → {result} paths")
