"""
Day 28 — DSA: Dynamic Programming
Topic: House Robber — LeetCode #198
Date: 15 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Linear DP

Rob houses to maximize money.
Can't rob adjacent houses!

State: dp[i] = max money from first i houses
Recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i])

Time Complexity:  O(n)
Space Complexity: O(1) optimized

Real World Connection:
    House Robber pattern used in:
    - Feature selection (can't select adjacent features)
    - Portfolio optimization with constraints
    - Task scheduling without conflicts in ML pipelines!
"""


def rob(nums: list[int]) -> int:
    """
    Maximize money robbed from non-adjacent houses.

    Key insight:
        At each house, decide:
        1. Skip this house → take dp[i-1]
        2. Rob this house → take dp[i-2] + nums[i]
        Take the maximum!

    Args:
        nums: Money in each house

    Returns:
        Maximum money that can be robbed

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        current = max(prev1,           # skip house i
                      prev2 + nums[i]) # rob house i
        prev2 = prev1
        prev1 = current

    return prev1


def rob_with_table(nums: list[int]) -> int:
    """
    House robber with full DP table — for visualization.

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i - 1],
                    dp[i - 2] + nums[i])

    return dp[n - 1]


if __name__ == "__main__":
    print("=== House Robber — LeetCode #198 ===")

    test_cases = [
        ([1, 2, 3, 1], 4),
        ([2, 7, 9, 3, 1], 12),
        ([2, 1, 1, 2], 4)
    ]

    for nums, expected in test_cases:
        result = rob(nums)
        status = "✅" if result == expected else "❌"
        print(f"{status} {nums} → {result}")

    print("\n=== DP Table Visualization ===")
    nums = [2, 7, 9, 3, 1]
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    print(f"Houses: {nums}")
    print(f"DP:     {dp}")
    print(f"Max:    {dp[-1]}")
