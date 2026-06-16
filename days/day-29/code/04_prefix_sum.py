"""
Day 29 — Advanced DSA Patterns
Topic: Prefix Sum Pattern
Date: 16 June 2026
Author: Bala Ravi

Prefix sum = precompute cumulative sums
for O(1) range queries!

Real World Connection:
    Cumulative portfolio returns (ArthAI!)
    Attention scores in Transformers!
    Feature aggregation in ML!
"""


def subarray_sum_equals_k(nums: list[int],
                           k: int) -> int:
    """
    Count subarrays summing to k.
    LeetCode #560

    Key insight:
        prefix[j] - prefix[i] = k
        → prefix[i] = prefix[j] - k
        Store prefix sums in hash map!

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    count = 0
    prefix_sum = 0
    seen = {0: 1}

    for num in nums:
        prefix_sum += num

        if prefix_sum - k in seen:
            count += seen[prefix_sum - k]

        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1

    return count


def range_sum_query(nums: list[int],
                    queries: list[tuple]) -> list[int]:
    """
    Answer multiple range sum queries in O(1) each.

    Time Complexity: O(n + q)
    Space Complexity: O(n)
    """
    prefix = [0] * (len(nums) + 1)
    for i, num in enumerate(nums):
        prefix[i + 1] = prefix[i] + num

    return [prefix[right + 1] - prefix[left]
            for left, right in queries]


def portfolio_cumulative_returns(
        monthly_returns: list[float]) -> dict:
    """
    Calculate portfolio metrics using prefix sum.
    DIRECTLY used in ArthAI! 🔥

    Args:
        monthly_returns: Monthly return percentages

    Returns:
        Portfolio analytics dictionary
    """
    # Cumulative value (starting with ₹1L)
    initial = 100000
    values = [initial]

    for ret in monthly_returns:
        values.append(values[-1] * (1 + ret / 100))

    total_return = ((values[-1] - initial) /
                    initial * 100)
    best_month = max(monthly_returns)
    worst_month = min(monthly_returns)
    avg_return = sum(monthly_returns) / len(monthly_returns)

    return {
        "initial_investment": f"₹{initial:,}",
        "final_value": f"₹{values[-1]:,.0f}",
        "total_return": f"{total_return:.1f}%",
        "best_month": f"{best_month:.1f}%",
        "worst_month": f"{worst_month:.1f}%",
        "avg_monthly": f"{avg_return:.2f}%",
        "values": [round(v) for v in values]
    }


if __name__ == "__main__":
    print("=== Prefix Sum Patterns ===")

    nums = [1, 1, 1, 2, 2, 3]
    k = 3
    print(f"Subarrays summing to {k}: "
          f"{subarray_sum_equals_k(nums, k)}")

    nums2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    queries = [(0, 4), (2, 7), (5, 9)]
    results = range_sum_query(nums2, queries)
    for (l, r), res in zip(queries, results):
        print(f"Sum[{l}:{r}] = {res}")

    print("\n=== ArthAI — Portfolio Analysis ===")
    nifty_returns = [1.2, -0.8, 2.1, 1.5,
                     -1.2, 3.0, 0.8, 1.9,
                     -0.5, 2.3, 1.1, 1.8]

    analytics = portfolio_cumulative_returns(nifty_returns)
    print("📊 NIFTY Portfolio Analysis (12 months):")
    for key, value in analytics.items():
        if key != "values":
            print(f"  {key}: {value}")
