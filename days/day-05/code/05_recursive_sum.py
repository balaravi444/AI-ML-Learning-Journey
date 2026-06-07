"""
Day 05 — Recursion
Topic: Recursive Sum of First N Numbers
Date: 23 May 2026
Author: Bala Ravi
"""


def sum_n(n: int) -> int:
    """
    Calculate sum of first n numbers using recursion.

    Args:
        n: Upper limit of sum

    Returns:
        Sum of numbers from 1 to n

    Example:
        sum_n(5) = 5 + 4 + 3 + 2 + 1 = 15
    """
    # Base case
    if n == 0:
        return 0

    # Recursive case
    return n + sum_n(n - 1)


if __name__ == "__main__":
    print(f"Sum of first 5:  {sum_n(5)}")
    print(f"Sum of first 10: {sum_n(10)}")
    print(f"Sum of first 100: {sum_n(100)}")
