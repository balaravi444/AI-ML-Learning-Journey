"""
Day 05 — Recursion
Topic: Recursive Factorial
Date: 23 May 2026
Author: Bala Ravi

Real World Connection:
    Factorial is used in:
    - Probability calculations in ML
    - Combinatorics for feature selection
    - Bayesian statistics
"""


def factorial(n: int) -> int:
    """
    Calculate factorial of n using recursion.

    Args:
        n: Non-negative integer

    Returns:
        Factorial of n (n!)

    Example:
        factorial(5) = 5 * 4 * 3 * 2 * 1 = 120
    """
    # Base case
    if n == 1 or n == 0:
        return 1

    # Recursive case
    return n * factorial(n - 1)


if __name__ == "__main__":
    for i in range(1, 8):
        print(f"{i}! = {factorial(i)}")
