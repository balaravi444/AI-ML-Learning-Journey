"""
Day 25 — DSA: Recursion & Backtracking
Topic: Fibonacci with Memoization
Date: 12 June 2026
Author: Bala Ravi

Time Complexity:
    Without memo: O(2ⁿ) — exponential!
    With memo:    O(n)  — linear!

Space Complexity: O(n)

Real World Connection:
    Memoization IS the foundation of Dynamic Programming!
    DP is used in:
    - Sequence alignment in bioinformatics
    - Optimal policy in Reinforcement Learning
    - CTC loss in speech recognition!
"""
from functools import lru_cache
import time


def fib_recursive(n: int) -> int:
    """
    Fibonacci without memoization.
    SLOW — calculates same values repeatedly!

    Time Complexity: O(2ⁿ)
    """
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_memo(n: int, memo: dict = None) -> int:
    """
    Fibonacci with manual memoization.
    FAST — caches results!

    Args:
        n: Fibonacci number to calculate
        memo: Cache dictionary

    Returns:
        nth Fibonacci number

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]  # cache hit!

    if n <= 1:
        return n

    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


@lru_cache(maxsize=None)
def fib_lru(n: int) -> int:
    """
    Fibonacci using Python's lru_cache decorator.
    Cleanest solution!

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n <= 1:
        return n
    return fib_lru(n - 1) + fib_lru(n - 2)


def fib_iterative(n: int) -> int:
    """
    Fibonacci iterative — most efficient!
    Use this in production!

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 1:
        return n

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr


if __name__ == "__main__":
    print("=== Fibonacci Comparison ===")

    # Speed comparison
    n = 35

    start = time.time()
    result = fib_recursive(n)
    recursive_time = time.time() - start
    print(f"Recursive (n={n}): {result} | Time: {recursive_time:.4f}s")

    start = time.time()
    result = fib_memo(n)
    memo_time = time.time() - start
    print(f"Memoized  (n={n}): {result} | Time: {memo_time:.6f}s")

    start = time.time()
    result = fib_iterative(n)
    iter_time = time.time() - start
    print(f"Iterative (n={n}): {result} | Time: {iter_time:.6f}s")

    print(f"\nMemo is {recursive_time/memo_time:.0f}x faster than recursive!")

    print("\n=== First 15 Fibonacci Numbers ===")
    print([fib_iterative(i) for i in range(15)])
