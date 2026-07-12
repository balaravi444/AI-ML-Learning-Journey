"""
Day 05 — Recursion
Topic: Recursive Countdown
Date: 23 May 2026
Author: Bala Ravi

Important Note:
    Base case is CRITICAL in recursion!
    Without it Python raises RecursionError after ~1000 calls.
    Python's default recursion limit: sys.getrecursionlimit() = 1000
"""


def countdown(n: int) -> None:
    """
    Count down from n to 0 using recursion.

    Args:
        n: Starting number for countdown

    Returns:
        None — prints each number

    Raises:
        RecursionError: If n is extremely large (>1000)
    """
    # Base case — CRITICAL! Without this → infinite recursion!
    if n == 0:
        print("🚀 Liftoff!")
        return

    print(n)
    countdown(n - 1)  # recursive call


if __name__ == "__main__":
    print("--- Countdown ---")
    countdown(5)
