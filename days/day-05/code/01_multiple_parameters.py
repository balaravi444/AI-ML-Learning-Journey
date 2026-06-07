"""
Day 05 — Advanced Functions
Topic: Function with Multiple Parameters
Date: 23 May 2026
Author: Bala Ravi
"""


def add_numbers(a: int, b: int, c: int) -> int:
    """
    Add three numbers together.

    Args:
        a: First number
        b: Second number
        c: Third number

    Returns:
        Sum of all three numbers
    """
    return a + b + c


if __name__ == "__main__":
    result = add_numbers(10, 20, 30)
    print(f"Sum = {result}")
