"""
Day 05 — Advanced Functions
Topic: Lambda Functions
Date: 23 May 2026
Author: Bala Ravi

Real World Connection:
    Pandas uses lambda with .apply() for data preprocessing!
    df['normalized'] = df['marks'].apply(lambda x: x / 100)
"""
from typing import Callable


def apply_operation(numbers: list[int], operation: Callable) -> list:
    """
    Apply a function to every element in a list.

    Args:
        numbers: List of integers to process
        operation: Function to apply to each element

    Returns:
        New list with operation applied to each element
    """
    return [operation(n) for n in numbers]


if __name__ == "__main__":
    # Basic lambda
    square = lambda x: x * x
    print(f"Square of 5: {square(5)}")

    # Lambda with apply_operation
    numbers = [1, 2, 3, 4, 5]

    squares = apply_operation(numbers, lambda x: x * x)
    print(f"Squares: {squares}")

    doubles = apply_operation(numbers, lambda x: x * 2)
    print(f"Doubles: {doubles}")

    normalized = apply_operation(numbers, lambda x: x / max(numbers))
    print(f"Normalized: {normalized}")
