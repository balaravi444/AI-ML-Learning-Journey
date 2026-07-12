"""
Day 21 — DSA: Arrays, Searching & Sorting
Topic: Bubble Sort
Date: 08 June 2026
Author: Bala Ravi

Time Complexity:  O(n²) worst/average, O(n) best
Space Complexity: O(1)

Note: Never use in production — only for learning!
Use Python's built-in sort() which is O(n log n)
"""


def bubble_sort(arr: list[int]) -> list[int]:
    """
    Sort array using bubble sort algorithm.
    Compares adjacent elements and swaps if wrong order.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list (ascending order)

    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    arr = arr.copy()  # don't modify original!
    n = len(arr)

    for i in range(n):
        swapped = False  # optimization!

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        # If no swaps in this pass — already sorted!
        if not swapped:
            break

    return arr


if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]

    print("=== Bubble Sort ===")
    print(f"Original: {arr}")
    sorted_arr = bubble_sort(arr)
    print(f"Sorted:   {sorted_arr}")
