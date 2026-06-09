"""
Day 21 — DSA: Arrays, Searching & Sorting
Topic: Selection Sort
Date: 08 June 2026
Author: Bala Ravi

Time Complexity:  O(n²) always
Space Complexity: O(1)

How it works:
Find minimum → place at beginning → repeat for rest
"""


def selection_sort(arr: list[int]) -> list[int]:
    """
    Sort array using selection sort.
    Finds minimum element and places at correct position.

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list (ascending order)

    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    arr = arr.copy()
    n = len(arr)

    for i in range(n):
        # Find minimum in remaining unsorted part
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        # Swap minimum with first unsorted element
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr


if __name__ == "__main__":
    arr = [64, 25, 12, 22, 11]

    print("=== Selection Sort ===")
    print(f"Original: {arr}")
    sorted_arr = selection_sort(arr)
    print(f"Sorted:   {sorted_arr}")
