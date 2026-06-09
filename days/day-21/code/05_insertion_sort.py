"""
Day 21 — DSA: Arrays, Searching & Sorting
Topic: Insertion Sort
Date: 08 June 2026
Author: Bala Ravi

Time Complexity:  O(n²) worst, O(n) best (nearly sorted)
Space Complexity: O(1)

Best use case: Nearly sorted arrays!
Python's Timsort uses insertion sort for small arrays!

Real World Connection:
    Python's built-in sort() uses Timsort which combines
    merge sort + insertion sort.
    Insertion sort runs when subarray size < 64 elements!
"""


def insertion_sort(arr: list[int]) -> list[int]:
    """
    Sort array using insertion sort.
    Like sorting playing cards in your hand!

    Args:
        arr: List of integers to sort

    Returns:
        Sorted list (ascending order)

    Time Complexity: O(n²) worst, O(n) best
    Space Complexity: O(1)
    """
    arr = arr.copy()

    for i in range(1, len(arr)):
        key = arr[i]        # current element
        j = i - 1

        # Move elements greater than key one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key    # place key in correct position

    return arr


if __name__ == "__main__":
    arr = [12, 11, 13, 5, 6]

    print("=== Insertion Sort ===")
    print(f"Original: {arr}")
    sorted_arr = insertion_sort(arr)
    print(f"Sorted:   {sorted_arr}")

    # Nearly sorted — insertion sort shines!
    nearly_sorted = [1, 2, 3, 5, 4]
    print(f"\nNearly sorted: {nearly_sorted}")
    print(f"Sorted: {insertion_sort(nearly_sorted)}")
