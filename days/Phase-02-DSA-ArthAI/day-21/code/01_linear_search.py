"""
Day 21 — DSA: Arrays, Searching & Sorting
Topic: Linear Search
Date: 08 June 2026
Author: Bala Ravi

Time Complexity:  O(n) — checks every element
Space Complexity: O(1) — no extra space needed

Real World Connection:
    Linear search is used when data is unsorted.
    In ML — searching for a specific sample in
    an unsorted dataset uses linear search!
"""


def linear_search(arr: list[int], target: int) -> int:
    """
    Search for target in array using linear search.

    Args:
        arr: List of integers to search in
        target: Value to find

    Returns:
        Index of target if found, -1 otherwise

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


def linear_search_all(arr: list[int], target: int) -> list[int]:
    """
    Find ALL occurrences of target in array.

    Args:
        arr: List of integers to search in
        target: Value to find

    Returns:
        List of all indices where target is found

    Time Complexity: O(n)
    Space Complexity: O(k) where k is number of occurrences
    """
    indices = []
    for i in range(len(arr)):
        if arr[i] == target:
            indices.append(i)
    return indices


if __name__ == "__main__":
    arr = [64, 25, 12, 22, 11, 25, 90]

    print("=== Linear Search ===")
    print(f"Array: {arr}")

    # Search for existing element
    target = 22
    result = linear_search(arr, target)
    if result != -1:
        print(f"Found {target} at index {result}")
    else:
        print(f"{target} not found")

    # Search for non-existing element
    target = 100
    result = linear_search(arr, target)
    if result != -1:
        print(f"Found {target} at index {result}")
    else:
        print(f"{target} not found")

    # Find all occurrences
    target = 25
    indices = linear_search_all(arr, target)
    print(f"\nAll occurrences of {target}: {indices}")
