"""
Day 21 — DSA: Arrays, Searching & Sorting
Topic: Binary Search
Date: 08
June 2026
Author: Bala Ravi

Time Complexity:  O(log n) — cuts search space in half!
Space Complexity: O(1) iterative, O(log n) recursive

REQUIREMENT: Array MUST be sorted!

Real World Connection:
    Binary search is used in:
    - Database indexing
    - Finding optimal ML hyperparameters
    - Finding classification threshold
    - Git bisect for finding bugs!
"""


def binary_search(arr: list[int], target: int) -> int:
    """
    Search for target using binary search.
    REQUIRES sorted array!

    Args:
        arr: SORTED list of integers
        target: Value to find

    Returns:
        Index of target if found, -1 otherwise

    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1      # search right half
        else:
            right = mid - 1     # search left half

    return -1


def binary_search_recursive(
    arr: list[int],
    target: int,
    left: int,
    right: int
) -> int:
    """
    Recursive binary search.

    Args:
        arr: SORTED list of integers
        target: Value to find
        left: Left boundary index
        right: Right boundary index

    Returns:
        Index of target if found, -1 otherwise

    Time Complexity: O(log n)
    Space Complexity: O(log n) — recursive call stack
    """
    # Base case
    if left > right:
        return -1

    mid = (left + right) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


def find_insert_position(arr: list[int], target: int) -> int:
    """
    Find position where target should be inserted
    to keep array sorted. (LeetCode #35)

    Args:
        arr: Sorted list of integers
        target: Value to insert

    Returns:
        Index where target should be inserted

    Time Complexity: O(log n)
    """
    left, right = 0, len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid

    return left


if __name__ == "__main__":
    arr = [2, 3, 4, 10, 40, 50, 60, 70, 80, 90]

    print("=== Binary Search ===")
    print(f"Array: {arr}")

    # Iterative
    target = 10
    result = binary_search(arr, target)
    print(f"\nIterative: Found {target} at index {result}")

    # Recursive
    result = binary_search_recursive(arr, target, 0, len(arr) - 1)
    print(f"Recursive: Found {target} at index {result}")

    # Not found
    target = 5
    result = binary_search(arr, target)
    print(f"\nSearch for {target}: {result}")

    # Insert position
    target = 35
    pos = find_insert_position(arr, target)
    print(f"\nInsert position for {target}: {pos}")

    print("\n=== Comparison: Linear vs Binary ===")
    import time

    large_arr = list(range(1000000))
    target = 999999

    # Linear search
    start = time.time()
    linear_result = -1
    for i in range(len(large_arr)):
        if large_arr[i] == target:
            linear_result = i
            break
    linear_time = time.time() - start

    # Binary search
    start = time.time()
    binary_result = binary_search(large_arr, target)
    binary_time = time.time() - start

    print(f"Linear Search: {linear_time:.6f} seconds")
    print(f"Binary Search: {binary_time:.6f} seconds")
    print(f"Binary is {linear_time/binary_time:.0f}x faster!")
