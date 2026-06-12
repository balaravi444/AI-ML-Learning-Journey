# Program 3 — Recursive Binary Search
# Day 25 — Recursion & Backtracking

# ============================================
# Iterative Binary Search (Day 21 version)
# ============================================
def binary_search_iterative(arr, target):
    left  = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# ============================================
# Recursive Binary Search — cleaner!
# ============================================
def binary_search_recursive(arr, target, left, right):
    # Base case — not found!
    if left > right:
        return -1

    mid = (left + right) // 2

    # Found!
    if arr[mid] == target:
        return mid

    # Search right half
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)

    # Search left half
    else:
        return binary_search_recursive(arr, target, left, mid - 1)


# ============================================
# Test both!
# ============================================
numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print("=== Iterative Binary Search ===")
print(binary_search_iterative(numbers, 70))   # 6
print(binary_search_iterative(numbers, 99))   # -1

print("\n=== Recursive Binary Search ===")
print(binary_search_recursive(numbers, 70, 0, len(numbers)-1))   # 6
print(binary_search_recursive(numbers, 99, 0, len(numbers)-1))   # -1

# Complexity:
# Time  → O(log n)  — same as iterative!
# Space → O(log n)  — extra stack frames! (iterative is O(1))
