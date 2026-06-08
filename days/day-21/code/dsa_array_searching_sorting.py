"""
Day 21 — DSA: Arrays, Searching & Sorting
Phase 2: Math & DSA for AI
AI/ML Learning Journey — Bala Ravi
https://github.com/balaravi444/AI-ML-Learning-Journey
"""

# ============================================================
# BIG O NOTATION — Quick Reference
# ============================================================
# O(1)      → constant      → arr[0]
# O(log n)  → logarithmic   → binary search
# O(n)      → linear        → one loop
# O(n²)     → quadratic     → loop inside loop
# O(n log n)→ linearithmic  → merge sort
# ============================================================


# ============================================================
# SECTION 1 — SEARCHING
# ============================================================

def linear_search(arr, target):
    """
    Linear Search — O(n)
    Check every element one by one.
    Works on unsorted arrays.
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i       # found! return index
    return -1              # not found


def binary_search(arr, target):
    """
    Binary Search — O(log n)
    MUST be sorted! Cuts array in half each step.
    1M elements → only ~20 steps!
    """
    left  = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return mid              # found!
        elif arr[mid] < target:
            left = mid + 1          # target is in RIGHT half
        else:
            right = mid - 1         # target is in LEFT half

    return -1                       # not found


# ============================================================
# SECTION 2 — SORTING
# ============================================================

def bubble_sort(arr):
    """
    Bubble Sort — O(n²)
    Compare neighbours, swap if wrong order.
    Biggest element 'bubbles' to end each pass.
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):    # last i elements already sorted
            if arr[j] > arr[j + 1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]   # Python swap!
    return arr


def selection_sort(arr):
    """
    Selection Sort — O(n²)
    Find minimum in remaining array, place at front.
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr):
    """
    Insertion Sort — O(n²) worst, O(n) best (nearly sorted)
    Like sorting playing cards in your hand.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]    # shift right to make space
            j -= 1
        arr[j + 1] = key           # place in correct position
    return arr


def merge_sort(arr):
    """
    Merge Sort — O(n log n) ⭐ BEST GENERAL PURPOSE SORT
    Divide into halves, sort each half, merge back.
    Always O(n log n) — guaranteed!
    """
    if len(arr) <= 1:
        return arr                       # base case

    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])        # sort left half (recursive)
    right = merge_sort(arr[mid:])        # sort right half (recursive)
    return _merge(left, right)


def _merge(left, right):
    """Helper function — merges two sorted arrays into one sorted array."""
    result = []
    i = j  = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])    # add any remaining left elements
    result.extend(right[j:])   # add any remaining right elements
    return result


# ============================================================
# SECTION 3 — INTERVIEW PROBLEMS (Easy)
# ============================================================

def find_max(arr):
    """
    Find maximum element — O(n)
    Track the biggest number seen so far.
    """
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val


def is_sorted(arr):
    """
    Check if array is sorted — O(n)
    Compare each element with the next one.
    Return False immediately if any pair is out of order.
    """
    for i in range(len(arr) - 1):    # NOTE: subtract INSIDE range()
        if arr[i] > arr[i + 1]:
            return False
    return True                       # NOTE: OUTSIDE the loop!


def reverse_array(arr):
    """
    Reverse array IN-PLACE — O(n) time, O(1) space
    Use two pointers moving from both ends toward middle.
    Better than arr[::-1] — no extra memory!
    """
    left  = 0
    right = len(arr) - 1

    while left < right:
        arr[left], arr[right] = arr[right], arr[left]  # swap!
        left  += 1
        right -= 1

    return arr


# ============================================================
# SECTION 4 — INTERVIEW PROBLEMS (Medium)
# ============================================================

def second_largest(arr):
    """
    Second Largest Element — O(n)
    Track two variables: first (largest) and second.
    Use float('-inf') as starting value so any number beats it.
    """
    first  = arr[0]
    second = float('-inf')    # negative infinity = smallest possible

    for num in arr:
        if num > first:
            second = first    # old largest becomes second
            first  = num      # new number becomes largest
        elif num > second and num != first:
            second = num      # update second if bigger (but not equal to first)

    return second


def find_pair(arr, target):
    """
    Find Pair with Given Sum — O(n) using Two-Pointer Technique ⭐
    REQUIRES sorted array!
    Move left pointer right (need bigger sum) or
          right pointer left (need smaller sum).

    Brute force = O(n²) — nested loops
    Two pointer  = O(n)  — MUCH BETTER!
    """
    left  = 0
    right = len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return True                   # found the pair!
        elif current_sum < target:
            left  += 1                    # sum too small → move left forward
        else:
            right -= 1                    # sum too big  → move right backward

    return False                          # no pair found


# ============================================================
# SECTION 5 — INTERVIEW PROBLEMS (Hard)
# ============================================================

def max_subarray(arr):
    """
    Maximum Subarray Sum — Kadane's Algorithm — O(n) 🔥
    FAANG FAVOURITE — asked at Google, Amazon, Microsoft!

    Key insight at each step:
        Should I EXTEND the current subarray? (current_sum + num)
        Or START FRESH from this number?       (num)
    Pick whichever is bigger!

    Example: [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    Answer = 6 → subarray [4, -1, 2, 1]
    """
    max_sum     = arr[0]    # best total sum seen so far
    current_sum = arr[0]    # running sum of current subarray

    for num in arr[1:]:
        # extend OR start fresh — pick the better option
        current_sum = max(num, current_sum + num)
        # update best if current is better
        max_sum     = max(max_sum, current_sum)

    return max_sum


# ============================================================
# MAIN — Run all tests
# ============================================================

if __name__ == "__main__":

    print("=" * 55)
    print("Day 21 — DSA: Arrays, Searching & Sorting")
    print("=" * 55)

    # --- Searching ---
    print("\n🔍 SEARCHING")
    nums = [10, 20, 30, 40, 50]
    print(f"  Linear Search (30):  index {linear_search(nums, 30)}")   # 2
    print(f"  Linear Search (99):  index {linear_search(nums, 99)}")   # -1
    print(f"  Binary Search (30):  index {binary_search(nums, 30)}")   # 2
    print(f"  Binary Search (99):  index {binary_search(nums, 99)}")   # -1

    # Binary Search trace on larger array
    big = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print(f"  Binary Search (70):  index {binary_search(big, 70)}")    # 6

    # --- Sorting ---
    print("\n📊 SORTING")
    print(f"  Bubble Sort:    {bubble_sort([64, 34, 25, 12, 22])}")
    print(f"  Selection Sort: {selection_sort([64, 25, 12, 22, 11])}")
    print(f"  Insertion Sort: {insertion_sort([12, 11, 13, 5, 6])}")
    print(f"  Merge Sort:     {merge_sort([38, 27, 43, 3, 9, 82, 10])}")

    # --- Easy Problems ---
    print("\n💪 EASY PROBLEMS")
    print(f"  find_max([3,7,1,9,4]):         {find_max([3, 7, 1, 9, 4])}")
    print(f"  is_sorted([1,2,3,4,5]):        {is_sorted([1, 2, 3, 4, 5])}")
    print(f"  is_sorted([1,3,2,4,5]):        {is_sorted([1, 3, 2, 4, 5])}")
    print(f"  reverse_array([1,2,3,4,5]):    {reverse_array([1, 2, 3, 4, 5])}")

    # --- Medium Problems ---
    print("\n🔥 MEDIUM PROBLEMS")
    print(f"  second_largest([3,7,1,9,4]):   {second_largest([3, 7, 1, 9, 4])}")
    print(f"  find_pair([2,7,11,15], 9):     {find_pair([2, 7, 11, 15], 9)}")
    print(f"  find_pair([2,7,11,15], 20):    {find_pair([2, 7, 11, 15], 20)}")

    # --- Hard Problems ---
    print("\n🏆 HARD PROBLEMS")
    print(f"  max_subarray([-2,1,-3,4,-1,2,1,-5,4]): {max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])}")
    # expected: 6 → [4, -1, 2, 1]

    print("\n" + "=" * 55)
    print("All tests passed! ✅")
    print("=" * 55)
