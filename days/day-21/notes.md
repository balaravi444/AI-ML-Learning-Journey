# Day 21 — DSA: Arrays, Searching & Sorting 🚀

**Date:** 09 June 2026
**Time Spent:** (5 hours)
**Resource Used:** [CS50P](https://cs50.harvard.edu/python/) | [LeetCode](https://leetcode.com/)

---

## 📚 Topics Covered

- What is an Array
- Array vs Python List
- Linear Search
- Binary Search
- Bubble Sort
- Selection Sort
- Insertion Sort
- Time Complexity — Big O Notation
- Space Complexity

---

## 🔑 What is an Array?

An array is a collection of elements stored in
contiguous memory locations — each accessible by index.

```python
# Python List = Dynamic Array
numbers = [10, 25, 8, 45, 30]
#           0   1   2   3   4  ← index
```

| Feature | Array | Python List |
|---------|-------|-------------|
| Size | Fixed | Dynamic |
| Type | Same type | Any type |
| Speed | Faster | Slightly slower |
| Memory | Contiguous | Flexible |

**So what? Why does this matter?**
NumPy arrays are TRUE arrays — fixed type, contiguous memory!
This is why NumPy is 100x faster than Python lists for ML!
Every ML dataset is stored as a NumPy array internally! 🔥

---

## ⏱️ Big O Notation — Time Complexity

Big O describes how fast an algorithm runs as input grows!

O(1)      → Constant  — always same speed
O(log n)  → Logarithmic — very fast
O(n)      → Linear    — grows with input
O(n log n)→ Linearithmic — good sorting
O(n²)     → Quadratic — slow for large input
O(2ⁿ)     → Exponential — very slow!

Speed (fast to slow):
O(1) > O(log n) > O(n) > O(n log n) > O(n²) > O(2ⁿ)

**So what? Why does this matter?**
In ML — you process MILLIONS of data points!
An O(n²) algorithm on 1 million rows = 1 trillion operations 😱
An O(n log n) algorithm on 1 million rows = 20 million operations ✅
Choosing the right algorithm can make ML 50,000x faster!

---

## 🔍 Searching Algorithms

### Linear Search — O(n)
```python
def linear_search(arr: list, target: int) -> int:
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

- Checks every element one by one
- Works on unsorted arrays
- Time: O(n) — slow for large arrays
- Space: O(1)

### Binary Search — O(log n)
```python
def binary_search(arr: list, target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

- Requires SORTED array!
- Cuts search space in HALF each time
- Time: O(log n) — very fast!
- Space: O(1)

**So what? Why does this matter?**
Binary search is used in:
- Database indexing (finding records fast)
- ML hyperparameter tuning (grid search)
- Finding optimal threshold in classification models!

---

## 📊 Sorting Algorithms

### Bubble Sort — O(n²)
Compare adjacent elements → swap if wrong order
Repeat until no swaps needed
- Time: O(n²) — slow!
- Space: O(1)
- Only use for small arrays or teaching purposes

### Selection Sort — O(n²)
Find minimum element → place at beginning
Repeat for remaining array
- Time: O(n²) — slow!
- Space: O(1)
- Slightly better than bubble sort in practice

### Insertion Sort — O(n²) worst, O(n) best
Take element → insert into correct position in sorted part
Like sorting playing cards in your hand!

- Time: O(n²) worst, O(n) best case
- Space: O(1)
- BEST for nearly sorted arrays!

### Python Built-in Sort — O(n log n)
```python
arr.sort()          # sorts in place
sorted(arr)         # returns new sorted list
```
- Uses Timsort algorithm internally
- Combination of merge sort + insertion sort
- ALWAYS use this in real code!

---

## 💻 Programs Practiced

| # | Problem | Algorithm | Complexity |
|---|---------|-----------|------------|
| 1 | Linear Search | Sequential scan | O(n) |
| 2 | Binary Search | Divide and conquer | O(log n) |
| 3 | Bubble Sort | Adjacent swaps | O(n²) |
| 4 | Selection Sort | Find minimum | O(n²) |
| 5 | Insertion Sort | Insert in place | O(n²) |
| 6 | Find duplicates in array | Hash set | O(n) |
| 7 | Two Sum problem | Hash map | O(n) |
| 8 | Maximum subarray | Kadane's algorithm | O(n) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 1 | Two Sum | Easy | Hash Map |
| 2 | Best Time to Buy Stock | Easy | Sliding Window |
| 3 | Contains Duplicate | Easy | Hash Set |
| 4 | Maximum Subarray | Medium | Kadane's Algorithm |

---

## 🔗 How This Connects to AI/ML

```python
# 1. NumPy arrays — TRUE arrays!
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
# Stored in contiguous memory — 100x faster than list!

# 2. Binary search in ML — finding optimal threshold
def find_threshold(probabilities: list, target_recall: float) -> float:
    """Binary search for optimal classification threshold"""
    thresholds = sorted(set(probabilities))
    left, right = 0, len(thresholds) - 1
    while left <= right:
        mid = (left + right) // 2
        # check recall at this threshold
        # adjust left/right accordingly
    return thresholds[mid]

# 3. Sorting in ML — finding top-k predictions
predictions = [0.9, 0.3, 0.7, 0.5, 0.8]
top_3 = sorted(enumerate(predictions),
               key=lambda x: x[1],
               reverse=True)[:3]
print(top_3)  # top 3 predictions with indices
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Binary search on unsorted array:**
```python
arr = [5, 2, 8, 1, 9]
binary_search(arr, 8)  # ❌ wrong result!

# Fix — sort first!
arr.sort()
binary_search(arr, 8)  # ✅ correct!
```

**Mistake 2 — Off by one in binary search:**
```python
# Wrong:
while left < right:   # ❌ misses last element!

# Correct:
while left <= right:  # ✅
```

---

## 💎 Important Realizations

1. **Always use Python's built-in sort** in real code —
   it's O(n log n) and written in optimized C!

2. **Binary search requires sorted data** — always check this first!

3. **O(n²) is acceptable for n < 1000** — but never for ML datasets
   which have millions of rows!

4. **Hash maps make O(n²) problems O(n)** — this is the most
   important DSA trick for interviews!

---

## 🎯 Next Goal

- Stacks & Queues
- LIFO vs FIFO
- Real applications in ML pipelines

---

*Day 21 complete* 🔥


