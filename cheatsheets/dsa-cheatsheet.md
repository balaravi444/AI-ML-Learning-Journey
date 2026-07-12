# 🔢 DSA Cheatsheet

Core data structures, algorithms, and Big-O — the stuff that shows up in every interview.

---

## Big-O Complexity (Cheat Table)

| Complexity | Name | Example |
|---|---|---|
| O(1) | Constant | array index access |
| O(log n) | Logarithmic | binary search |
| O(n) | Linear | single loop |
| O(n log n) | Linearithmic | merge sort, quicksort (avg) |
| O(n²) | Quadratic | nested loops, bubble sort |
| O(2ⁿ) | Exponential | recursive fibonacci (naive) |

## Time Complexity by Operation

| Structure | Access | Search | Insert | Delete |
|---|---|---|---|---|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1) | O(1) |
| Hash Map | — | O(1) avg | O(1) avg | O(1) avg |
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) |
| Stack/Queue | O(n) | O(n) | O(1) | O(1) |

---

## Arrays & Two Pointers

```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return [left, right]
        elif s < target:
            left += 1
        else:
            right -= 1
    return []
```

## Binary Search

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

## Sliding Window

```python
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

## Linked List

```python
class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def reverse_list(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev
```

## BFS / DFS (Graphs & Trees)

```python
from collections import deque

def bfs(graph, start):
    visited, queue = {start}, deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order

def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

## Sorting Algorithms

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]
```

## Dynamic Programming — Pattern

```python
# Bottom-up template
def dp_solve(n):
    dp = [0] * (n + 1)
    dp[0] = 1  # base case
    for i in range(1, n + 1):
        dp[i] = dp[i - 1] + (dp[i - 2] if i >= 2 else 0)
    return dp[n]
```

## Pattern Recognition Guide

| If the problem says... | Think... |
|---|---|
| "sorted array" | Binary search / two pointers |
| "subarray/substring" | Sliding window |
| "shortest path" | BFS |
| "all paths" | DFS / backtracking |
| "optimal/min/max with choices" | Dynamic programming |
| "k largest/smallest" | Heap |
| "next greater/smaller" | Monotonic stack |
| "connected components" | Union-Find / DFS |
