# Day 29 — DSA: Advanced Patterns 🚀

**Date:** 16 June 2026
**Time Spent:** (3 hours)
**Resource Used:** [LeetCode](https://leetcode.com/)

---

## 📚 Topics Covered

- Sliding Window Pattern
- Two Pointers Pattern
- Binary Search on Answer
- Prefix Sum Pattern
- Monotonic Stack Pattern
- Fast & Slow Pointer Pattern
- Merge Intervals Pattern

---

## 🔑 Why Patterns Matter
Instead of solving each problem from scratch,

recognize the PATTERN and apply the template!
Problem → Recognize Pattern → Apply Template → Solve!
This is how top engineers solve LeetCode in minutes!

This is how ML engineers optimize data pipelines!
---

## 🔑 Pattern 1 — Sliding Window

**When to use:**
→ Subarray/substring problems

→ "Maximum/minimum in window of size k"

→ "Longest subarray with condition"

**Template:**
```python
def sliding_window(arr: list, k: int) -> int:
    left = 0
    result = 0
    window_state = {}   # track window contents

    for right in range(len(arr)):
        # Add arr[right] to window
        window_state[arr[right]] = \
            window_state.get(arr[right], 0) + 1

        # Shrink window if invalid
        while window_invalid(window_state):
            window_state[arr[left]] -= 1
            if window_state[arr[left]] == 0:
                del window_state[arr[left]]
            left += 1

        # Update result
        result = max(result, right - left + 1)

    return result
```

**Real ML Connection:**
```python
# Moving average in time series analysis!
def moving_average(prices: list, k: int) -> list:
    window_sum = sum(prices[:k])
    result = [window_sum / k]
    for i in range(k, len(prices)):
        window_sum += prices[i] - prices[i-k]
        result.append(window_sum / k)
    return result
# Used in stock price analysis for ArthAI! 🔥
```

---

## 🔑 Pattern 2 — Two Pointers

**When to use:**
→ Sorted array problems

→ "Find pair with sum X"

→ "Remove duplicates"

→ "Palindrome check"
**Template:**
```python
def two_pointers(arr: list) -> int:
    left, right = 0, len(arr) - 1
    result = 0

    while left < right:
        if condition_met(arr[left], arr[right]):
            result = update(result)
            left += 1
            right -= 1
        elif need_larger():
            left += 1
        else:
            right -= 1

    return result
```

---

## 🔑 Pattern 3 — Binary Search on Answer

**When to use:**
→ "Find minimum/maximum value that satisfies condition"

→ Answer has monotonic property

→ Can verify answer in O(n)
**Template:**
```python
def binary_search_answer(arr: list) -> int:
    left, right = min_possible, max_possible

    while left < right:
        mid = (left + right) // 2

        if can_achieve(arr, mid):
            right = mid        # try smaller
        else:
            left = mid + 1     # need larger

    return left
```

---

## 🔑 Pattern 4 — Prefix Sum

**When to use:**
→ Range sum queries

→ "Sum of subarray from i to j"

→ Multiple queries on same array

**Template:**
```python
def prefix_sum(arr: list) -> list:
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i+1] = prefix[i] + arr[i]
    return prefix

# Range sum in O(1)!
def range_sum(prefix, left, right):
    return prefix[right+1] - prefix[left]
```

**Real ML Connection:**
```python
# Cumulative returns in portfolio analysis!
# This is DIRECTLY used in ArthAI! 🔥
def cumulative_returns(daily_returns: list) -> list:
    prefix = [1.0]
    for r in daily_returns:
        prefix.append(prefix[-1] * (1 + r/100))
    return prefix
```

---

## 🔑 Pattern 5 — Monotonic Stack

**When to use:**
→ "Next greater/smaller element"

→ "Previous greater/smaller element"

→ Histogram problems

**Template:**
```python
def monotonic_stack(arr: list) -> list:
    stack = []  # stores indices
    result = [-1] * len(arr)

    for i, num in enumerate(arr):
        # Pop while stack top is smaller (for NGE)
        while stack and arr[stack[-1]] < num:
            idx = stack.pop()
            result[idx] = num
        stack.append(i)

    return result
```

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | Maximum Average Subarray | Sliding Window | O(n) |
| 2 | Longest Substring K Distinct | Sliding Window | O(n) |
| 3 | 3Sum | Two Pointers | O(n²) |
| 4 | Container With Most Water | Two Pointers | O(n) |
| 5 | Koko Eating Bananas | Binary Search | O(n log n) |
| 6 | Subarray Sum Equals K | Prefix Sum | O(n) |
| 7 | Largest Rectangle Histogram | Monotonic Stack | O(n) |
| 8 | Merge Intervals | Sorting + Greedy | O(n log n) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 643 | Maximum Average Subarray | Easy | Sliding Window |
| 15 | 3Sum | Medium | Two Pointers |
| 11 | Container With Most Water | Medium | Two Pointers |
| 875 | Koko Eating Bananas | Medium | Binary Search |
| 560 | Subarray Sum Equals K | Medium | Prefix Sum |
| 84 | Largest Rectangle Histogram | Hard | Monotonic Stack |
| 56 | Merge Intervals | Medium | Sort + Merge |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Sliding Window → Stock price analysis (ArthAI!)
def max_profit_window(prices: list,
                      window: int) -> float:
    """Find best investment window."""
    max_avg = sum(prices[:window]) / window
    window_sum = sum(prices[:window])
    for i in range(window, len(prices)):
        window_sum += prices[i] - prices[i-window]
        max_avg = max(max_avg, window_sum/window)
    return max_avg

# 2. Prefix Sum → Cumulative portfolio returns!
def portfolio_growth(investments: list[float],
                     returns: list[float]) -> list:
    """Track portfolio value over time."""
    value = [investments[0]]
    for i in range(1, len(returns)):
        value.append(value[-1] * (1 + returns[i]/100))
    return value

# 3. Binary Search → Optimal SIP amount!
def find_min_sip(target_corpus: float,
                 years: int,
                 rate: float) -> float:
    """Find minimum monthly SIP to reach target."""
    left, right = 100, 100000
    while left < right:
        mid = (left + right) // 2
        corpus = calculate_corpus(mid, years, rate)
        if corpus >= target_corpus:
            right = mid
        else:
            left = mid + 1
    return left

# 4. Merge Intervals → Calendar-based financial planning!
def merge_financial_goals(goals: list[list]) -> list:
    """Merge overlapping financial goal periods."""
    goals.sort()
    merged = [goals[0]]
    for start, end in goals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

---

## 💎 Important Realizations

1. **Patterns reduce infinite problems to finite templates**
   Once you know 10 patterns — you can solve 500+ problems!

2. **Sliding Window is the foundation of time series ML**
   Every moving average, rolling std, rolling correlation
   in financial ML uses sliding window! ArthAI uses this!

3. **Binary search on answer is used in ML optimization**
   "Find minimum learning rate that converges"
   "Find minimum batch size that fits in GPU memory"
   Both are binary search on answer problems!

4. **Prefix sum makes range queries O(1)**
   Used in attention mechanisms in Transformers!
   Cumulative attention scores use prefix sum!

---

## 🎯 Next Goal

- Start building ArthAI! 🚀
- Apply DSA patterns to real financial calculations
- Build the foundation of a product that helps millions!

---

*Day 29 complete — DSA Phase Complete! 🏆🔥*



