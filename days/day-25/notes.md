# Day 25 — Recursion & Backtracking 🔄

**Date:** 12 June 2026
**Time Spent:** (5 hours)
**Resource Used:** Python Docs, LeetCode, GeeksForGeeks

---

## 📚 Topics Covered

- What is Recursion
- Base Case & Recursive Case
- Call Stack — how recursion works in memory
- Recursion complexity analysis
- Types of Recursion — linear, tail, tree
- Classic recursion problems
- What is Backtracking
- Backtracking template
- Classic backtracking problems
- Recursion vs Iteration comparison
- Interview patterns

---

## 🔑 Core Concept — Recursion

**Recursion = a function that calls ITSELF!**

Every recursive function needs TWO things:
```
1. Base Case    →  when to STOP   (most important!)
2. Recursive Case →  call itself with smaller input
```

Without base case = infinite loop = stack overflow! 💀

---

## 🧠 How Call Stack Works

```python
factorial(4)
    └── 4 * factorial(3)
            └── 3 * factorial(2)
                    └── 2 * factorial(1)
                            └── return 1   ← base case!
                    └── 2 * 1 = 2
            └── 3 * 2 = 6
    └── 4 * 6 = 24
```

Each call waits for the next one to finish!
Stack builds UP then unwinds DOWN! 🔄

---

## 📊 Complexity of Common Recursive Problems

| Problem | Time | Space |
|---------|------|-------|
| Factorial | O(n) | O(n) |
| Fibonacci (naive) | O(2ⁿ) | O(n) |
| Fibonacci (memoized) | O(n) | O(n) |
| Binary Search | O(log n) | O(log n) |
| Merge Sort | O(n log n) | O(n) |
| Tower of Hanoi | O(2ⁿ) | O(n) |

---

## 🔑 Core Concept — Backtracking

**Backtracking = try → if wrong → undo → try next!**

```
Think of it like a maze:
→ Go forward
→ Hit dead end?
→ Come BACK (backtrack)
→ Try different path!
```

### Backtracking Template — MEMORIZE THIS!
```python
def backtrack(state, choices):
    if is_solution(state):        # base case — found answer!
        save_solution(state)
        return

    for choice in choices:
        if is_valid(choice):      # only try valid choices!
            make_choice(choice)   # choose
            backtrack(state, remaining_choices)  # explore
            undo_choice(choice)   # UN-choose (backtrack!)
```

---

## 💡 Recursion vs Iteration

| | Recursion | Iteration |
|--|-----------|-----------|
| Code | Clean, short | Verbose |
| Memory | O(n) stack space | O(1) |
| Speed | Slower | Faster |
| Use when | Tree/graph problems | Simple loops |
| Risk | Stack overflow | None |

---

## 🔥 Key Patterns for Interviews

```
Pattern 1 — Reduce by 1:     f(n) = f(n-1) + something
Pattern 2 — Divide by 2:     f(n) = f(n/2) + something
Pattern 3 — Two branches:    f(n) = f(n-1) + f(n-2)
Pattern 4 — Backtrack:       try → undo → try next
```

---

## 🔗 How This Connects to AI/ML

```python
# Decision Trees use recursion internally!
# Each node splits → recursive call on each branch

# Neural Network backpropagation = backtracking concept!
# Forward pass → calculate error → backtrack gradients

# Graph traversal (DFS) = recursion!
# Used in recommendation systems, knowledge graphs
```

---

## ✅ Day 25 Wins

- ✅ Understood base case vs recursive case
- ✅ Traced call stack manually
- ✅ Solved factorial, fibonacci, power
- ✅ Implemented recursive binary search
- ✅ Understood backtracking template
- ✅ Solved N-Queens and subset problems
- ✅ Analysed time & space complexity

---

## 🎯 Next Goal

- Day 26 — Linked Lists
- Singly & Doubly Linked Lists
- Insert, Delete, Reverse operations
- Floyd's cycle detection algorithm!

---

*Day 25 complete* 🔥
