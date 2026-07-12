# Day 22 — DSA: Stacks & Queues 🚀

**Date:** 09 June 2026
**Time Spent:** (5 hours)
**Resource Used:** [LeetCode](https://leetcode.com/) | [CS50P](https://cs50.harvard.edu/python/)

---

## 📚 Topics Covered

- What is a Stack
- Stack operations — push, pop, peek
- What is a Queue
- Queue operations — enqueue, dequeue
- LIFO vs FIFO
- Implementing Stack and Queue in Python
- Monotonic Stack
- Real applications in ML

---

## 🔑 Stack — LIFO (Last In First Out)

```
Think of a stack of plates:
- Add plate on TOP   → push
- Remove from TOP    → pop
- Look at TOP plate  → peek

Last plate added = First plate removed!
```

```python
stack = []
stack.append(1)   # push → [1]
stack.append(2)   # push → [1, 2]
stack.append(3)   # push → [1, 2, 3]
stack.pop()       # pop  → [1, 2] returns 3
stack[-1]         # peek → 2
```

**So what? Why does this matter?**
Stacks are used in:
- Function call stack (how Python runs your code!)
- Undo/redo operations
- Browser back button
- Depth First Search (DFS) in graphs
- Backtracking algorithms in ML!

---

## 🔑 Queue — FIFO (First In First Out)

```
Think of a queue at a bank:
- Join at BACK   → enqueue
- Leave from FRONT → dequeue

First person joined = First person served!
```

```python
from collections import deque

queue = deque()
queue.append(1)     # enqueue → [1]
queue.append(2)     # enqueue → [1, 2]
queue.append(3)     # enqueue → [1, 2, 3]
queue.popleft()     # dequeue → [2, 3] returns 1
queue[0]            # peek    → 2
```

**So what? Why does this matter?**
Queues are used in:
- BFS (Breadth First Search)
- ML training data pipelines
- Message queues in production ML systems
- Task scheduling in ML training jobs!

---

## 📊 Stack vs Queue

| Feature | Stack | Queue |
|---------|-------|-------|
| Order | LIFO | FIFO |
| Add | push (top) | enqueue (back) |
| Remove | pop (top) | dequeue (front) |
| Use case | DFS, backtracking | BFS, pipelines |
| Python | list or deque | deque |

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | Implement Stack | Array/List | O(1) push/pop |
| 2 | Implement Queue | deque | O(1) enqueue/dequeue |
| 3 | Valid Parentheses | Stack | O(n) |
| 4 | Min Stack | Stack | O(1) min |
| 5 | Queue using Stacks | Two Stacks | Amortized O(1) |
| 6 | Next Greater Element | Monotonic Stack | O(n) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 20 | Valid Parentheses | Easy | Stack |
| 155 | Min Stack | Medium | Stack |
| 232 | Queue using Stacks | Easy | Two Stacks |
| 496 | Next Greater Element | Easy | Monotonic Stack |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Call stack — how Python executes your ML code!
def train_model():          # pushed to stack
    def preprocess():       # pushed to stack
        def normalize():    # pushed to stack
            pass            # popped
        normalize()         # popped
    preprocess()            # popped

# 2. Queue in ML data pipeline
from collections import deque

data_pipeline = deque()
data_pipeline.append("load_data")
data_pipeline.append("preprocess")
data_pipeline.append("train_model")
data_pipeline.append("evaluate")

while data_pipeline:
    step = data_pipeline.popleft()
    print(f"Running: {step}")

# 3. Monotonic stack in time series
# Finding next greater value — used in
# stock price analysis and anomaly detection!
```

---

## 💎 Important Realizations

1. **Python's function calls use a stack internally!**
   Every time you call a function — it's pushed to call stack.
   When function returns — it's popped!
   RecursionError happens when stack overflows!

2. **deque is better than list for queues**
   list.pop(0) → O(n) slow!
   deque.popleft() → O(1) fast!

3. **Monotonic stack solves many hard problems in O(n)**
   Problems that seem O(n²) can be solved O(n) with monotonic stack!

---

## ❌ Mistakes & Fixes

**Mistake 1 — Using list for queue:**
```python
# Wrong — O(n) for dequeue!
queue = []
queue.pop(0)    # ❌ slow!

# Correct — O(1)!
from collections import deque
queue = deque()
queue.popleft() # ✅ fast!
```

**Mistake 2 — Not checking empty stack:**
```python
# Wrong — crashes on empty stack!
top = stack.pop()  # ❌ IndexError!

# Correct — check first!
if stack:
    top = stack.pop()  # ✅
```

---

## 🎯 Next Goal

- Linked Lists
- Singly and Doubly Linked Lists
- Real applications in ML memory management

---

*Day 22 complete* 🔥
