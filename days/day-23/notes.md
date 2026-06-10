# Day 23 — DSA: Linked Lists 🚀

**Date:** 10 June 2026
**Time Spent:** (add your hours)
**Resource Used:** [LeetCode](https://leetcode.com/) | [CS50P](https://cs50.harvard.edu/python/)

---

## 📚 Topics Covered

- What is a Linked List
- Array vs Linked List
- Singly Linked List
- Doubly Linked List
- Linked List Operations
- Fast & Slow Pointer Technique
- Real applications in ML

---

## 🔑 What is a Linked List?

Array stores elements in contiguous memory:

[10] [20] [30] [40] [50]
0    1    2    3    4   ← indices

Linked List stores elements as NODES connected by pointers:
[10|→] → [20|→] → [30|→] → [40|→] → [50|None]
Each node has:
- **data** — the value
- **next** — pointer to next node

**So what? Why does this matter?**
Array:
✅ Fast access by index — O(1)
❌ Slow insert/delete in middle — O(n)
❌ Fixed size (in true arrays)
Linked List:
✅ Fast insert/delete — O(1) with pointer
❌ Slow access by index — O(n)
✅ Dynamic size — grows as needed

In ML — understanding linked lists helps you understand:
- How PyTorch builds computation graphs
- How gradient flows backward through layers
- How neural network layers are connected!

---

## 🔑 Node Structure

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None  # pointer to next node
```

---

## 🔑 Singly Linked List
Head → [1|→] → [2|→] → [3|→] → [4|None]

Operations:
append()      → add at end    — O(n)
prepend()     → add at start  — O(1)
delete()      → remove node   — O(n)
search()      → find value    — O(n)
reverse()     → reverse list  — O(n)
---

## 🔑 Fast & Slow Pointer Technique

Two pointers moving at different speeds!

```python
slow = head      # moves 1 step
fast = head      # moves 2 steps

while fast and fast.next:
    slow = slow.next        # 1 step
    fast = fast.next.next   # 2 steps
```

Used to:
- Find middle of linked list — O(n)
- Detect cycle in linked list — O(n)
- Find nth node from end — O(n)

**So what? Why does this matter?**
Fast & slow pointer is used in:
- Cycle detection in ML computation graphs
- Finding midpoint for merge sort
- Floyd's algorithm in optimization!

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | Implement Singly Linked List | Pointers | O(n) |
| 2 | Reverse Linked List | Iterative | O(n) |
| 3 | Find Middle of Linked List | Fast/Slow | O(n) |
| 4 | Detect Cycle | Floyd's Algorithm | O(n) |
| 5 | Merge Two Sorted Lists | Two Pointers | O(n) |
| 6 | Remove Nth Node from End | Fast/Slow | O(n) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 206 | Reverse Linked List | Easy | Iterative/Recursive |
| 876 | Middle of Linked List | Easy | Fast/Slow Pointer |
| 141 | Linked List Cycle | Easy | Floyd's Algorithm |
| 21 | Merge Two Sorted Lists | Easy | Two Pointers |
| 19 | Remove Nth Node from End | Medium | Fast/Slow Pointer |

---

## 🔗 How This Connects to AI/ML

```python
# 1. PyTorch computation graph is a linked list!
# Each layer points to the next layer:
# Input → Linear → ReLU → Linear → Output
# Gradient flows BACKWARD through this chain!

# 2. Linked list in NLP — token sequences
# Each token points to the next:
# "I" → "love" → "Python" → "and" → "AI"

# 3. Merge sort uses linked lists internally
# Python's Timsort merges sorted sublists —
# same concept as merging two sorted linked lists!

# 4. Detecting infinite loops in ML training
# If loss stops decreasing — might be a cycle!
# Floyd's algorithm detects cycles in O(n) O(1) space!
```

---

## 📊 Array vs Linked List — When to Use

| Operation | Array | Linked List |
|-----------|-------|-------------|
| Access by index | O(1) ✅ | O(n) ❌ |
| Search | O(n) | O(n) |
| Insert at beginning | O(n) ❌ | O(1) ✅ |
| Insert at end | O(1) ✅ | O(n) |
| Delete from middle | O(n) ❌ | O(1) ✅ |
| Memory | Contiguous | Scattered |

**Use Array when:** Random access needed (ML datasets!)
**Use Linked List when:** Frequent insert/delete (ML pipelines!)

---

## ❌ Mistakes & Fixes

**Mistake 1 — Losing head pointer:**
```python
# Wrong — lost the original head!
head = head.next  # ❌ can't go back!

# Correct — save reference first!
current = head    # ✅ keep head safe
current = current.next
```

**Mistake 2 — Not handling empty list:**
```python
# Wrong — crashes on empty list!
def get_length(head):
    count = 0
    while head:           # ❌ what if head is None?
        count += 1
        head = head.next

# Correct — None check built in!
def get_length(head):
    count = 0
    current = head
    while current:        # ✅ handles None!
        count += 1
        current = current.next
    return count
```

---

## 💎 Important Realizations

1. **Linked lists teach pointer thinking** — the foundation
   of understanding how ML frameworks manage memory!

2. **Fast & slow pointer is genius** — two pointers,
   one problem, O(n) time, O(1) space!

3. **Reversing a linked list in-place** teaches you to think
   about memory and references — critical for understanding
   backpropagation in neural networks!

4. **Most linked list problems follow a pattern:**
   - Dummy node for edge cases
   - Fast/slow pointers for cycle/middle
   - Two pointers for merging

---

## 🎯 Next Goal

- Hash Maps & Sets
- O(1) average lookup — the most powerful DSA tool!
- Used in EVERY optimization problem!

---

*Day 23 complete* 🔥

