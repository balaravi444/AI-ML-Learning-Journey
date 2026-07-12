# Day 26 — DSA: Trees & Binary Search Trees 🚀

**Date:** 13 June 2026
**Time Spent:** (5 hours)
**Resource Used:** [LeetCode](https://leetcode.com/) | [CS50P](https://cs50.harvard.edu/python/)

---

## 📚 Topics Covered

- What is a Tree
- Binary Tree
- Binary Search Tree (BST)
- Tree Traversals — Inorder, Preorder, Postorder
- BFS — Level Order Traversal
- BST operations — insert, search, delete
- Balanced vs Unbalanced trees
- Real applications in ML

---

## 🔑 What is a Tree?

A tree is a hierarchical data structure:

     1          ← Root
    / \
   2   3        ← Internal nodes
  / \   \
 4   5   6      ← Leaf nodes

 Key terms:
 Root     → top node (no parent)

Leaf     → bottom node (no children)

Height   → longest path from root to leaf

Depth    → distance from root to node
**So what? Why does this matter?**
Decision Trees in ML ARE literally trees!
Every split in a Random Forest = a tree node!
HTML DOM is a tree — web scraping for ML data!
File systems are trees — data storage for ML! 🔥

---

## 🔑 Binary Tree

Each node has AT MOST 2 children:

```python
class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None
```

---

## 🔑 Tree Traversals

### Inorder — Left → Root → Right
```python
def inorder(root):
    if not root:
        return
    inorder(root.left)   # left
    print(root.val)      # root
    inorder(root.right)  # right
```
Result on BST: **sorted order!** 🎯

### Preorder — Root → Left → Right
```python
def preorder(root):
    if not root:
        return
    print(root.val)       # root first!
    preorder(root.left)
    preorder(root.right)
```
Used to: **copy/serialize a tree!**

### Postorder — Left → Right → Root
```python
def postorder(root):
    if not root:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)       # root last!
```
Used to: **delete a tree, evaluate expressions!**

### Level Order (BFS) — Level by level
```python
from collections import deque

def level_order(root):
    queue = deque([root])
    while queue:
        node = queue.popleft()
        print(node.val)
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
```
Used to: **find shortest path, level-wise processing!**

---

## 🔑 Binary Search Tree (BST)

**BST Property:** Left < Root < Right!
    5
   / \
  3   7
 / \ / \
2  4 6  8

Operations:
Search: O(log n) average, O(n) worst

Insert: O(log n) average, O(n) worst
**So what? Why does this matter?**
BST gives O(log n) search — same as binary search!
But BST can handle dynamic insertions/deletions!
Used in database indexes for fast ML data lookup!

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | Binary Tree Traversals | DFS/BFS | O(n) |
| 2 | Maximum Depth | DFS | O(n) |
| 3 | Invert Binary Tree | DFS | O(n) |
| 4 | Same Tree | DFS | O(n) |
| 5 | BST Insert & Search | BST | O(log n) |
| 6 | Validate BST | DFS | O(n) |
| 7 | Lowest Common Ancestor | DFS | O(n) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 104 | Maximum Depth of Binary Tree | Easy | DFS |
| 226 | Invert Binary Tree | Easy | DFS |
| 100 | Same Tree | Easy | DFS |
| 102 | Binary Tree Level Order | Medium | BFS |
| 98 | Validate Binary Search Tree | Medium | DFS |
| 235 | Lowest Common Ancestor of BST | Medium | BST |
| 108 | Sorted Array to BST | Medium | Recursion |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Decision Tree IS a binary tree!
class DecisionNode:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
        self.left = None   # ← same as TreeNode!
        self.right = None  # ← same as TreeNode!

# 2. Tree traversal = Decision Tree prediction!
def predict(node, sample):
    # Base case — leaf node
    if node.is_leaf:
        return node.prediction

    # Go left or right based on feature
    if sample[node.feature] <= node.threshold:
        return predict(node.left, sample)   # inorder style!
    return predict(node.right, sample)

# 3. Level order = how neural network layers work!
# Layer 1 (root) → Layer 2 → Layer 3 (leaves)
# Information flows level by level!

# 4. File system tree for loading ML datasets!
import os
def load_dataset(root_path: str) -> list:
    data = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.csv'):
                data.append(os.path.join(root, file))
    return data
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Not handling None:**
```python
# Wrong — crashes on empty tree!
def height(root):
    return 1 + max(height(root.left),
                   height(root.right))  # ❌ None.left crash!

# Correct — check None first!
def height(root):
    if not root:      # ✅ base case!
        return 0
    return 1 + max(height(root.left),
                   height(root.right))
```

**Mistake 2 — Confusing traversal orders:**
Inorder   → L, Root, R → gives SORTED for BST

Preorder  → Root, L, R → root comes FIRST

Postorder → L, R, Root → root comes LAST
---

## 💎 Important Realizations

1. **Decision Trees in Scikit-learn ARE binary trees!**
   When you call `tree.fit()` it builds a binary tree
   using the exact same TreeNode structure!

2. **Inorder traversal of BST = sorted array!**
   This is used to efficiently extract
   sorted features from a BST in ML pipelines!

3. **Tree height = model depth in ML!**
   `max_depth` parameter in RandomForestClassifier
   literally controls the height of each tree!

4. **BFS level order = how CNNs process layers!**
   Each convolutional layer = one level of the tree!

---

## 🎯 Next Goal

- Graphs — BFS & DFS
- Most powerful data structure for relationships!
- Used in knowledge graphs for AI!

---

*Day 26 complete* 🔥



Delete: O(log n) average, O(n) worst


 
