# Day 27 — DSA: Graphs — BFS & DFS 🚀

**Date:** 14 June 2026
**Time Spent:** (5 hours)
**Resource Used:** [LeetCode](https://leetcode.com/) | [CS50P](https://cs50.harvard.edu/python/)

---

## 📚 Topics Covered

- What is a Graph
- Graph Representations
- BFS — Breadth First Search
- DFS — Depth First Search
- Cycle Detection
- Topological Sort
- Connected Components
- Shortest Path
- Real applications in ML

---

## 🔑 What is a Graph?

A graph is a collection of nodes connected by edges!
1 --- 2
|     |
3 --- 4
     |
     5
Unlike trees — graphs can have:
- Cycles (A → B → C → A)
- Multiple paths between nodes
- Disconnected components
- Directed or undirected edges

**Types of Graphs:**
Undirected → edges have no direction (friendship)

Directed   → edges have direction (Twitter follow)

Weighted   → edges have weights (road distances)

Unweighted → all edges equal

Cyclic     → has cycles

Acyclic    → no cycles (DAG)

**So what? Why does this matter?**
Graphs are EVERYWHERE in AI/ML:
- Social network analysis → GNN (Graph Neural Networks)
- Knowledge graphs → powering ChatGPT's knowledge
- Recommendation systems → user-item graphs
- Fraud detection → transaction graphs
- Drug discovery → molecular graphs! 🔥

---

## 🔑 Graph Representations

### Adjacency List — Most Common!
```python
# Undirected graph
graph = {
    1: [2, 3],
    2: [1, 4],
    3: [1, 4],
    4: [2, 3, 5],
    5: [4]
}
```

### Adjacency Matrix
```python
# 1 means connected, 0 means not
matrix = [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0]
]
```

| | Adjacency List | Adjacency Matrix |
|--|---------------|-----------------|
| Space | O(V + E) | O(V²) |
| Check edge | O(degree) | O(1) |
| Find neighbors | O(degree) | O(V) |
| Best for | Sparse graphs | Dense graphs |

**Use adjacency list for ML graphs** —
real-world graphs are sparse (few connections)!

---

## 🔑 BFS — Breadth First Search

Explores level by level using a QUEUE!

```python
from collections import deque

def bfs(graph: dict, start: int) -> list:
    visited = set()
    queue = deque([start])
    visited.add(start)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order
```

**Properties:**
Time:  O(V + E)

Space: O(V)

Finds: SHORTEST path in unweighted graph!

Uses:  Queue (FIFO)
**So what? Why does this matter?**
BFS finds shortest path → used in:
- GPS navigation AI
- Social network degrees of separation
- Web crawlers for training data collection
- Level-order processing in neural networks!

---

## 🔑 DFS — Depth First Search

Explores as deep as possible using a STACK!

```python
def dfs(graph: dict, start: int) -> list:
    visited = set()
    order = []

    def explore(node: int) -> None:
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                explore(neighbor)

    explore(start)
    return order
```

**Properties:**

Time:  O(V + E)

Space: O(V) — recursion stack

Finds: All paths, cycles, components

Uses:  Stack (implicit via recursion)
**So what? Why does this matter?**
DFS is used in:
- Topological sort for ML pipeline ordering
- Finding connected components in data
- Maze solving in robotics AI
- Dependency resolution in ML systems!

---

## 🔑 BFS vs DFS

| | BFS | DFS |
|--|-----|-----|
| Data Structure | Queue | Stack/Recursion |
| Explores | Level by level | Deep first |
| Finds shortest path | ✅ Yes | ❌ No |
| Memory | O(width) | O(depth) |
| Use for | Shortest path | All paths, cycles |

---

## 🔑 Topological Sort

Order tasks so dependencies come first!

ML Pipeline example:

load_data → preprocess → train → evaluate → deploy
Represented as DAG:

load_data → preprocess

preprocess → train

train → evaluate

evaluate → deploy

**Algorithm:**
1. Find all nodes with in-degree 0 (no dependencies)
2. Process them → remove their edges
3. Repeat until all processed!

**So what? Why does this matter?**
Topological sort is used in:
- ML pipeline scheduling (which step runs first?)
- Neural network forward pass ordering
- Dependency management in ML systems
- Build systems for ML model deployment!

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | BFS traversal | Queue | O(V+E) |
| 2 | DFS traversal | Recursion | O(V+E) |
| 3 | Number of Islands | DFS/BFS | O(m*n) |
| 4 | Clone Graph | BFS | O(V+E) |
| 5 | Course Schedule | Topological Sort | O(V+E) |
| 6 | Shortest Path | BFS | O(V+E) |
| 7 | Connected Components | DFS | O(V+E) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 200 | Number of Islands | Medium | DFS/BFS |
| 133 | Clone Graph | Medium | BFS |
| 207 | Course Schedule | Medium | Topological Sort |
| 417 | Pacific Atlantic Water Flow | Medium | BFS/DFS |
| 695 | Max Area of Island | Medium | DFS |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Knowledge Graph — powering LLMs!
knowledge_graph = {
    "Python": ["Programming", "AI", "ML"],
    "AI": ["ML", "DL", "NLP", "Computer Vision"],
    "ML": ["Supervised", "Unsupervised", "RL"],
    "ChatGPT": ["NLP", "LLM", "Transformer"]
}

# BFS on knowledge graph = how LLMs reason!
def find_connection(graph, start, end):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

# 2. Graph Neural Networks (GNN)!
# GNNs process social networks, molecular graphs
# using BFS-like message passing between nodes!

# 3. Recommendation system — graph problem!
user_item_graph = {
    "Bala": ["Python Book", "ML Course", "DSA Book"],
    "Ravi": ["ML Course", "Deep Learning Book"],
}
# "Users who liked X also liked Y" = graph traversal!

# 4. ML Pipeline as DAG!
pipeline = {
    "load_data": ["preprocess"],
    "preprocess": ["feature_eng", "split_data"],
    "feature_eng": ["train_model"],
    "split_data": ["train_model"],
    "train_model": ["evaluate"],
    "evaluate": ["deploy"]
}
# Topological sort determines execution order!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Not tracking visited:**
```python
# Wrong — infinite loop on cycles!
def dfs(graph, node):
    print(node)
    for neighbor in graph[node]:
        dfs(graph, neighbor)  # ❌ cycles cause infinite loop!

# Correct — track visited!
def dfs(graph, node, visited=set()):
    visited.add(node)
    print(node)
    for neighbor in graph[node]:
        if neighbor not in visited:  # ✅
            dfs(graph, neighbor, visited)
```

**Mistake 2 — Using list for visited (slow!):**
```python
visited = []
if node not in visited:   # ❌ O(n) check!
    visited.append(node)

visited = set()
if node not in visited:   # ✅ O(1) check!
    visited.add(node)
```

**Mistake 3 — BFS with wrong data structure:**
```python
queue = []
queue.pop(0)      # ❌ O(n) — wrong!

from collections import deque
queue = deque()
queue.popleft()   # ✅ O(1) — correct!
```

---

## 💎 Important Realizations

1. **Graphs are the most powerful DS for AI**
   Social networks, knowledge bases, molecules —
   all represented as graphs!

2. **BFS always finds shortest path in unweighted graphs**
   This is why GPS uses BFS-like algorithms!

3. **Topological sort = ML pipeline ordering**
   Every ML system runs steps in topological order!

4. **Graph Neural Networks (GNNs) are the future of AI**
   They apply neural networks directly on graph structure!
   Used in drug discovery, social network analysis,
   recommendation systems at Google and Meta!

---

## 🎯 Next Goal

- Dynamic Programming
- Most powerful technique for optimization!
- Used in Reinforcement Learning and sequence models!

---

*Day 27 complete* 🔥
     
