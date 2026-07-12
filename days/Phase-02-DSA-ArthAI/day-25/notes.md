# Day 25 — DSA: Recursion & Backtracking 🚀

**Date:** 12 June 2026
**Time Spent:** (4 hours)
**Resource Used:** [LeetCode](https://leetcode.com/) | [CS50P](https://cs50.harvard.edu/python/)

---

## 📚 Topics Covered

- Recursion deep dive
- Recursion vs Iteration
- Backtracking pattern
- Subsets generation
- Permutations
- Combination Sum
- N-Queens problem
- Memoization basics

---

## 🔑 What is Recursion Really?

```python
def solve(problem):
    # Base case — smallest version
    if is_simple(problem):
        return solve_directly(problem)
    
    # Recursive case — break into smaller
    smaller = break_down(problem)
    return combine(solve(smaller))
```

**The 3 Laws of Recursion:**
Must have a BASE CASE
Must MOVE TOWARD base case
Must CALL ITSELF
**So what? Why does this matter?**
Every Decision Tree splits recursively!
Every neural network layer calls forward() recursively!
Every file system traversal uses recursion!
Understanding recursion = understanding how AI thinks! 🔥

---

## 🔑 Recursion vs Iteration

```python
# Factorial — Iterative
def factorial_iter(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Factorial — Recursive
def factorial_rec(n: int) -> int:
    if n == 0:        # base case
        return 1
    return n * factorial_rec(n - 1)
```

| | Iterative | Recursive |
|--|-----------|-----------|
| Speed | Faster | Slower |
| Memory | O(1) | O(n) call stack |
| Readability | Sometimes complex | Often cleaner |
| Use when | Performance critical | Problem naturally recursive |

**So what? Why does this matter?**
In ML — recursive algorithms are used for:
- Building Decision Trees (split until leaf!)
- Searching hyperparameter spaces
- Implementing minimax in game AI
- Tree traversals in neural architectures!

---

## 🔑 What is Backtracking?

Backtracking = Try something → If it fails → Undo → Try next option!

Think of it like a maze:

→ Go right

→ Dead end!

← Go back (backtrack!)

→ Go left

→ Found exit! ✅
**Template for every backtracking problem:**
```python
def backtrack(state, choices):
    # Base case — found solution
    if is_solution(state):
        result.append(state.copy())
        return
    
    for choice in choices:
        # Make choice
        state.append(choice)
        
        # Recurse
        backtrack(state, remaining_choices)
        
        # UNDO choice (backtrack!)
        state.pop()
```

**So what? Why does this matter?**
Backtracking is used in:
- Hyperparameter search in ML
- Neural Architecture Search (NAS)
- Constraint satisfaction in AI planning
- Game playing AI (chess, go!)

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | Fibonacci — Memoization | Recursion + Cache | O(n) |
| 2 | Generate Subsets | Backtracking | O(2ⁿ) |
| 3 | Permutations | Backtracking | O(n!) |
| 4 | Combination Sum | Backtracking | O(2ⁿ) |
| 5 | Word Search | Backtracking + DFS | O(m*n*4^L) |
| 6 | N-Queens | Backtracking | O(n!) |
| 7 | Sudoku Solver | Backtracking | O(9^m) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 78 | Subsets | Medium | Backtracking |
| 46 | Permutations | Medium | Backtracking |
| 39 | Combination Sum | Medium | Backtracking |
| 79 | Word Search | Medium | Backtracking + DFS |
| 51 | N-Queens | Hard | Backtracking |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Decision Tree building — pure recursion!
def build_tree(data, depth=0):
    # Base case
    if depth == max_depth or is_pure(data):
        return LeafNode(predict(data))
    
    # Find best split
    feature, threshold = best_split(data)
    
    # Recursive calls!
    left = build_tree(data[data[feature] < threshold])
    right = build_tree(data[data[feature] >= threshold])
    
    return DecisionNode(feature, threshold, left, right)

# 2. Hyperparameter search — backtracking!
def search_hyperparams(params, choices):
    if all_params_set(params):
        score = evaluate_model(params)
        return score
    
    best_score = 0
    for value in choices[current_param]:
        params[current_param] = value        # try
        score = search_hyperparams(params)   # recurse
        best_score = max(best_score, score)
        del params[current_param]            # backtrack!
    
    return best_score

# 3. Neural Architecture Search (NAS)!
# Google uses backtracking to find best neural architecture!
```

---

## 🔑 Memoization — Making Recursion Fast!

```python
# Without memoization — O(2ⁿ) — very slow!
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)  # calculates same values repeatedly!

# With memoization — O(n) — fast!
def fib_memo(n: int, memo: dict = {}) -> int:
    if n in memo:
        return memo[n]   # cache hit!
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

**So what? Why does this matter?**
Memoization IS Dynamic Programming!
DP is used in:
- Sequence alignment in bioinformatics ML
- Optimal policy learning in Reinforcement Learning
- CTC loss in speech recognition models!

---

## ❌ Mistakes & Fixes

**Mistake 1 — Forgetting to backtrack:**
```python
# Wrong — never undoes choice!
def permute(nums, current):
    if len(current) == len(nums):
        result.append(current)
        return
    for num in nums:
        current.append(num)
        permute(nums, current)
        # ❌ forgot to remove!

# Correct — always backtrack!
def permute(nums, current):
    if len(current) == len(nums):
        result.append(current.copy())  # copy!
        return
    for num in nums:
        if num not in current:
            current.append(num)
            permute(nums, current)
            current.pop()              # ✅ backtrack!
```

**Mistake 2 — Not copying result:**
```python
result.append(current)        # ❌ appends reference!
result.append(current.copy()) # ✅ appends copy!
```

---

## 💎 Important Realizations

1. **Every backtracking problem follows the same template**
   Make choice → Recurse → Undo choice

2. **Memoization turns O(2ⁿ) into O(n)**
   This is the foundation of Dynamic Programming!

3. **Recursion depth matters in production**
   Python default limit is 1000 — use iteration
   for deep recursion in ML pipelines!

4. **Backtracking is brute force with pruning**
   The art is knowing WHEN to prune (stop early)!

---

## 🎯 Next Goal

- Trees & Binary Search Trees
- Most important data structure for ML!
- Decision Trees ARE binary trees!

---

*Day 25 complete* 🔥


