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
