# Day 28 — DSA: Dynamic Programming 🚀

**Date:** 15 June 2026
**Time Spent:** (add your hours)
**Resource Used:** [LeetCode](https://leetcode.com/) | [CS50P](https://cs50.harvard.edu/python/)

---

## 📚 Topics Covered

- What is Dynamic Programming
- Memoization (Top-Down DP)
- Tabulation (Bottom-Up DP)
- 1D DP Problems
- 2D DP Problems
- Classic DP Patterns
- Real applications in ML/AI

---

## 🔑 What is Dynamic Programming?

DP = Recursion + Memoization + Optimal Substructure
Three conditions for DP:

Overlapping subproblems — same subproblem solved multiple times
Optimal substructure — optimal solution built from optimal subsolutions
Can be broken into smaller subproblems

**Simple explanation:**
Remember solutions to subproblems

so you never solve the same thing twice!
Brute Force → solve everything from scratch

Memoization → store and reuse solutions

Tabulation  → build solution bottom-up

**So what? Why does this matter?**
DP is used in:
- Sequence alignment in bioinformatics ML
- Viterbi algorithm in speech recognition
- CTC loss in neural networks
- Optimal policy in Reinforcement Learning
- Edit distance in NLP spell checking! 🔥

---

## 🔑 Two Approaches

### Top-Down (Memoization)
```python
# Start from problem → break down → cache results
def fib(n: int, memo: dict = {}) -> int:
    if n in memo:
        return memo[n]      # use cached result!
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]
```

### Bottom-Up (Tabulation)
```python
# Start from base case → build up to answer
def fib(n: int) -> int:
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

| | Top-Down | Bottom-Up |
|--|----------|-----------|
| Approach | Recursive + cache | Iterative + table |
| Space | O(n) + call stack | O(n) |
| Speed | Slightly slower | Faster |
| Intuition | Easier to think | Harder to think |
| Use when | Complex state | Performance critical |

---

## 🔑 Classic DP Patterns

### Pattern 1 — Linear DP (1D)
dp[i] depends on dp[i-1] or dp[i-2]

Examples: Fibonacci, Climbing Stairs, House Robber

### Pattern 2 — Grid DP (2D)
dp[i][j] depends on dp[i-1][j] or dp[i][j-1]

Examples: Unique Paths, Minimum Path Sum

### Pattern 3 — String DP
dp[i][j] involves two strings/sequences

Examples: Longest Common Subsequence, Edit Distance

### Pattern 4 — Knapsack DP
Choose items with constraints

Examples: 0/1 Knapsack, Coin Change

### Pattern 5 — Interval DP
dp[i][j] covers interval from i to j

Examples: Burst Balloons, Matrix Chain Multiplication

---

## 🔑 How to Solve Any DP Problem
Step 1 → Define the state

What does dp[i] mean?
Step 2 → Find the recurrence relation

How does dp[i] relate to previous states?
Step 3 → Define base cases

What are the smallest subproblems?
Step 4 → Decide order of computation

Which states need to be computed first?
Step 5 → Extract answer

Which state contains the final answer?

---

## 💻 Programs Practiced

| # | Problem | Pattern | Complexity |
|---|---------|---------|------------|
| 1 | Climbing Stairs | Linear DP | O(n) |
| 2 | House Robber | Linear DP | O(n) |
| 3 | Coin Change | Knapsack DP | O(n*amount) |
| 4 | Longest Common Subsequence | String DP | O(m*n) |
| 5 | Unique Paths | Grid DP | O(m*n) |
| 6 | Edit Distance | String DP | O(m*n) |
| 7 | 0/1 Knapsack | Knapsack DP | O(n*W) |
| 8 | Maximum Product Subarray | Linear DP | O(n) |

---

## 🎯 LeetCode Problems Solved

| # | Problem | Difficulty | Pattern |
|---|---------|------------|---------|
| 70 | Climbing Stairs | Easy | Linear DP |
| 198 | House Robber | Easy | Linear DP |
| 322 | Coin Change | Medium | Knapsack DP |
| 1143 | Longest Common Subsequence | Medium | String DP |
| 62 | Unique Paths | Medium | Grid DP |
| 72 | Edit Distance | Hard | String DP |
| 152 | Maximum Product Subarray | Medium | Linear DP |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Edit Distance — used in NLP spell checking!
# "kitten" → "sitting" = 3 edits
# Used in autocorrect, fuzzy search, NLP preprocessing!

def edit_distance(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],
                                    dp[i][j-1],
                                    dp[i-1][j-1])
    return dp[m][n]

# 2. Longest Common Subsequence — DNA sequence alignment!
# Used in bioinformatics ML to compare DNA sequences!
# Used in plagiarism detection NLP models!

# 3. Viterbi Algorithm (Hidden Markov Models)
# Uses DP to find most likely sequence of states
# Used in: speech recognition, NLP POS tagging!
# dp[t][s] = probability of being in state s at time t

# 4. Reinforcement Learning — Bellman Equation IS DP!
# V(s) = max_a [R(s,a) + γ * V(s')]
# Value function IS a DP table!
# Q-learning, Value Iteration all use DP!

# 5. CTC Loss in Speech Recognition
# Uses DP to align audio frames with text labels!
# Powers Google Voice, Siri, Alexa!

# 6. Coin Change → Portfolio Optimization!
# "Minimum coins to make amount"
# = "Minimum investments to reach target return"!
```

---

## 🔑 Detailed Problem Walkthroughs

### Climbing Stairs (LeetCode #70)
Problem: n stairs, can climb 1 or 2 at a time.

How many ways to reach top?
State: dp[i] = ways to reach stair i

Recurrence: dp[i] = dp[i-1] + dp[i-2]

Base: dp[1] = 1, dp[2] = 2
Looks exactly like Fibonacci! 🎯

### House Robber (LeetCode #198)
Problem: Rob houses, can't rob adjacent ones.

Maximize money robbed.
State: dp[i] = max money from first i houses

Recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i])

Base: dp[0] = nums[0], dp[1] = max(nums[0], nums[1])
Real ML connection: Feature selection!

"Choose features (houses) that don't conflict"

### Coin Change (LeetCode #322)

Problem: Minimum coins to make amount.
State: dp[i] = min coins to make amount i

Recurrence: dp[i] = min(dp[i - coin] + 1) for each coin

Base: dp[0] = 0
Real ML connection: Portfolio optimization!

"Minimum investments to reach target return"

### Longest Common Subsequence (LeetCode #1143)
Problem: Find LCS of two strings.
State: dp[i][j] = LCS of s1[:i] and s2[:j]

Recurrence:

if s1[i] == s2[j]: dp[i][j] = dp[i-1][j-1] + 1

else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
Real ML connection: DNA alignment, plagiarism detection!
---

## ❌ Mistakes & Fixes

**Mistake 1 — Wrong base case:**
```python
# Wrong — missing base cases!
def climb_stairs(n):
    dp = [0] * (n + 1)
    for i in range(n + 1):
        dp[i] = dp[i-1] + dp[i-2]  # ❌ dp[-1] is wrong!

# Correct — set base cases first!
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1   # ✅ base case 1
    dp[2] = 2   # ✅ base case 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

**Mistake 2 — Wrong recurrence direction:**
```python
# Wrong — processing in wrong order!
for i in range(amount + 1):
    for coin in coins:
        if coin <= i:
            dp[i] = min(dp[i], dp[coin - i] + 1)  # ❌

# Correct — subtract coin from amount!
for i in range(amount + 1):
    for coin in coins:
        if coin <= i:
            dp[i] = min(dp[i], dp[i - coin] + 1)  # ✅
```

**Mistake 3 — Forgetting to copy 2D array:**
```python
# Wrong — modifying original!
dp = [[0] * n] * m   # ❌ all rows are same object!

# Correct — independent rows!
dp = [[0] * n for _ in range(m)]  # ✅
```

---

## 💎 Important Realizations

1. **DP = Smart Brute Force**
   You still explore all possibilities —
   but remember results so you never repeat!

2. **Bellman Equation in RL IS DP!**
   V(s) = max[R + γV(s')] is literally
   a recurrence relation — the heart of DP!

3. **Edit Distance powers autocorrect in every keyboard**
   The Levenshtein distance you implement today
   is the same algorithm running in your phone
   right now when you type! 🤯

4. **Every DP problem has a visual table**
   Drawing the dp table before coding
   makes the solution obvious!
   Always draw first — code second!

5. **Space optimization is important for ML**
   Many DP solutions use O(n²) space
   but can be optimized to O(n)
   This matters when processing large ML datasets!

---

## 🗺️ DP Problem Roadmap---

## ❌ Mistakes & Fixes

**Mistake 1 — Wrong base case:**
```python
# Wrong — missing base cases!
def climb_stairs(n):
    dp = [0] * (n + 1)
    for i in range(n + 1):
        dp[i] = dp[i-1] + dp[i-2]  # ❌ dp[-1] is wrong!

# Correct — set base cases first!
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1   # ✅ base case 1
    dp[2] = 2   # ✅ base case 2
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

**Mistake 2 — Wrong recurrence direction:**
```python
# Wrong — processing in wrong order!
for i in range(amount + 1):
    for coin in coins:
        if coin <= i:
            dp[i] = min(dp[i], dp[coin - i] + 1)  # ❌

# Correct — subtract coin from amount!
for i in range(amount + 1):
    for coin in coins:
        if coin <= i:
            dp[i] = min(dp[i], dp[i - coin] + 1)  # ✅
```

**Mistake 3 — Forgetting to copy 2D array:**
```python
# Wrong — modifying original!
dp = [[0] * n] * m   # ❌ all rows are same object!

# Correct — independent rows!
dp = [[0] * n for _ in range(m)]  # ✅
```

---

## 💎 Important Realizations

1. **DP = Smart Brute Force**
   You still explore all possibilities —
   but remember results so you never repeat!

2. **Bellman Equation in RL IS DP!**
   V(s) = max[R + γV(s')] is literally
   a recurrence relation — the heart of DP!

3. **Edit Distance powers autocorrect in every keyboard**
   The Levenshtein distance you implement today
   is the same algorithm running in your phone
   right now when you type! 🤯

4. **Every DP problem has a visual table**
   Drawing the dp table before coding
   makes the solution obvious!
   Always draw first — code second!

5. **Space optimization is important for ML**
   Many DP solutions use O(n²) space
   but can be optimized to O(n)
   This matters when processing large ML datasets!

---

## 🗺️ DP Problem Roadmap
Easy:

✅ Fibonacci

✅ Climbing Stairs

✅ House Robber

✅ Maximum Subarray (Kadane's)
Medium:

✅ Coin Change

✅ Longest Common Subsequence

✅ Unique Paths

✅ Maximum Product Subarray

✅ Word Break

✅ Partition Equal Subset Sum
Hard:

✅ Edit Distance

✅ Burst Balloons

✅ Regular Expression Matching

✅ Wildcard Matching

---

## 🎯 Next Goal

- DSA Pattern Review + LeetCode Practice
- Sliding Window, Two Pointers patterns
- Prepare for technical interviews!
- Start building ArthAI project!

---

*Day 28 complete* 🔥






