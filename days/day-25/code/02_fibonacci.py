# Program 2 — Fibonacci & Memoization
# Day 25 — Recursion & Backtracking

import time

# ============================================
# 1. Naive Fibonacci — O(2^n) SLOW!
# ============================================
def fib_naive(n):
    if n <= 1:                          # base case
        return n
    return fib_naive(n-1) + fib_naive(n-2)  # two recursive calls!

print("=== Naive Fibonacci ===")
print(fib_naive(10))   # 55
print(fib_naive(20))   # 6765

# Why is this slow?
# fib(5) calls fib(4) and fib(3)
# fib(4) calls fib(3) and fib(2)   ← fib(3) calculated TWICE!
# fib(3) calls fib(2) and fib(1)   ← fib(2) calculated MANY times!
# Exponential duplicate work! O(2^n) 💀


# ============================================
# 2. Fibonacci with Memoization — O(n) FAST!
# ============================================
# Memoization = store results, don't recalculate!

def fib_memo(n, memo={}):
    if n in memo:                           # already calculated!
        return memo[n]
    if n <= 1:                              # base case
        return n

    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

print("\n=== Memoized Fibonacci ===")
print(fib_memo(10))    # 55
print(fib_memo(50))    # 12586269025 — instant! ✅
print(fib_memo(100))   # huge number — still instant! ✅


# ============================================
# 3. Speed comparison!
# ============================================
print("\n=== Speed Comparison ===")

start = time.time()
fib_naive(35)
end = time.time()
print(f"Naive fib(35)  : {end - start:.4f} seconds")

start = time.time()
fib_memo(35)
end = time.time()
print(f"Memo  fib(35)  : {end - start:.4f} seconds")

# Naive:  several seconds!  😫
# Memo:   0.0000 seconds!   ✅


# ============================================
# 4. Fibonacci sequence — first n numbers
# ============================================
def fib_sequence(n):
    sequence = []
    for i in range(n):
        sequence.append(fib_memo(i))
    return sequence

print("\n=== Fibonacci Sequence ===")
print(fib_sequence(10))   # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
