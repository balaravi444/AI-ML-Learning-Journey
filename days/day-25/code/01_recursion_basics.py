# Program 1 — Recursion Basics
# Day 25 — Recursion & Backtracking

# ============================================
# 1. Factorial — Classic recursion!
# ============================================
# factorial(5) = 5 * 4 * 3 * 2 * 1 = 120

def factorial(n):
    # Base case — STOP here!
    if n == 0 or n == 1:
        return 1

    # Recursive case — call itself!
    return n * factorial(n - 1)

print("=== Factorial ===")
print(factorial(5))    # 120
print(factorial(0))    # 1
print(factorial(10))   # 3628800

# Trace for factorial(4):
# factorial(4) = 4 * factorial(3)
#                    3 * factorial(2)
#                        2 * factorial(1)
#                            return 1   ← base case!
#                        2 * 1 = 2
#                    3 * 2 = 6
#                4 * 6 = 24 ✅


# ============================================
# 2. Sum of N numbers
# ============================================
def sum_n(n):
    if n == 0:             # base case
        return 0
    return n + sum_n(n-1)  # recursive case

print("\n=== Sum of N ===")
print(sum_n(5))    # 15  (5+4+3+2+1)
print(sum_n(10))   # 55


# ============================================
# 3. Power function — x^n
# ============================================
def power(x, n):
    if n == 0:                    # anything^0 = 1
        return 1
    return x * power(x, n - 1)   # x^n = x * x^(n-1)

print("\n=== Power ===")
print(power(2, 10))   # 1024
print(power(3, 4))    # 81
print(power(5, 0))    # 1


# ============================================
# 4. Count down using recursion
# ============================================
def countdown(n):
    if n < 0:          # base case
        return
    print(n)
    countdown(n - 1)   # recursive case

print("\n=== Countdown ===")
countdown(5)           # 5 4 3 2 1 0
