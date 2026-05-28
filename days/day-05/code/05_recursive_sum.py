# Problem 5 - Recursive Sum of First N Numbers
# Day 05 - Advanced Functions & Recursion

def sum_n(n):
    if n == 0:
        return 0
    return n + sum_n(n - 1)

print("Sum =", sum_n(5))
