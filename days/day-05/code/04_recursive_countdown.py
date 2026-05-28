# Problem 4 - Recursive Countdown
# Day 05 - Advanced Functions & Recursion

def countdown(n):
    if n == 0:
        return
    print(n)
    countdown(n - 1)

countdown(5)
