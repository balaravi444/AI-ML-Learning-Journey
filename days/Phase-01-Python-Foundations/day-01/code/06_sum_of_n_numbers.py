# Problem 6 - Sum of First N Numbers
# Day 01 - Python Fundamentals

n = int(input("Enter a number: "))
total = 0
for i in range(1, n + 1):
    total = total + i
print("Sum =", total)
