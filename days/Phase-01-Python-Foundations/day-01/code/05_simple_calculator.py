# Problem 5 - Simple Calculator
# Day 01 - Python Fundamentals

a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
choice = input("Enter operation (+, -, *, /): ")
if choice == "+":
    print("Result =", a + b)
elif choice == "-":
    print("Result =", a - b)
elif choice == "*":
    print("Result =", a * b)
elif choice == "/":
    print("Result =", a / b)
else:
    print("Invalid operation")
