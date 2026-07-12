# Problem 1 -Safe Division Program
# Day 09 - Exception Handling

try:
  a = int(input("enter the first number: ""))
  b = int(input("enter the second number: ""))
  result = a / b
except ZeroDivisionError:
  print("Error: Cannot divide by zero")
except valueError:
  print("Error: Please enter valid numbers")
else:
  print("Result =", result
finally:
  print(Program finished")
  
