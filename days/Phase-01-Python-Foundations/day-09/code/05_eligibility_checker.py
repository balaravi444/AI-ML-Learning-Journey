# Problem 5 - Exception -Based Eligibility Checker
# Day 09 - Exception Handling

try:
  marks = int(input("Enter the marks:"))
  age = int(input("Enter age: "))
  if marks < 0 or marks > 100:
    raise ValueError("Invalid marks entered")
  if age < 0:
    raise ValueError("Invalid age entered")
  if marks >= 75 and age >= 18:
    print("Eligible for AI Program")
  else:
    print("Not Eligible for AI Program")
except ValueError as e:
  print("Error:", e)
