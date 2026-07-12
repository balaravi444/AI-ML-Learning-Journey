# Problem 4 - Grade Calculator
# Day 03 - Control Flow

marks = int(input("enter the marks: "))
if marks >= 90:
  print("A Grade")
elif marks >= 75:
  print("B Grade")
elif marks >= 50:
  print("C Grade")
elif marks >= 35:
  print("Pass")
else:
  print("Fail")
