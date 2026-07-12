# Problem 4 - Student Marks Validation
# Day 09 - Exception Handling

try:
  marks = int(input("Enter marks (0-100): "))
  if marks < 0 or marks > 100:
    raise ValueError("Marks must be between 0 and 100")
  print("Valid marks:", marks)
except ValueError as e:
  print("Error:", e)
