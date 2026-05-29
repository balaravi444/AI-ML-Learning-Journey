# Problem 2 - Handling Invalid User Input
# Day 09 - Exception Handling

try:
  age = int(input("Enter your age:"))
  print("Your age is:", age)
except ValueError:
  print("Error: Please Enter a valid number")
finally:
  print("Input process complete")
            
