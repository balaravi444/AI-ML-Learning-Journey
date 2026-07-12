# Problem 3 - List Index Validation
# Day 09 - Exception Handling

numbers = [10, 20, 30, 40, 50]
try: 
  index = int(input("Enter index to access: "))
  print("Element:", numbers[index])
except IndexError:
  print("Error: Index out of range")
except ValudError:
  print("Error: Please enter a valid number")
  
