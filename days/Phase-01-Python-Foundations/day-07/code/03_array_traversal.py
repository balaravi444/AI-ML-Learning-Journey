# Problem 3 - Array Traversal and Searching
# Day 07 - Sets, Arrays & List Comprehensions

numbers = [10, 20, 30, 40, 50]
search = int(input("Enter number to search: "))
found = False
for num in numbers:
    if num == search:
        found = True
        break
if found:
    print(search, "found in array")
else:
    print(search, "not found in array")
