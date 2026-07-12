# Problem 2 - Count Vowels
# Day 06 - Python Data Structures

text = input("Enter a string: ")
count = 0
for char in text:
    if char.lower() in "aeiou":
        count += 1
print("Number of vowels:", count)
