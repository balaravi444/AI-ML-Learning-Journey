# Problem 8 - Count Vowels in a String
# Day 01 - Python Fundamentals

text = input("Enter a string: ")
count = 0
for char in text:
    if char.lower() in "aeiou":
        count += 1
print("Number of vowels:", count)
