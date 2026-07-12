# Program 1 - Built-in Modules
# Day 14 - Modules & Packages

import math
import random

print("=== Math Module ===")
print(f"PI: {math.pi}")
print(f"Square root of 144: {math.sqrt(144)}")
print(f"2 to power 10: {math.pow(2, 10)}")
print(f"Floor of 3.9: {math.floor(3.9)}")
print(f"Ceil of 3.1: {math.ceil(3.1)}")

print("\n=== Random Module ===")
print(f"Random number (1-100): {random.randint(1, 100)}")

fruits = ["apple", "mango", "banana", "orange"]
print(f"Random fruit: {random.choice(fruits)}")

numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
print(f"Shuffled list: {numbers}")
