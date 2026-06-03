# Program 3 - Using Our Custom Module
# Day 14 - Modules & Packages

import ai_utils

print("========================================")
print("       🤖 AI Utils Module Tester        ")
print("========================================")
print(f"Module Author  : {ai_utils.AUTHOR}")
print(f"Module Version : {ai_utils.VERSION}")

print("\n=== Test 1 — Calculate Accuracy ===")
accuracy = ai_utils.calculate_accuracy(85, 100)
print(f"Model got 85/100 correct")
print(f"Accuracy: {accuracy}%")

accuracy2 = ai_utils.calculate_accuracy(0, 0)
print(f"Edge case (0/0): {accuracy2}%")

print("\n=== Test 2 — Normalize ===")
normalized = ai_utils.normalize(75, 0, 100)
print(f"Mark 75 in range 0-100: {normalized}")

normalized2 = ai_utils.normalize(50, 0, 200)
print(f"Value 50 in range 0-200: {normalized2}")

print("\n=== Test 3 — Outlier Detection ===")
marks = [85, 82, 88, 90, 87, 91]
print(f"Dataset: {marks}")
print(f"Is 2 an outlier?  {ai_utils.is_outlier(2, marks)}")
print(f"Is 88 an outlier? {ai_utils.is_outlier(88, marks)}")
print(f"Is 100 an outlier? {ai_utils.is_outlier(100, marks)}")

print("\n=== Test 4 — Statistics ===")
data = [85, 82, 88, 90, 87, 91, 78, 95]
stats = ai_utils.get_statistics(data)
print(f"Dataset: {data}")
print(f"Mean : {stats['mean']}")
print(f"Min  : {stats['min']}")
print(f"Max  : {stats['max']}")
print(f"Std  : {stats['std']}")
print(f"Count: {stats['count']}")
