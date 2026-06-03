# ai_utils.py

import math

VERSION = "1.0.0"

def calculate_accuracy(correct, total):
    if total == 0:
        return 0
    return (correct / total) * 100

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def euclidean_distance(x1, x2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

# main.py

import ai_utils

print("Accuracy:", ai_utils.calculate_accuracy(95, 100))
print("Normalized:", ai_utils.normalize(75, 0, 100))

point1 = [1, 2]
point2 = [4, 6]

print("Distance:", ai_utils.euclidean_distance(point1, point2))
