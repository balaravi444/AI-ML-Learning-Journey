# Program 2 - Custom AI Utility Module
# Day 14 - Modules & Packages
# This module contains real AI/ML utility functions!

import math

VERSION = "1.0.0"
AUTHOR  = "Bala Ravi"


def calculate_accuracy(correct, total):
    """
    Calculate model prediction accuracy.
    Same formula used in every ML model evaluation!
    """
    if total == 0:
        return 0.0
    return (correct / total) * 100


def normalize(number, min_val, max_val):
    """
    Min-Max Normalization.
    Converts any value to range between 0 and 1.
    Used in EVERY ML project before training!
    """
    if max_val == min_val:
        return 0.0
    return (number - min_val) / (max_val - min_val)


def is_outlier(value, numbers):
    """
    Check if a value is an outlier.
    Uses 2 standard deviations rule.
    Used in data cleaning before ML training!
    """
    if not numbers:
        return False
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std = math.sqrt(variance)
    return abs(value - mean) > 2 * std


def get_statistics(numbers):
    """
    Get basic statistics of a list.
    Returns mean, min, max, std deviation.
    """
    if not numbers:
        return None
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std = math.sqrt(variance)
    return {
        "mean"  : round(mean, 2),
        "min"   : min(numbers),
        "max"   : max(numbers),
        "std"   : round(std, 2),
        "count" : len(numbers)
    }
