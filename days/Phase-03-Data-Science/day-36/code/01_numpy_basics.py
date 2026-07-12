"""
Day 36 — NumPy: Arrays & Operations
Topic: Array Creation and Basic Operations
Date: 23 June 2026
Author: Bala Ravi

NumPy = Numerical Python
Foundation of ALL data science and ML!

Real World Connection:
    Every ML dataset = NumPy array
    Every neural network weight = NumPy array
    Pandas, TensorFlow, PyTorch all built on NumPy!
"""
import numpy as np


def demonstrate_array_creation() -> None:
    """Show all ways to create NumPy arrays."""
    print("=== Array Creation ===\n")

    # From Python list
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"From list: {arr1}")
    print(f"  Shape: {arr1.shape}, dtype: {arr1.dtype}")

    # 2D array
    matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    print(f"\n2D array:\n{matrix}")
    print(f"  Shape: {matrix.shape}")
    print(f"  ndim: {matrix.ndim}")
    print(f"  size: {matrix.size}")

    # Special arrays
    print(f"\nZeros (3x4):\n{np.zeros((3, 4))}")
    print(f"\nOnes (2x3):\n{np.ones((2, 3))}")
    print(f"\nIdentity (3x3):\n{np.eye(3)}")

    # Ranges
    print(f"\narange(0, 10, 2): {np.arange(0, 10, 2)}")
    print(f"linspace(0, 1, 5): {np.linspace(0, 1, 5)}")

    # Random
    np.random.seed(42)
    print(f"\nRandom (3x3):\n{np.random.rand(3, 3).round(3)}")
    print(f"Normal (5,): {np.random.randn(5).round(3)}")


def demonstrate_indexing() -> None:
    """Show NumPy indexing and slicing."""
    print("\n=== Indexing & Slicing ===\n")

    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    print(f"Array:\n{arr}")
    print(f"\narr[0, 0] = {arr[0, 0]}")
    print(f"arr[2, 2] = {arr[2, 2]}")
    print(f"arr[0, :] = {arr[0, :]}")
    print(f"arr[:, 1] = {arr[:, 1]}")
    print(f"arr[1:, 1:] =\n{arr[1:, 1:]}")

    # Boolean indexing
    scores = np.array([45, 78, 92, 56, 88, 34, 95])
    print(f"\nScores: {scores}")
    print(f"Passed (>=60): {scores[scores >= 60]}")
    print(f"Failed (<60):  {scores[scores < 60]}")
    print(f"Top scorers (>90): {scores[scores > 90]}")


def demonstrate_operations() -> None:
    """Show NumPy vectorized operations."""
    print("\n=== Vectorized Operations ===\n")

    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    print(f"b / 10 = {b / 10}")
    print(f"np.sqrt(a) = {np.sqrt(a).round(3)}")

    # Statistical operations
    data = np.array([23, 45, 67, 12, 89, 34, 56])
    print(f"\nData: {data}")
    print(f"Mean:   {data.mean():.2f}")
    print(f"Std:    {data.std():.2f}")
    print(f"Min:    {data.min()}")
    print(f"Max:    {data.max()}")
    print(f"Sum:    {data.sum()}")
    print(f"Median: {np.median(data):.2f}")


def demonstrate_broadcasting() -> None:
    """Show NumPy broadcasting."""
    print("\n=== Broadcasting ===\n")

    # Feature normalization using broadcasting
    X = np.array([[1, 200, 0.5],
                  [2, 400, 0.8],
                  [3, 600, 0.3],
                  [4, 800, 0.9]])

    print(f"Raw features (age, salary, score):\n{X}")

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X_normalized = (X - mean) / std

    print(f"\nMean per feature: {mean}")
    print(f"Std per feature:  {std.round(2)}")
    print(f"\nNormalized:\n{X_normalized.round(3)}")
    print("\nNow all features are on the same scale!")
    print("This is Z-score normalization — used in EVERY ML model!")


if __name__ == "__main__":
    demonstrate_array_creation()
    demonstrate_indexing()
    demonstrate_operations()
    demonstrate_broadcasting()
