"""
Day 36 — NumPy: Arrays & Operations
Topic: NumPy vs Python Speed Comparison
Date: 23 June 2026
Author: Bala Ravi

This file PROVES why NumPy is essential for ML!
Same operations — massive speed difference!
"""
import numpy as np
import time


def speed_comparison(n: int = 1_000_000) -> None:
    """
    Compare Python vs NumPy speed.

    Args:
        n: Number of elements to process
    """
    print(f"=== Speed Comparison (n={n:,}) ===\n")

    data_list = list(range(n))
    data_numpy = np.array(data_list)

    # Test 1 — Sum
    start = time.perf_counter()
    python_sum = sum(data_list)
    python_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_sum = np.sum(data_numpy)
    numpy_time = time.perf_counter() - start

    print(f"Sum of {n:,} numbers:")
    print(f"  Python: {python_time:.4f}s")
    print(f"  NumPy:  {numpy_time:.6f}s")
    print(f"  NumPy is {python_time/numpy_time:.0f}x faster!")

    # Test 2 — Element-wise multiplication
    data2 = list(range(n))
    arr2 = np.array(data2)

    start = time.perf_counter()
    python_result = [a * b for a, b in
                     zip(data_list, data2)]
    python_time = time.perf_counter() - start

    start = time.perf_counter()
    numpy_result = data_numpy * arr2
    numpy_time = time.perf_counter() - start

    print(f"\nElement-wise multiply {n:,} numbers:")
    print(f"  Python: {python_time:.4f}s")
    print(f"  NumPy:  {numpy_time:.6f}s")
    print(f"  NumPy is {python_time/numpy_time:.0f}x faster!")

    # Test 3 — Normalization
    start = time.perf_counter()
    min_val = min(data_list)
    max_val = max(data_list)
    normalized_python = [
        (x - min_val) / (max_val - min_val)
        for x in data_list]
    python_time = time.perf_counter() - start

    start = time.perf_counter()
    normalized_numpy = ((data_numpy - data_numpy.min()) /
                        (data_numpy.max() - data_numpy.min()))
    numpy_time = time.perf_counter() - start

    print(f"\nNormalization of {n:,} numbers:")
    print(f"  Python: {python_time:.4f}s")
    print(f"  NumPy:  {numpy_time:.6f}s")
    print(f"  NumPy is {python_time/numpy_time:.0f}x faster!")

    print(f"\n🔥 Conclusion: NumPy is consistently")
    print(f"   50-200x faster than pure Python!")
    print(f"   This is why ML is fast enough to be practical!")


def ml_operations_demo() -> None:
    """
    Show core ML operations using NumPy.
    These run inside every neural network!
    """
    print("\n=== Core ML Operations ===\n")
    np.random.seed(42)

    # Simulate a neural network layer
    batch_size = 32
    input_features = 10
    output_features = 5

    # Input data
    X = np.random.randn(batch_size, input_features)

    # Weights and bias
    W = np.random.randn(input_features,
                         output_features) * 0.01
    b = np.zeros(output_features)

    # Forward pass — matrix multiplication!
    Z = np.dot(X, W) + b

    # Activation function
    A = np.maximum(0, Z)  # ReLU

    print(f"Input shape:  {X.shape}")
    print(f"Weights shape: {W.shape}")
    print(f"Output shape: {Z.shape}")
    print(f"After ReLU:   {A.shape}")
    print(f"\nThis is literally what neural networks do!")
    print(f"np.dot(X, W) + b = EVERY neural network layer!")


if __name__ == "__main__":
    speed_comparison()
    ml_operations_demo()
