"""
Day 37 — NumPy: Linear Algebra
Topic: Matrix Multiplication — The Heart of ML!
Date: 24 June 2026
Author: Bala Ravi

np.dot(A, B) or A @ B
This single operation powers ALL of deep learning!
"""
import numpy as np


def matrix_multiply_visual(
        A: np.ndarray,
        B: np.ndarray) -> None:
    """
    Show matrix multiplication step by step.

    Args:
        A: First matrix
        B: Second matrix
    """
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"Result shape: ({A.shape[0]}, {B.shape[1]})\n")

    C = A @ B

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            terms = [f"{A[i,k]}*{B[k,j]}"
                     for k in range(A.shape[1])]
            print(f"C[{i},{j}] = "
                  f"{' + '.join(terms)} = {C[i,j]}")

    print(f"\nResult:\n{C}")


def neural_network_layer(
        X: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        activation: str = "relu") -> np.ndarray:
    """
    Simulate a neural network layer.
    Just matrix multiplication + activation!

    Args:
        X: Input (batch_size, input_features)
        W: Weights (input_features, output_features)
        b: Bias (output_features,)
        activation: Activation function name

    Returns:
        Layer output (batch_size, output_features)
    """
    Z = X @ W + b

    if activation == "relu":
        return np.maximum(0, Z)
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-Z))
    elif activation == "tanh":
        return np.tanh(Z)
    elif activation == "softmax":
        exp_Z = np.exp(Z - Z.max(axis=1, keepdims=True))
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)
    return Z


def full_neural_network_demo() -> None:
    """
    Build and run a complete neural network!
    From scratch. Just NumPy. Just matrix multiplication.
    """
    np.random.seed(42)

    print("=== Complete Neural Network Demo ===\n")

    # Architecture
    # Input → 784 → 256 → 128 → 10
    batch_size = 32
    input_size = 784    # 28x28 pixels
    hidden1 = 256
    hidden2 = 128
    output_size = 10    # 10 digit classes

    # Initialize weights
    W1 = np.random.randn(input_size, hidden1) * 0.01
    b1 = np.zeros(hidden1)
    W2 = np.random.randn(hidden1, hidden2) * 0.01
    b2 = np.zeros(hidden2)
    W3 = np.random.randn(hidden2, output_size) * 0.01
    b3 = np.zeros(output_size)

    # Input batch
    X = np.random.randn(batch_size, input_size)

    print(f"Input: {X.shape}")

    # Forward pass — just matrix multiplications!
    A1 = neural_network_layer(X, W1, b1, "relu")
    print(f"After Layer 1 (ReLU): {A1.shape}")

    A2 = neural_network_layer(A1, W2, b2, "relu")
    print(f"After Layer 2 (ReLU): {A2.shape}")

    A3 = neural_network_layer(A2, W3, b3, "softmax")
    print(f"After Layer 3 (Softmax): {A3.shape}")

    # Predictions
    predictions = np.argmax(A3, axis=1)
    print(f"\nPredictions: {predictions}")
    print(f"Confidence: {A3.max(axis=1).round(3)}")
    print("\nThis is EXACTLY what PyTorch/TensorFlow do!")
    print("Just on GPU with automatic differentiation! 🔥")


if __name__ == "__main__":
    print("=== Matrix Multiplication ===\n")
    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])
    matrix_multiply_visual(A, B)

    full_neural_network_demo()
