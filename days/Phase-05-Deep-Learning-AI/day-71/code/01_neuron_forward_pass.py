"""
Day 71 — Neural Networks from Scratch
Topic: Neuron + Forward Pass
Date: 28 July 2026
Author: Bala Ravii

Building every piece of a neural network
from scratch using only NumPy!
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ─── Activation Functions ─────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    """σ(z) = 1 / (1 + e^(-z))"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """σ'(z) = σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU(z) = max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """ReLU'(z) = 1 if z > 0 else 0"""
    return (z > 0).astype(float)


def tanh(z: np.ndarray) -> np.ndarray:
    """tanh(z) = (e^z - e^-z) / (e^z + e^-z)"""
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """tanh'(z) = 1 - tanh(z)²"""
    return 1 - np.tanh(z) ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax for multi-class output."""
    z_shift = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def demonstrate_activations() -> None:
    """Show activation functions behavior."""
    print("=== Activation Functions ===\n")

    z_values = np.array([-3, -1, 0, 1, 3])

    print(f"{'z':>6} | "
          f"{'Sigmoid':>8} | "
          f"{'ReLU':>6} | "
          f"{'Tanh':>6}")
    print("-" * 35)

    for z in z_values:
        print(f"{z:>6.1f} | "
              f"{sigmoid(np.array([z]))[0]:>8.4f} | "
              f"{relu(np.array([z]))[0]:>6.4f} | "
              f"{tanh(np.array([z]))[0]:>6.4f}")

    print(f"\n💡 ReLU is used in hidden layers")
    print(f"   (avoids vanishing gradient!)")
    print(f"   Sigmoid is used in output layer")
    print(f"   (gives probability 0-1)")


def single_neuron_demo() -> None:
    """Show how a single neuron works."""
    print("\n=== Single Neuron ===\n")
    print("z = w₁x₁ + w₂x₂ + b")
    print("a = activation(z)\n")

    np.random.seed(42)
    weights = np.array([0.5, -0.3])
    bias = 0.1

    inputs = np.array([
        [2.0, 1.0],
        [0.5, 3.0],
        [-1.0, 0.5]
    ])

    print(f"Weights: {weights}")
    print(f"Bias:    {bias}\n")

    print(f"{'Input':>15} | "
          f"{'z':>8} | "
          f"{'ReLU(z)':>8} | "
          f"{'σ(z)':>8}")
    print("-" * 48)

    for x in inputs:
        z = np.dot(weights, x) + bias
        a_relu = relu(np.array([z]))[0]
        a_sigmoid = sigmoid(np.array([z]))[0]
        print(f"{str(x):>15} | "
              f"{z:>8.4f} | "
              f"{a_relu:>8.4f} | "
              f"{a_sigmoid:>8.4f}")


def forward_pass_layer(
        X: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        activation: str = 'relu'
        ) -> tuple:
    """
    Forward pass through one layer.

    Args:
        X: Input (batch_size, n_inputs)
        W: Weights (n_inputs, n_neurons)
        b: Biases (1, n_neurons)
        activation: 'relu', 'sigmoid', 'tanh'

    Returns:
        (Z, A) — pre-activation and activation
    """
    Z = X @ W + b

    if activation == 'relu':
        A = relu(Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'tanh':
        A = tanh(Z)
    elif activation == 'softmax':
        A = softmax(Z)
    else:
        A = Z  # linear

    return Z, A


def full_forward_pass_demo() -> None:
    """Show full forward pass through 3-layer NN."""
    print("\n=== Full Forward Pass ===\n")
    print("Architecture: 4 → 8 → 4 → 1")
    print("Activations:  input → ReLU → ReLU → Sigmoid\n")

    np.random.seed(42)
    n_samples = 5
    n_inputs = 4

    X = np.random.randn(n_samples, n_inputs)

    # Layer 1: 4 → 8
    W1 = np.random.randn(4, 8) * 0.1
    b1 = np.zeros((1, 8))

    # Layer 2: 8 → 4
    W2 = np.random.randn(8, 4) * 0.1
    b2 = np.zeros((1, 4))

    # Output: 4 → 1
    W3 = np.random.randn(4, 1) * 0.1
    b3 = np.zeros((1, 1))

    print(f"Input X:  {X.shape}")

    Z1, A1 = forward_pass_layer(
        X, W1, b1, 'relu')
    print(f"Layer 1 (ReLU):    Z1={Z1.shape}, "
          f"A1={A1.shape}")

    Z2, A2 = forward_pass_layer(
        A1, W2, b2, 'relu')
    print(f"Layer 2 (ReLU):    Z2={Z2.shape}, "
          f"A2={A2.shape}")

    Z3, A3 = forward_pass_layer(
        A2, W3, b3, 'sigmoid')
    print(f"Output (Sigmoid):  Z3={Z3.shape}, "
          f"Ŷ={A3.shape}")

    print(f"\nPredictions (probabilities):")
    for i, pred in enumerate(A3.flatten()):
        label = "Positive" if pred >= 0.5 else "Negative"
        print(f"  Sample {i+1}: "
              f"{pred:.4f} → {label}")


if __name__ == "__main__":
    demonstrate_activations()
    single_neuron_demo()
    full_forward_pass_demo()
