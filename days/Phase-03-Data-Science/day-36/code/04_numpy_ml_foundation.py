"""
Day 36 — NumPy: Arrays & Operations
Topic: ML Foundations with NumPy
Date: 23 June 2026
Author: Bala Ravi

This file shows how ML algorithms work
internally using NumPy operations!
"""
import numpy as np


def linear_regression_numpy(
        X: np.ndarray,
        y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Implement linear regression from scratch!
    Uses NumPy matrix operations.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)

    Returns:
        weights, bias
    """
    n_samples = len(X)
    learning_rate = 0.01
    n_iterations = 1000

    weights = np.zeros(X.shape[1])
    bias = 0.0

    for _ in range(n_iterations):
        # Forward pass
        y_pred = np.dot(X, weights) + bias

        # Gradients (using vectorization!)
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)

        # Update
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias


def k_nearest_neighbors(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        k: int = 3) -> np.ndarray:
    """
    K-Nearest Neighbors from scratch with NumPy!

    Uses vectorized distance calculation —
    no loops for computing distances!

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        k: Number of neighbors

    Returns:
        Predicted labels
    """
    predictions = []

    for test_point in X_test:
        # Vectorized distance calculation!
        distances = np.sqrt(
            np.sum((X_train - test_point) ** 2, axis=1))

        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]

        # Majority vote
        unique, counts = np.unique(k_labels,
                                    return_counts=True)
        predictions.append(unique[np.argmax(counts)])

    return np.array(predictions)


def feature_engineering_numpy(
        X: np.ndarray) -> np.ndarray:
    """
    Common feature engineering operations.
    All vectorized — no loops!

    Args:
        X: Raw feature matrix

    Returns:
        Engineered feature matrix
    """
    features = [X]

    # Polynomial features (degree 2)
    for i in range(X.shape[1]):
        features.append(X[:, i:i+1] ** 2)

    # Interaction features
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
            features.append(interaction)

    return np.hstack(features)


if __name__ == "__main__":
    np.random.seed(42)
    print("=== ML Foundations with NumPy ===\n")

    # Linear Regression Demo
    print("1. Linear Regression from scratch:")
    n = 100
    X = np.random.randn(n, 2)
    true_weights = np.array([3.0, -1.5])
    y = np.dot(X, true_weights) + 2.0 + np.random.randn(n) * 0.1

    weights, bias = linear_regression_numpy(X, y)
    print(f"   True weights: {true_weights}")
    print(f"   Learned weights: {weights.round(2)}")
    print(f"   True bias: 2.0, Learned bias: {bias:.2f}")

    # KNN Demo
    print("\n2. K-Nearest Neighbors from scratch:")
    X_train = np.random.randn(50, 2)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(10, 2)

    predictions = k_nearest_neighbors(
        X_train, y_train, X_test, k=3)
    print(f"   Test predictions: {predictions}")
    print(f"   (1=positive class, 0=negative class)")

    # Feature Engineering
    print("\n3. Feature Engineering:")
    X_raw = np.random.randn(5, 3)
    X_engineered = feature_engineering_numpy(X_raw)
    print(f"   Raw features shape:        {X_raw.shape}")
    print(f"   Engineered features shape: {X_engineered.shape}")
    print(f"   Added polynomial + interaction features!")
