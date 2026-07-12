"""
Day 37 — NumPy: Linear Algebra
Topic: Eigenvalues, Eigenvectors & PCA
Date: 24 June 2026
Author: Bala Ravi

PCA (Principal Component Analysis) =
Eigenvalue decomposition of covariance matrix!

Used in:
- Dimensionality reduction
- Face recognition
- Image compression
- Noise removal from ML data!
"""
import numpy as np


def explain_eigenvalues() -> None:
    """Demonstrate eigenvalues and eigenvectors."""
    print("=== Eigenvalues & Eigenvectors ===\n")

    A = np.array([[4, 2],
                  [1, 3]])

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print(f"Matrix A:\n{A}")
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors.round(3)}")

    print("\nVerification: A @ v = λ * v")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        Av = A @ v
        lv = lam * v
        print(f"  λ{i+1}={lam:.1f}: "
              f"A@v={Av.round(3)}, "
              f"λ*v={lv.round(3)} ✅")


def pca_from_scratch(
        X: np.ndarray,
        n_components: int) -> dict:
    """
    Implement PCA from scratch using eigenvalues!

    Steps:
    1. Center the data
    2. Compute covariance matrix
    3. Find eigenvalues and eigenvectors
    4. Sort by eigenvalues (most important first!)
    5. Project data onto top components

    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components to keep

    Returns:
        PCA results dictionary
    """
    n_samples, n_features = X.shape

    # Step 1 — Center data
    X_centered = X - X.mean(axis=0)

    # Step 2 — Covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Step 3 — Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Step 4 — Sort by eigenvalues descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5 — Project to n_components
    components = eigenvectors[:, :n_components]
    X_transformed = X_centered @ components

    # Explained variance
    total_variance = eigenvalues.sum()
    explained_variance = eigenvalues[:n_components] / total_variance * 100

    return {
        "X_transformed": X_transformed,
        "eigenvalues": eigenvalues[:n_components],
        "explained_variance_pct": explained_variance,
        "total_explained": explained_variance.sum(),
        "components": components
    }


def pca_demo() -> None:
    """Demonstrate PCA on synthetic dataset."""
    print("\n=== PCA From Scratch ===\n")
    np.random.seed(42)

    # Generate correlated data
    n_samples = 200
    X = np.random.randn(n_samples, 5)
    X[:, 1] = X[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2
    X[:, 2] = X[:, 0] * 0.6 + np.random.randn(n_samples) * 0.3

    print(f"Original data shape: {X.shape}")
    print(f"Original features: {X.shape[1]}")

    # Apply PCA
    result = pca_from_scratch(X, n_components=2)

    print(f"\nAfter PCA → shape: {result['X_transformed'].shape}")
    print(f"Reduced to: {result['X_transformed'].shape[1]} components")
    print(f"\nExplained variance per component:")
    for i, var in enumerate(result['explained_variance_pct']):
        print(f"  PC{i+1}: {var:.1f}%")
    print(f"Total explained: {result['total_explained']:.1f}%")
    print(f"\nKept {result['total_explained']:.0f}% of information")
    print(f"with only {result['X_transformed'].shape[1]}"
          f" of {X.shape[1]} features!")


def cosine_similarity_matrix(
        embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    Used in NLP for document similarity!

    Args:
        embeddings: Matrix of embeddings (n, d)

    Returns:
        Similarity matrix (n, n)
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def linear_regression_normal_equation(
        X: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    """
    Solve linear regression using normal equation.
    w = (X.T @ X)^(-1) @ X.T @ y

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        Optimal weights
    """
    return np.linalg.solve(X.T @ X, X.T @ y)


if __name__ == "__main__":
    explain_eigenvalues()
    pca_demo()

    print("\n=== Linear Regression — Normal Equation ===\n")
    np.random.seed(42)
    n = 100
    X_raw = np.random.randn(n, 3)
    X_with_bias = np.hstack([np.ones((n, 1)), X_raw])
    true_w = np.array([2.0, 1.5, -0.8, 0.3])
    y = X_with_bias @ true_w + np.random.randn(n) * 0.1

    w_learned = linear_regression_normal_equation(
        X_with_bias, y)
    print(f"True weights:    {true_w}")
    print(f"Learned weights: {w_learned.round(3)}")

    print("\n=== Document Similarity ===\n")
    docs = np.array([
        [1.0, 0.8, 0.1, 0.2],  # ML doc
        [0.9, 0.7, 0.1, 0.3],  # AI doc
        [0.1, 0.2, 0.9, 0.8],  # Cooking doc
        [0.2, 0.1, 0.8, 0.9]   # Food doc
    ])
    labels = ["ML", "AI", "Cooking", "Food"]
    sim_matrix = cosine_similarity_matrix(docs)

    print("Cosine Similarity Matrix:")
    print(f"{'':>10}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label:>10}", end="")
        for j in range(len(labels)):
            print(f"{sim_matrix[i,j]:>10.3f}", end="")
        print()
    print("\nML and AI are most similar (0.99+)")
    print("Cooking and Food are most similar")
    print("ML and Cooking are least similar ✅")
