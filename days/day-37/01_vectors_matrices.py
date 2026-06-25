"""
Day 37 — NumPy: Linear Algebra
Topic: Vectors and Matrices
Date: 24 June 2026
Author: Bala Ravi

Linear Algebra = Language of Machine Learning!
Every ML operation is a vector/matrix operation!
"""
import numpy as np


def vector_operations_demo() -> None:
    """Demonstrate core vector operations."""
    print("=== Vector Operations ===\n")

    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"\nv1 + v2 = {v1 + v2}")
    print(f"v1 - v2 = {v1 - v2}")
    print(f"v1 * 2  = {v1 * 2}")

    # Dot product
    dot = np.dot(v1, v2)
    print(f"\nDot product v1·v2 = {dot}")
    print(f"  = {v1[0]}*{v2[0]} + "
          f"{v1[1]}*{v2[1]} + "
          f"{v1[2]}*{v2[2]} = {dot}")

    # Magnitude
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    print(f"\n|v1| = {mag1:.3f}")
    print(f"|v2| = {mag2:.3f}")

    # Unit vector
    v1_unit = v1 / mag1
    print(f"v1 unit vector = {v1_unit.round(3)}")
    print(f"Magnitude of unit vector: "
          f"{np.linalg.norm(v1_unit):.3f}")

    # Cosine similarity
    cos_sim = dot / (mag1 * mag2)
    print(f"\nCosine similarity = {cos_sim:.3f}")
    print(f"  (1.0 = identical, 0.0 = perpendicular,"
          f" -1.0 = opposite)")


def matrix_operations_demo() -> None:
    """Demonstrate core matrix operations."""
    print("\n=== Matrix Operations ===\n")

    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    print(f"A =\n{A}")
    print(f"\nB =\n{B}")

    print(f"\nA + B =\n{A + B}")
    print(f"\nA * B (element-wise) =\n{A * B}")
    print(f"\nA @ B (matrix multiply) =\n{A @ B}")
    print(f"\nA.T (transpose) =\n{A.T}")
    print(f"\nDet(A) = {np.linalg.det(A):.2f}")


def word_embeddings_demo() -> None:
    """
    Demonstrate word embeddings using vectors.
    Shows how NLP uses linear algebra!
    """
    print("\n=== Word Embeddings Demo ===\n")

    # Simplified word embeddings
    embeddings = {
        "king":   np.array([0.9, 0.1, 0.8, 0.1]),
        "queen":  np.array([0.9, 0.9, 0.8, 0.1]),
        "man":    np.array([0.1, 0.1, 0.8, 0.1]),
        "woman":  np.array([0.1, 0.9, 0.8, 0.1]),
        "python": np.array([0.2, 0.1, 0.1, 0.9]),
        "java":   np.array([0.2, 0.1, 0.1, 0.8])
    }

    def cosine_similarity(v1: np.ndarray,
                           v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2))

    # king - man + woman = ?
    result = (embeddings["king"] -
              embeddings["man"] +
              embeddings["woman"])

    print("king - man + woman =")
    similarities = {
        word: cosine_similarity(result, emb)
        for word, emb in embeddings.items()
        if word not in ["king", "man", "woman"]
    }

    closest = max(similarities, key=similarities.get)
    print(f"  Closest word: '{closest}' "
          f"(similarity: {similarities[closest]:.3f})")
    print(f"  Expected: 'queen'!")

    print("\nAll similarities:")
    for word, sim in sorted(similarities.items(),
                             key=lambda x: x[1],
                             reverse=True):
        print(f"  {word}: {sim:.3f}")


if __name__ == "__main__":
    vector_operations_demo()
    matrix_operations_demo()
    word_embeddings_demo()
