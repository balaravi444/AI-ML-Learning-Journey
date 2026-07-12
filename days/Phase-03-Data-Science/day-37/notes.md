# Day 37 — NumPy: Linear Algebra 🚀

**Date:** 24 June 2026
**Time Spent:** (4.5 hours)
**Resource Used:** [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html) | [3Blue1Brown Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

---

## 📚 Topics Covered

- Vectors and Matrices
- Matrix Multiplication
- Dot Product
- Transpose
- Determinant
- Inverse Matrix
- Eigenvalues & Eigenvectors
- Solving Linear Systems
- Real ML applications

---

## 🔑 Why Linear Algebra for ML?
Neural Networks     = matrix multiplications

PCA                 = eigenvalues/eigenvectors

Linear Regression   = solving linear systems

Word Embeddings     = vector operations

Attention in GPT    = matrix multiplication
Everything in ML is LINEAR ALGEBRA! 🔥
**So what? Why does this matter?**
When you call `model.fit()` in Scikit-learn —
it solves a system of linear equations internally!
When ChatGPT generates text — it does billions
of matrix multiplications every second!
Understanding linear algebra = understanding ML! 🧠

---

## 🔑 Vectors

A vector is a 1D array with magnitude and direction:

```python
import numpy as np

# Column vector (feature vector for one sample!)
v = np.array([2, 3, 4])

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Addition
v1 + v2  # [5, 7, 9]

# Dot product — measures similarity!
np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32

# Magnitude (length)
np.linalg.norm(v1)  # sqrt(1+4+9) = 3.74

# Unit vector (normalize)
v_unit = v1 / np.linalg.norm(v1)
```

**So what? Why does this matter?**
Word embeddings in NLP = vectors!
"king" - "man" + "woman" = "queen"
This is VECTOR ARITHMETIC on word embeddings!
Cosine similarity between vectors = how similar
two documents/words are — used in search engines!

---

## 🔑 Matrix Multiplication

The MOST important operation in ML!

```python
A = np.array([[1, 2],
              [3, 4]])  # shape (2, 2)

B = np.array([[5, 6],
              [7, 8]])  # shape (2, 2)

# Matrix multiplication
C = np.dot(A, B)    # or A @ B

# Rule: (m, n) @ (n, p) = (m, p)
# Inner dimensions must match!
```

**Visual explanation:**
[1, 2]   [5, 6]   [15+27, 16+28]   [19, 22]
[3, 4] @ [7, 8] = [35+47, 36+48] = [43, 50]
**ML Connection:**
```python
# Neural network forward pass = matrix multiplication!
# X (batch, features) @ W (features, neurons) = Z (batch, neurons)

X = np.random.randn(32, 784)   # 32 images, 784 pixels
W = np.random.randn(784, 128)  # 784 inputs, 128 neurons
b = np.zeros(128)

Z = X @ W + b   # shape: (32, 128)
# This IS a neural network layer! 🤯
```

---

## 🔑 Transpose

Flip rows and columns:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)

A.T   # shape (3, 2)
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

**ML Connection:**
```python
# Gradient computation uses transpose!
# dL/dW = X.T @ dL/dZ

dL_dZ = np.random.randn(32, 128)  # gradient
dL_dW = X.T @ dL_dZ               # weight gradient
# This is BACKPROPAGATION! 🔥
```

---

## 🔑 Determinant

A scalar value describing a matrix's properties:

```python
A = np.array([[3, 8],
              [4, 6]])

det = np.linalg.det(A)  # 3*6 - 8*4 = -14

# If det = 0 → matrix is singular (no inverse!)
# If det ≠ 0 → matrix is invertible
```

**ML Connection:**
```python
# Multivariate Gaussian distribution uses determinant!
# Used in Gaussian Mixture Models (GMM) for clustering!
# PDF = (1/sqrt(2π|Σ|)) * exp(-0.5 * (x-μ)T Σ⁻¹ (x-μ))
```

---

## 🔑 Matrix Inverse

The matrix equivalent of division:

```python
A = np.array([[3, 1],
              [2, 4]])

A_inv = np.linalg.inv(A)
print(A @ A_inv)  # identity matrix!
```

**ML Connection:**
```python
# Normal equation for Linear Regression!
# w = (X.T @ X)^(-1) @ X.T @ y
# This solves linear regression EXACTLY!

def linear_regression_normal_eq(
        X: np.ndarray,
        y: np.ndarray) -> np.ndarray:
    return np.linalg.inv(X.T @ X) @ X.T @ y
```

---

## 🔑 Eigenvalues & Eigenvectors

Special vectors that only scale under transformation:

```python
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print(eigenvalues)   # [5. 2.]
print(eigenvectors)  # columns are eigenvectors
```

**ML Connection:**
```python
# PCA uses eigenvalues to find principal components!
# Steps:
# 1. Compute covariance matrix
# 2. Find eigenvalues and eigenvectors
# 3. Sort by eigenvalues (largest = most important!)
# 4. Project data onto top k eigenvectors

def pca_from_scratch(
        X: np.ndarray,
        n_components: int) -> np.ndarray:
    # Center data
    X_centered = X - X.mean(axis=0)

    # Covariance matrix
    cov = np.cov(X_centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalues descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project to n_components dimensions
    return X_centered @ eigenvectors[:, :n_components]
```

**So what? Why does this matter?**
PCA reduces 1000 features to 10 most important!
Used in face recognition, image compression,
and removing noise from ML training data!

---

## 🔑 Solving Linear Systems

```python
# Ax = b — find x!
A = np.array([[3, 1],
              [2, 4]])
b = np.array([9, 8])

# Direct solve — faster than computing inverse!
x = np.linalg.solve(A, b)
print(x)  # [2.5, 1.5]
```

**ML Connection:**
```python
# Linear regression is a linear system!
# X.T @ X @ w = X.T @ y
# Solve for w (weights)!
```

---

## 💻 Programs Practiced

| # | Topic | ML Application |
|---|-------|----------------|
| 1 | Vector operations | Word embeddings |
| 2 | Matrix multiplication | Neural network layers |
| 3 | Transpose | Backpropagation |
| 4 | Inverse | Normal equation |
| 5 | Eigenvalues | PCA implementation |
| 6 | Linear systems | Linear regression |
| 7 | Cosine similarity | Document similarity |
| 8 | Full neural net | Forward + backward pass |

---

## 🔗 How This Connects to AI/ML

```python
# THE most important equation in ML:
# Neural network forward pass

def neural_network_forward(
        X: np.ndarray,
        layers: list[tuple]) -> np.ndarray:
    """
    Complete neural network forward pass!
    Just matrix multiplications and activations!
    """
    activation = X
    for W, b in layers:
        Z = activation @ W + b    # LINEAR ALGEBRA!
        activation = np.maximum(0, Z)  # ReLU
    return activation

# This is PyTorch/TensorFlow under the hood!
# They just do this on GPU with auto-differentiation!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Wrong matrix multiplication shape:**
```python
A = np.array([[1, 2], [3, 4]])  # (2, 2)
B = np.array([[1, 2, 3]])       # (1, 3)

A @ B  # ❌ ValueError — (2,2) @ (1,3) doesn't work!

# Fix — check inner dimensions match!
# (2, 2) @ (2, 3) ✅ → (2, 3)
B_correct = np.array([[1, 2, 3],
                       [4, 5, 6]])  # (2, 3)
A @ B_correct  # ✅ (2, 3)
```

**Mistake 2 — Using inv() instead of solve():**
```python
# Slow and numerically unstable!
x = np.linalg.inv(A) @ b  # ❌

# Better — use solve() directly!
x = np.linalg.solve(A, b)  # ✅ faster and stable!
```

---

## 💎 Important Realizations

1. **Every neural network = stacked matrix multiplications**
   GPT-4 does TRILLIONS of matrix multiplications.
   Understanding `A @ B` means understanding GPT!

2. **PCA = eigenvalues in disguise**
   Feature reduction, noise removal, image compression —
   all just eigenvalue decomposition!

3. **Linear algebra is the LANGUAGE of ML**
   Not just a topic to learn — it's how you
   READ and WRITE ML code professionally!

4. **np.linalg is your ML math toolkit**
   Every algorithm from basic regression to
   deep learning uses it!

---

## 🎯 Next Goal

- Pandas — DataFrames
- Load and manipulate real datasets
- Foundation for the Indian Job Market Analyzer!

---

*Day 37 complete — Linear Algebra unlocked! 🔥*
