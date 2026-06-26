# Day 36 — NumPy: Arrays & Operations 🚀

**Date:** 23 June 2026
**Time Spent:** (3.30 hours)
**Resource Used:** [NumPy Docs](https://numpy.org/doc/) | [Kaggle NumPy](https://www.kaggle.com/learn)

---

## 📚 Topics Covered

- What is NumPy and why it exists
- ndarray vs Python list
- Creating arrays
- Array indexing and slicing
- Array operations — vectorization
- Broadcasting
- Universal functions (ufuncs)
- Shape manipulation
- Real ML applications

---

## 🔑 Why NumPy Exists

```python
# Python list — slow for math!
prices = [100, 200, 300, 400, 500]
doubled = [p * 2 for p in prices]  # loop needed!

# NumPy array — fast math!
import numpy as np
prices = np.array([100, 200, 300, 400, 500])
doubled = prices * 2  # no loop needed! 🔥
```

**Why is NumPy 100x faster?**
Python list:

→ Stores objects in memory (flexible but slow)

→ Each operation goes through Python interpreter

→ No type checking optimization
NumPy array:
→ Stores raw numbers in CONTIGUOUS memory (C-style)
→ Operations run in optimized C code
→ Fixed data type = maximum optimization
→ SIMD (Single Instruction Multiple Data) CPU ops!
**So what? Why does this matter?**
Every ML dataset is a NumPy array internally!
Pandas uses NumPy under the hood!
TensorFlow and PyTorch are built on NumPy concepts!
Without NumPy — ML would be 100x slower! 🔥

---

## 🔑 ndarray — The Core Data Structure

```python
import numpy as np

# 1D array — a vector
vector = np.array([1, 2, 3, 4, 5])

# 2D array — a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Key properties
print(vector.shape)   # (5,)
print(matrix.shape)   # (3, 3)
print(matrix.ndim)    # 2
print(matrix.dtype)   # int64
print(matrix.size)    # 9 (total elements)
```

**Shape intuition for ML:**
(100,)       → 100 samples, 1 feature
(100, 5)     → 100 samples, 5 features
(100, 28, 28) → 100 images, 28x28 pixels each!
---

## 🔑 Creating Arrays — All Methods

```python
import numpy as np

# From Python list
arr = np.array([1, 2, 3, 4, 5])

# Zeros and ones
zeros = np.zeros((3, 4))      # 3x4 matrix of 0s
ones = np.ones((2, 3))        # 2x3 matrix of 1s
identity = np.eye(3)          # 3x3 identity matrix

# Ranges
range_arr = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1]

# Random arrays
random = np.random.rand(3, 3)     # uniform 0-1
normal = np.random.randn(1000)    # normal distribution
integers = np.random.randint(0, 100, size=(5, 5))
```

**ML Connection:**
```python
# Initialize neural network weights!
weights = np.random.randn(input_size, hidden_size) * 0.01
bias = np.zeros(hidden_size)

# Create training labels
labels = np.zeros(n_samples)
labels[:n_positive] = 1
```

---

## 🔑 Indexing & Slicing

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Basic indexing
arr[0, 0]      # 1 — first row, first col
arr[2, 1]      # 8 — third row, second col
arr[-1, -1]    # 9 — last row, last col

# Slicing
arr[0, :]      # [1, 2, 3] — first row
arr[:, 0]      # [1, 4, 7] — first column
arr[1:, 1:]    # bottom-right 2x2 submatrix

# Boolean indexing — CRITICAL for ML!
scores = np.array([45, 78, 92, 56, 88, 34])
passed = scores[scores >= 60]    # [78, 92, 88]
failed = scores[scores < 60]     # [45, 56, 34]

# Fancy indexing
selected = arr[[0, 2], :]  # rows 0 and 2
```

**ML Connection:**
```python
# Filter training data!
X = np.random.randn(1000, 5)  # 1000 samples
y = np.random.randint(0, 2, 1000)  # labels

# Get only positive class samples
X_positive = X[y == 1]
X_negative = X[y == 0]
```

---

## 🔑 Vectorization — No More Loops!

```python
import numpy as np
import time

# Python loop — SLOW!
def normalize_python(data: list) -> list:
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# NumPy vectorized — FAST!
def normalize_numpy(data: np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min())

# Speed comparison
n = 1_000_000
data = list(range(n))
arr = np.array(data)

start = time.time()
normalize_python(data)
python_time = time.time() - start

start = time.time()
normalize_numpy(arr)
numpy_time = time.time() - start

print(f"Python: {python_time:.3f}s")
print(f"NumPy:  {numpy_time:.3f}s")
print(f"NumPy is {python_time/numpy_time:.0f}x faster!")
```

---

## 🔑 Broadcasting — NumPy's Superpower!

```python
# Broadcasting = operations on different shapes!
arr = np.array([[1, 2, 3],
                [4, 5, 6]])  # shape (2, 3)

scalar = 10
result = arr + scalar  # adds 10 to every element!

row = np.array([1, 2, 3])  # shape (3,)
result = arr + row  # adds row to each row of arr!

# Broadcasting rules:
# 1. Shapes are compared from trailing dimensions
# 2. Dimensions must be equal OR one of them is 1
```

**ML Connection:**
```python
# Feature normalization with broadcasting!
X = np.random.randn(100, 5)  # 100 samples, 5 features
mean = X.mean(axis=0)         # shape (5,)
std = X.std(axis=0)           # shape (5,)

# Broadcasting normalizes all 100 samples at once!
X_normalized = (X - mean) / std
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Array Creation | All methods |
| 2 | Indexing & Slicing | Boolean indexing |
| 3 | Vectorization | Speed comparison |
| 4 | Broadcasting | Shape operations |
| 5 | Statistical Operations | Mean, std, var |
| 6 | Financial Analysis | ArthAI with NumPy |
| 7 | ML Data Pipeline | Preprocessing |

---

## 🔗 How This Connects to AI/ML

```python
import numpy as np

# 1. Feature scaling — used in EVERY ML pipeline!
def min_max_scale(X: np.ndarray) -> np.ndarray:
    return (X - X.min(axis=0)) / (
        X.max(axis=0) - X.min(axis=0))

# 2. Neural network forward pass!
def forward(X: np.ndarray,
            W: np.ndarray,
            b: np.ndarray) -> np.ndarray:
    return np.dot(X, W) + b  # matrix multiplication!

# 3. Activation function
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

# 4. Loss calculation
def mse_loss(y_true: np.ndarray,
             y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)

# 5. ArthAI financial calculations — vectorized!
def calculate_emi_batch(
        principals: np.ndarray,
        rates: np.ndarray,
        years: np.ndarray) -> np.ndarray:
    """Calculate EMI for multiple loans at once!"""
    monthly_rates = rates / 12 / 100
    months = years * 12
    emi = (principals * monthly_rates *
           (1 + monthly_rates) ** months /
           ((1 + monthly_rates) ** months - 1))
    return np.round(emi, 2)
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Confusing shape and size:**
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3) — rows, cols
print(arr.size)   # 6 — total elements
print(len(arr))   # 2 — first dimension only!
```

**Mistake 2 — Copy vs view:**
```python
arr = np.array([1, 2, 3, 4, 5])

# View — shares memory!
view = arr[1:3]
view[0] = 99    # modifies original arr too! ❌

# Copy — independent!
copy = arr[1:3].copy()
copy[0] = 99    # doesn't modify original ✅
```

**Mistake 3 — Wrong axis in operations:**
```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])

X.mean()          # 3.5 — mean of ALL elements
X.mean(axis=0)    # [2.5, 3.5, 4.5] — mean per COLUMN
X.mean(axis=1)    # [2.0, 5.0] — mean per ROW
```

---

## 💎 Important Realizations

1. **NumPy is the language of ML**
   Every neural network weight is an ndarray!
   Every dataset row is an ndarray!
   Every prediction is an ndarray!

2. **Vectorization replaces all ML loops**
   Loops in Python = slow ML training
   NumPy operations = fast ML training
   This is why GPU training is possible!

3. **Broadcasting is genius**
   Normalizing 10,000 samples with 100 features
   takes ONE line with broadcasting — no loops!

4. **Shape is everything in ML**
   Wrong shape = wrong model = wrong results
   Always print .shape when debugging ML code!

---

## 🎯 Next Goal

- NumPy Linear Algebra
- Matrix multiplication
- The math foundation of neural networks!

---

*Day 36 complete — Phase 3 started! 🔥*
