# Day 56 — SVM & KNN 🚀

**Date:** 13 July 2026
**Time Spent:** (add your hours)
**Resource Used:** [Scikit-learn Docs](https://scikit-learn.org/) | [StatQuest](https://www.youtube.com/@statquest)

---

## 📚 Topics Covered

- What is SVM — Support Vector Machine
- The Maximum Margin concept
- Support Vectors
- Kernel Trick — handling non-linear data
- What is KNN — K-Nearest Neighbors
- Choosing K
- Distance metrics
- SVM vs KNN vs Random Forest
- When to use which algorithm

---

## 🔑 What is SVM?
Find the decision boundary that maximizes
the MARGIN between classes!
✅ PASS          ❌ FAIL
          |    MARGIN    |
 ●  ●  ●  |←————————————→|  ○  ○  ○
 ●  ●     |              |     ○  ○
          |  Hyperplane  |
Support Vectors = the points CLOSEST to
the boundary — they define the margin!
Maximize this margin = best generalization

**So what? Why does this matter?**
SVM with the right kernel can separate
data that NO straight line can separate!
The kernel trick maps data to higher
dimensions where it BECOMES separable!
Used in:
→ Image classification (before deep learning)
→ Text classification (SVMs + TF-IDF)
→ Bioinformatics (gene expression)
→ Anywhere with small datasets!

---

## 🔑 The Kernel Trick
Linear kernel:    straight line boundary
Use when data is linearly separable
Polynomial kernel: curved boundary (degree d)
Use for moderate non-linearity
RBF (Gaussian):    infinite-dimensional mapping!
Use for complex non-linear data
Most commonly used kernel! 🔥
Sigmoid kernel:    similar to neural network layer
```python
from sklearn.svm import SVC

# Linear kernel
model = SVC(kernel='linear', C=1.0)

# RBF kernel (default and most common)
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# Polynomial kernel
model = SVC(kernel='poly', degree=3, C=1.0)
```

---

## 🔑 SVM Hyperparameters

C (Regularization):
→ High C → small margin, fits training data closely
(may overfit)
→ Low C  → large margin, allows misclassifications
(better generalization)
gamma (for RBF kernel):
→ High gamma → each point has local influence
(may overfit)
→ Low gamma  → each point has global influence
(smoother boundary)
'scale' (default) = 1/(n_features * X.var())
'auto'            = 1/n_features
---

## 🔑 What is KNN?'
K-Nearest Neighbors — the simplest ML algorithm!
"A student is likely to get the same grade
as the K most similar students to them!"
Algorithm:

Store ALL training data (no actual "training"!)
For new sample:
a. Calculate distance to EVERY training sample
b. Find K nearest neighbors
c. Majority vote among K neighbors = prediction!

No model learned. No parameters optimized.
Just distances and votes! 🔥
---

## 🔑 Choosing K
K = 1:  → predict same as nearest neighbor
Very sensitive to noise! (overfit)
K = N:  → predict same class always (majority)
Ignores all local patterns! (underfit)
K = √n: → common rule of thumb
Usually odd number to avoid ties!
Tune K using cross-validation!
```python
from sklearn.neighbors import KNeighborsClassifier

# Always scale features first for KNN!
# KNN uses DISTANCES — unscaled salary (₹30000)
# will dominate age (25) completely!

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)

model = KNeighborsClassifier(
    n_neighbors=5,          # K
    metric='euclidean',     # distance metric
    weights='uniform'       # all neighbors equal
)
model.fit(X_train_s, y_train)
```

---

## 🔑 Distance Metrics
Euclidean: √(Σ(a-b)²)    — straight line distance
Most common! Use for continuous features.
Manhattan: Σ|a-b|         — city block distance
More robust to outliers.
Minkowski: general form (p=2 → Euclidean, p=1 → Manhattan)
Cosine:    angle between vectors
Used in NLP (text similarity)!
---

## 🔑 SVM vs KNN vs Random Forest
╔══════════════╦═══════════╦═══════════╦═══════════╗
║ Property     ║ SVM       ║ KNN       ║ RF        ║
╠══════════════╬═══════════╬═══════════╬═══════════╣
║ Training     ║ Slow      ║ Instant   ║ Moderate  ║
║ Prediction   ║ Fast      ║ Very Slow ║ Fast      ║
║ Scaling      ║ Required  ║ Required  ║ Not needed║
║ Interpretable║ No        ║ Semi      ║ Yes (FI)  ║
║ Large data   ║ Struggles ║ Struggles ║ Handles   ║
║ Small data   ║ Excellent ║ Good      ║ Good      ║
║ Noisy data   ║ Robust    ║ Sensitive ║ Robust    ║
╚══════════════╩═══════════╩═══════════╩═══════════╝
---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | SVM Linear kernel | Maximum margin |
| 2 | SVM RBF kernel | Kernel trick |
| 3 | KNN from scratch | Distance + voting |
| 4 | K selection | Cross-validation |
| 5 | All classifiers compared | When to use which |
| 6 | Student predictor complete | Full comparison |

---

## 🔗 How This Connects to AI/ML

```python
# SVM → text classification!
# TF-IDF features + SVM = powerful text classifier
# Used in spam detection, sentiment analysis!
# Before transformers — SVM was king of NLP!

# KNN → recommendation systems!
# "Users similar to you also liked..."
# Find K nearest users by preference vectors
# → recommend what they liked!

# KNN → anomaly detection!
# If a point has no close neighbors → anomaly!
# Used in fraud detection — unusual transactions!

# Distance from Day 37 (Linear Algebra)
# powers KNN entirely!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Not scaling for SVM/KNN:**
```python
# Wrong — features on different scales!
svm.fit(X_raw, y)    # ❌ salary dominates!
knn.fit(X_raw, y)    # ❌ salary dominates!

# Correct — always scale first!
X_s = StandardScaler().fit_transform(X_train)
svm.fit(X_s, y_train)  # ✅
knn.fit(X_s, y_train)  # ✅
```

**Mistake 2 — Even K causes ties in KNN:**
```python
knn = KNeighborsClassifier(n_neighbors=4)  # ❌
# 2 vs 2 tie — undefined behavior!

knn = KNeighborsClassifier(n_neighbors=5)  # ✅
# Always use ODD K for binary classification!
```

**Mistake 3 — Using KNN on large datasets:**
```python
# KNN prediction requires comparing to ALL
# training samples — O(n) per prediction!
# 1M training samples → very slow!

# For large data → use Random Forest instead!
# Or use approximate KNN (FAISS, Annoy libraries)
```

---

## 💎 Important Realizations

1. **SVM finds the MOST ROBUST boundary**
   Not just any boundary that separates —
   the one with MAXIMUM distance from both classes!
   That's why it generalizes better!

2. **Kernel trick is mathematically brilliant**
   Instead of actually mapping to high dimensions
   (too expensive) — it computes the dot product
   in that space directly! O(n) instead of O(n^d)!

3. **KNN has NO training phase**
   fit() just stores the data!
   All work happens at prediction time!
   This makes it memory-intensive but flexible!

4. **Use SVM for small, high-dimensional data**
   Text classification: thousands of features, few samples
   SVM handles this better than RF or KNN!

---

## 🎯 Next Goal

- Model Evaluation — metrics deep dive!
- ROC curves, precision-recall curves
- Everything about evaluating classifiers!

---

*Day 56 complete — SVM & KNN mastered! ⚡🔥*


