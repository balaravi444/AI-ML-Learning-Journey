# Day 53 — Logistic Regression 🚀

**Date:** 10 July 2026
**Time Spent:** (2 hours)
**Resource Used:** [Scikit-learn Docs](https://scikit-learn.org/) | [StatQuest](https://www.youtube.com/@statquest)

---

## 📚 Topics Covered

- What is Logistic Regression
- Linear vs Logistic Regression
- The Sigmoid Function
- Log Loss / Binary Cross-Entropy
- Decision Boundary
- Multi-class classification
- Regularization in Logistic Regression
- From scratch implementation
- Real application — student pass/fail predictor

---

## 🔑 What is Logistic Regression?
Linear Regression  → predicts a NUMBER (salary)
Logistic Regression → predicts a PROBABILITY (0 to 1)
then converts to CLASS (pass/fail)
Same linear equation underneath:
z = w₁x₁ + w₂x₂ + ... + b
But instead of outputting z directly —
pass it through the SIGMOID function!
p = sigmoid(z) = 1 / (1 + e^(-z))
Output is always between 0 and 1!

**So what? Why does this matter?**
Logistic Regression powers:
→ Spam detection (spam or not spam)
→ Disease prediction (sick or healthy)
→ Credit scoring (default or not)
→ Student pass/fail prediction
→ Click prediction in ad systems
It's the SIMPLEST and most interpretable
classifier — always try it first! 🔥

---

## 🔑 The Sigmoid Function
sigmoid(z) = 1 / (1 + e^(-z))
Properties:
→ Output always between 0 and 1
→ sigmoid(0) = 0.5
→ Large positive z → output ≈ 1
→ Large negative z → output ≈ 0
Decision boundary:
→ p >= 0.5 → predict class 1
→ p < 0.5  → predict class 0

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Test
print(sigmoid(0))    # 0.5 — right on boundary
print(sigmoid(10))   # ≈ 1.0 — very confident class 1
print(sigmoid(-10))  # ≈ 0.0 — very confident class 0
```

---

## 🔑 Log Loss — The Cost Function
For Linear Regression:  MSE = (y - y_pred)²
For Logistic Regression: Log Loss (Binary Cross-Entropy)
Loss = -[y * log(p) + (1 - y) * log(1 - p)]
When y = 1 (actual positive):
→ Loss = -log(p)
→ if p = 0.99 → loss ≈ 0 (good!)
→ if p = 0.01 → loss ≈ 4.6 (bad!)
When y = 0 (actual negative):
→ Loss = -log(1 - p)
→ if p = 0.01 → loss ≈ 0 (good!)
→ if p = 0.99 → loss ≈ 4.6 (bad!)

**Why not use MSE for classification?**
MSE with sigmoid creates non-convex loss!
Multiple local minima → gradient descent fails!
Log Loss is convex → gradient descent works! ✅

---

## 🔑 Multi-class Classification

```python
# Binary (2 classes)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Multi-class (3+ classes) — OvR strategy
model = LogisticRegression(multi_class='ovr')
# Trains one binary classifier per class
# "Is this class A? Yes/No"
# "Is this class B? Yes/No"
# "Is this class C? Yes/No"

# Softmax (multinomial)
model = LogisticRegression(multi_class='multinomial',
                            solver='lbfgs')
# Single model outputs probabilities for ALL classes
# Probabilities sum to 1!
```

---

## 🔑 Regularization

```python
# C = inverse of regularization strength
# High C → less regularization (may overfit)
# Low C  → more regularization (may underfit)

# L2 regularization (default)
model = LogisticRegression(C=1.0, penalty='l2')

# L1 regularization (feature selection!)
model = LogisticRegression(C=1.0, penalty='l1',
                            solver='liblinear')

# No regularization
model = LogisticRegression(C=1e10, penalty='l2')
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Logistic Regression from scratch | Sigmoid + Log Loss + Gradient Descent |
| 2 | Sklearn binary classification | fit, predict, predict_proba |
| 3 | Decision boundary visualization | threshold tuning |
| 4 | Multi-class classification | OvR vs Softmax |
| 5 | Complete student predictor | Pass/fail classification |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Neural Networks = stacked logistic regression!
# Each neuron: z = wx + b
# Each activation: a = sigmoid(z)  ← logistic!
# Deep learning = MANY logistic regressions!

# 2. Logistic Regression IS a one-layer neural net!
# Input → Linear → Sigmoid → Output
# Same as: Input → Dense(1) → sigmoid → Output (Keras)

# 3. Softmax = multi-class logistic regression
# Used in the OUTPUT LAYER of every classifier NN!

# 4. Log Loss = Binary Cross-Entropy
# The SAME loss function used in deep learning!
# Keras: model.compile(loss='binary_crossentropy')
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Using predict() when you need probabilities:**
```python
# Wrong — gives only 0 or 1!
y_pred = model.predict(X_test)  # ❌ no probabilities!

# Correct — gives probability scores!
y_prob = model.predict_proba(X_test)[:, 1]  # ✅
# Use probabilities for threshold tuning!
```

**Mistake 2 — Not scaling features:**
```python
# Wrong — gradient descent struggles!
model.fit(X_raw, y)  # ❌ salary dominates age!

# Correct!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y_train)  # ✅
```

**Mistake 3 — Using accuracy on imbalanced data:**
```python
# If 95% students pass — predicting "pass" always
# gives 95% accuracy but is USELESS!
# Use F1, Precision, Recall for imbalanced classes!
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred)  # ✅
```

---

## 💎 Important Realizations

1. **Logistic Regression = Linear Regression + Sigmoid**
   The ONLY difference is the sigmoid function!
   All the math — weights, gradient descent — is same!

2. **Log Loss connects to Information Theory**
   It measures how surprised the model is by the truth!
   This is the same cross-entropy used in every NN!

3. **predict_proba() is more powerful than predict()**
   Raw probabilities let you tune the threshold!
   Medical: lower threshold (catch all disease cases)
   Spam: higher threshold (don't block real emails)

4. **Logistic Regression should always be baseline**
   Before trying Random Forest or Neural Networks —
   always try Logistic Regression first!
   Fast, interpretable, often good enough!

---

## 🎯 Next Goal

- Decision Trees — split data using information gain!
- Non-linear boundaries — beyond what LR can do!
- Foundation of Random Forest!

---

*Day 53 complete — Classification unlocked! 🎯🔥*

