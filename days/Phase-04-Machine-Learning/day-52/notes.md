# Day 52 — Linear Regression 🚀

**Date:** 09 July 2026
**Time Spent:** (3/4 hours)
**Resource Used:** [Scikit-learn Docs](https://scikit-learn.org/) | [StatQuest Linear Regression](https://www.youtube.com/@statquest)

---

## 📚 Topics Covered

- What is Linear Regression
- Simple vs Multiple Linear Regression
- The math behind fit() — OLS
- Assumptions of Linear Regression
- Implementing from scratch with NumPy
- Sklearn implementation
- Evaluating Linear Regression
- Regularization — Ridge and Lasso
- Real application — salary prediction

---

## 🔑 What is Linear Regression?
Find the best straight line through data points!y = w₁x₁ + w₂x₂ + ... + wₙxₙ + bWhere:
y  = prediction (salary)
x  = features (experience, skills)
w  = weights (learned during training)
b  = bias/intercept
**So what? Why does this matter?**
Linear Regression is the FOUNDATION of ML!
Understanding it means understanding:
→ Neural Networks (stacked linear layers!)
→ Logistic Regression (linear + sigmoid)
→ SVMs (linear decision boundary)
→ Every regression problem you'll ever face!

---

## 🔑 The Math — OLS (Ordinary Least Squares)

**Goal:**

Find weights w that minimize:
Loss = Σ(y_true - y_pred)²
= Σ(y - Xw)²This is Mean Squared Error × n!
**Closed form solution:**
w = (X.T @ X)⁻¹ @ X.T @ yThis is what sklearn's fit() computes!
One matrix equation = perfect weights
**Geometric interpretation:**
We're finding the projection of y
onto the column space of X!
Pure linear algebra from Day 37! 🔥

---

## 🔑 Assumptions of Linear Regression
Linearity
→ Relationship between X and y is linear
Independence
→ Observations are independent
Homoscedasticity
→ Constant variance of residuals
Normality of Residuals
   → Residuals ~ Normal(0, σ²)
No Multicollinearity
→ Features shouldn't be highly correlated

**Check assumptions:**
```python
# 1. Check linearity — scatter plot
# 2. Check residuals — plot residuals
# 3. Check normality — Q-Q plot or Shapiro test
# 4. Check multicollinearity — correlation matrix
```

---

## 🔑 Simple vs Multiple Linear Regression

```python
# Simple — one feature
salary = w * experience + b

# Multiple — many features
salary = w₁*experience + w₂*skills +
         w₃*city_encoded + w₄*remote + b
```

---

## 🔑 Ridge and Lasso Regression

### Ridge (L2 Regularization)
```python
# Adds penalty: λ * Σ(wᵢ²)
# Shrinks all weights toward zero
# Never zeros them out!
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)  # alpha = λ
```

### Lasso (L1 Regularization)
```python
# Adds penalty: λ * Σ|wᵢ|
# Can zero out weights completely!
# Built-in feature selection!
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
```

**When to use:**
Ridge → When all features matter
(keep them all, just shrink)
Lasso → When some features are irrelevant
(eliminate them automatically!)
---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Linear Regression from scratch | OLS math |
| 2 | Sklearn Linear Regression | fit, predict |
| 3 | Assumption checking | Residual analysis |
| 4 | Ridge Regression | L2 regularization |
| 5 | Lasso Regression | L1 + feature selection |
| 6 | Salary prediction | Real application |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Neural Networks = stacked linear regression!
# Each neuron: z = w₁x₁ + w₂x₂ + b (LINEAR!)
# Then activation function: a = ReLU(z)
# Multiple layers = deep linear + nonlinear!

# 2. Gradient Descent (when OLS isn't possible)
# For large datasets — can't invert huge matrix!
for epoch in range(n_epochs):
    y_pred = X @ weights + bias
    loss = MSE(y_true, y_pred)
    # Update weights using gradient
    weights -= lr * gradient

# This IS how neural networks train!

# 3. Logistic Regression (Day 53):
# Linear Regression → add sigmoid function
# = binary classification!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Not scaling features:**
```python
# Wrong — salary (₹30000) dominates age (25)!
model.fit(X_raw, y)  # ❌

# Correct — scale first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y_train)  # ✅
```

**Mistake 2 — Interpreting R² wrongly:**
```python
# R² = 0.85 does NOT mean
# "model is 85% accurate"!
# It means "model explains 85% of variance"
# MAE tells you actual prediction error!
```

**Mistake 3 — Multicollinearity:**
```python
# If experience and age are highly correlated:
# Coefficients become unstable!
# Fix: drop one correlated feature
# Or use Ridge regression!
```

---

## 💎 Important Realizations

1. **Linear Regression is NOT just a simple model**
   With polynomial features it becomes powerful!
   With regularization it handles complex data!
   This is the foundation everything is built on!

2. **OLS solution is elegant pure math**
   w = (X.T @ X)⁻¹ @ X.T @ y
   We derived this in Day 37 (Linear Algebra)!
   The entire model is ONE matrix equation!

3. **Lasso does feature selection automatically**
   Set irrelevant weights to exactly zero!
   This is why Lasso is so powerful in practice —
   it finds the most important features for you!

4. **When Linear Regression fails → next algorithm**
   Non-linear relationship → polynomial features
   Many irrelevant features → Lasso
   Highly correlated features → Ridge
   Complex patterns → Day 54 Decision Trees!

---

## 🎯 Next Goal

- Logistic Regression — classification!
- The same linear math + sigmoid function
- Binary classification for student pass/fail!

---

*Day 52 complete — Linear Regression mastered! 📈🔥*
