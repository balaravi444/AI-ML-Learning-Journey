# Day 54 — Decision Trees 🚀

**Date:** 11 July 2026
**Time Spent:** (3 hours)
**Resource Used:** [Scikit-learn Docs](https://scikit-learn.org/) | [StatQuest](https://www.youtube.com/@statquest)

---

## 📚 Topics Covered

- What is a Decision Tree
- How trees split data — Information Gain & Gini
- Entropy and Information Theory
- Building a tree from scratch
- Sklearn DecisionTreeClassifier
- Overfitting in Decision Trees
- Pruning strategies
- Feature importance
- Regression Trees
- Real application — student performance

---

## 🔑 What is a Decision Tree?
A flowchart that asks questions about features
and makes decisions at each node!           study_hours > 6?
          /               
YES                NO
/                    
attendance > 70?         prev_score > 60?
/      \               /           
YES       NO           YES            NO
PASS      FAIL         PASS           FAIL

**So what? Why does this matter?**
Decision Trees are:
→ Completely interpretable (you can explain every decision!)
→ Foundation of Random Forest (Day 55)
→ Foundation of XGBoost (most powerful ML algorithm)
→ Used in medical diagnosis, fraud detection, credit scoring"If study > 6hrs AND attendance > 70% → PASS"
A human can understand and verify this! 🔥
---

## 🔑 How Trees Split — Entropy & Information Gain

### Entropy — measure of impurity/disorder
Entropy = -Σ p(c) * log₂(p(c))Pure node (all same class):
entropy = 0 (no disorder!)Perfectly mixed node (50/50):
entropy = 1 (maximum disorder!)Formula for binary classification:
H = -p*log₂(p) - (1-p)*log₂(1-p)

### Information Gain — how good is a split?
IG = H(parent) - [weighted average H(children)]Best split = MAXIMUM information gain!
Reduces disorder the most!
### Gini Impurity — alternative to entropy
Gini = 1 - Σ p(c)²Pure node: Gini = 0
Mixed node: Gini approaches 0.5Sklearn uses Gini by default (faster to compute!)
Entropy and Gini give similar results in practice
---

## 🔑 The Splitting Algorithm
For each feature:
For each possible threshold:
Calculate Information Gain of this splitPick the feature + threshold with HIGHEST IG
Split data into left and right branches
Repeat recursively for each branch!Stop when:
→ max_depth reached
→ min_samples_split not met
→ node is pure (all same class)
→ no improvement possible
---

## 🔑 Overfitting — The Main Problem

```python
# Unconstrained tree — OVERFITS badly!
from sklearn.tree import DecisionTreeClassifier

# Bad — memorizes training data!
model = DecisionTreeClassifier()
# train_acc ≈ 1.00, test_acc ≈ 0.72 → overfit!

# Better — constrain tree depth!
model = DecisionTreeClassifier(
    max_depth=5,           # max 5 levels deep
    min_samples_split=20,  # need 20 samples to split
    min_samples_leaf=10,   # need 10 samples in leaf
    max_features='sqrt'    # use sqrt(n_features)
)
```

**Detect overfitting:**
train_acc = 0.99, test_acc = 0.72 → OVERFIT!
Gap > 0.10 → reduce max_depth!
---

## 🔑 Feature Importance

```python
model.fit(X_train, y_train)

# Feature importance = total IG contribution
importances = model.feature_importances_
# Higher = more important for predictions!

# Plot
for feat, imp in zip(feature_names, importances):
    print(f"{feat}: {imp:.3f}")
```

**So what? Why does this matter?**
This tells you WHICH features actually matter!
Same as correlation analysis from Day 45 —
but learned from the data automatically! 🔥

---

## 🔑 Regression Trees

```python
from sklearn.tree import DecisionTreeRegressor

# Same algorithm — but predicts average value
# instead of majority class at each leaf!
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # continuous values!
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Entropy + Gini from scratch | Information theory |
| 2 | Decision Tree from scratch | Recursive splitting |
| 3 | Sklearn classifier | fit, predict, feature_importance |
| 4 | Hyperparameter tuning | max_depth, min_samples |
| 5 | Regression tree | Continuous prediction |
| 6 | Student performance tree | Real application |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Random Forest = ensemble of Decision Trees!
# (Day 55 — tomorrow!)
from sklearn.ensemble import RandomForestClassifier
# 100 trees vote → more accurate + less overfit!

# 2. XGBoost = boosted Decision Trees
# Most powerful algorithm for tabular data!
# Kaggle competitions are won with XGBoost!

# 3. Feature importance from trees
# MUCH more powerful than correlation!
# Captures non-linear relationships!

# 4. Decision Trees explain ML to stakeholders
# "The model says PASS because:
#   study_hours > 6 AND attendance > 75%"
# Doctors, judges, banks need this explainability!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Not pruning the tree:**
```python
# Wrong — perfect train, bad test!
model = DecisionTreeClassifier()  # ❌ no limits!
model.fit(X_train, y_train)
# train_acc = 1.00, test_acc = 0.72 (overfit!)

# Correct — add constraints!
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10)  # ✅
```

**Mistake 2 — Not using feature importance:**
```python
# Missing powerful insight!
model.fit(X_train, y_train)
# What drives predictions? No idea! ❌

# Correct — always check importance!
importances = model.feature_importances_
# Now you know what matters! ✅
```

---

## 💎 Important Realizations

1. **Decision Trees are pure recursive algorithms**
   Day 25 (Recursion) is literally inside every tree!
   build_tree() calls itself on left and right subsets!

2. **Entropy is from information theory**
   Shannon entropy — same formula used in
   data compression, NLP, and cryptography!

3. **Gini vs Entropy — doesn't matter much**
   In practice both give similar trees.
   Gini is slightly faster to compute.
   Sklearn uses Gini by default.

4. **Unpruned tree = memorized training data**
   Deepest tree always gets 100% train accuracy.
   That's not learning — that's cheating!
   max_depth=5 is often a good starting point.

---

## 🎯 Next Goal

- Random Forest — ensemble of Decision Trees!
- More powerful, less overfit, better accuracy!
- The most used ML algorithm in industry!

---

*Day 54 complete — Decision Trees mastered! 🌳🔥*
