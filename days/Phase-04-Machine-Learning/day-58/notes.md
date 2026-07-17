# Day 58 — Cross Validation + Hyperparameter Tuning 🚀

**Date:** 15 July 2026
**Time Spent:** (3 hours)
**Resource Used:** [Scikit-learn Docs](https://scikit-learn.org/) | [Kaggle Learn](https://www.kaggle.com/learn)

---

## 📚 Topics Covered

- Why single train/test split is unreliable
- K-Fold Cross Validation
- Stratified K-Fold
- Leave-One-Out CV
- Cross-validation for time series
- Hyperparameter vs model parameters
- GridSearchCV — exhaustive search
- RandomizedSearchCV — smarter search
- Early stopping
- Nested cross-validation
- Real application — student predictor tuning

---

## 🔑 Why Single Split is Unreliable
Split 1: train=[1-800], test=[801-1000] → accuracy=0.87
Split 2: train=[1-600,801-1000], test=[601-800] → accuracy=0.91
Split 3: train=[201-1000], test=[1-200] → accuracy=0.84
Which one is the "true" accuracy?
None of them! You got lucky or unlucky!
Cross-Validation:
→ Try ALL possible splits
→ Average the results
→ Get a STABLE estimate of true performance!
---

## 🔑 K-Fold Cross Validation

K=5 example with 1000 samples:
Fold 1: train=[200-1000], test=[1-200]    → score₁
Fold 2: train=[1-200, 401-1000], test=[201-400]  → score₂
Fold 3: train=[1-400, 601-1000], test=[401-600]  → score₃
Fold 4: train=[1-600, 801-1000], test=[601-800]  → score₄
Fold 5: train=[1-800], test=[801-1000]   → score₅
Final score = mean(score₁...score₅) ± std

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    cv=5,              # 5 folds
    scoring='f1')      # metric to use

print(f"Mean: {scores.mean():.4f}")
print(f"Std:  {scores.std():.4f}")
# Low std = stable model! High std = unstable!
```

---

## 🔑 Stratified K-Fold
Regular KFold: splits randomly
→ some folds might have 0% class 1!
Stratified KFold: preserves class ratio in EACH fold!
If overall 30% pass → each fold ~30% pass
ALWAYS use Stratified for classification! 🔥
```python
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42)

scores = cross_val_score(
    model, X, y, cv=cv, scoring='f1')
```

---

## 🔑 Hyperparameters vs Parameters
Model Parameters (learned from data):
→ Linear Regression weights w₁, w₂, ...
→ Decision Tree split thresholds
→ Neural Network weights
→ YOU DON'T SET THESE — model learns them!
Hyperparameters (set by YOU before training):
→ max_depth in Decision Tree
→ n_estimators in Random Forest
→ C and kernel in SVM
→ K in KNN
→ learning_rate in Neural Networks
→ YOU SET THESE — model doesn't learn them!
Hyperparameter Tuning = finding the best
hyperparameters using cross-validation!

---

## 🔑 GridSearchCV — Exhaustive Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [5, 10, None],
    'model__min_samples_leaf': [1, 5, 10]
}
# Total combinations: 3 × 3 × 3 = 27
# With 5-fold CV: 27 × 5 = 135 model fits!

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,        # use all CPU cores
    verbose=1)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
```

**When to use:**
Small param grid (< 100 combinations) → GridSearchCV

---

## 🔑 RandomizedSearchCV — Smarter Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'model__n_estimators': randint(50, 500),
    'model__max_depth': randint(3, 20),
    'model__min_samples_leaf': randint(1, 20),
    'model__max_features': uniform(0.3, 0.7)
}

random_search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=50,      # try 50 random combos
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42)

random_search.fit(X_train, y_train)
```

**When to use:**
Large param grid (100+ combinations) → RandomizedSearchCV
→ Often finds equally good results in fraction of time!

---

## 🔑 Nested Cross Validation
Outer CV: estimates generalization performance
Inner CV: selects best hyperparameters
Without nesting → OPTIMISTIC bias!
With nesting    → Honest estimate!
Use when you need to report honest performance
after hyperparameter tuning! 🔥

```python
from sklearn.model_selection import cross_val_score

# Inner CV — hyperparameter selection
inner_cv = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=inner_cv, scoring='f1')

# Outer CV — performance estimation
outer_cv = StratifiedKFold(n_splits=5)
nested_scores = cross_val_score(
    grid_search, X, y,
    cv=outer_cv, scoring='f1')

print(f"Nested CV F1: {nested_scores.mean():.4f}")
# This is the HONEST performance estimate!
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Cross-validation types | KFold, Stratified, LOO |
| 2 | CV stability analysis | mean ± std |
| 3 | GridSearchCV | Exhaustive tuning |
| 4 | RandomizedSearchCV | Efficient tuning |
| 5 | Nested CV | Honest evaluation |
| 6 | Complete tuned pipeline | Production-ready |

---

## 🔗 How This Connects to AI/ML

```python
# Cross-validation is used EVERYWHERE!

# 1. Model selection
# Compare RF vs SVM using CV (not test set!)
# Test set is SACRED — only use at the very end!

# 2. Feature selection
# Select features based on CV score
# Not test set performance!

# 3. Early stopping in neural networks
# Monitor val_loss across CV folds
# Stop when validation loss increases!

# 4. Kaggle competitions
# Local CV score should match leaderboard
# "My CV gives 0.923 — LB should be similar"

# 5. Student Performance Predictor (Day 59!)
# Tune the final model using GridSearchCV
# Then evaluate ONCE on test set!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Using test set for tuning:**
```python
# Wrong — data leakage!
for params in param_grid:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # ❌ peeking!
    # Now X_test influenced your model!

# Correct — tune on train/val, evaluate once on test!
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_train, y_train)
final_score = grid.score(X_test, y_test)  # ✅ once!
```

**Mistake 2 — Not fitting on full train after CV:**
```python
# Wrong — using cross-validated model on test!
grid.fit(X_train, y_train)
# best_estimator_ is already refitted on all X_train ✅
# GridSearchCV does this automatically!

# But if you do manual CV:
best_model.fit(X_train, y_train)  # refit on ALL train!
score = best_model.score(X_test, y_test)
```

**Mistake 3 — Not using Pipeline in GridSearchCV:**
```python
# Wrong — preprocessing not inside CV loop!
X_scaled = scaler.fit_transform(X_train)  # ❌ leakage!
grid = GridSearchCV(model, params, cv=5)
grid.fit(X_scaled, y_train)

# Correct — Pipeline handles it automatically!
pipeline = Pipeline([('scaler', StandardScaler()),
                    ('model', model)])
grid = GridSearchCV(pipeline, params, cv=5)  # ✅
grid.fit(X_train, y_train)  # no leakage!
```

---

## 💎 Important Realizations

1. **The test set is sacred**
   Use it ONCE at the very end.
   Never tune based on test set performance.
   CV is your validation — test is your final exam!

2. **RandomizedSearchCV often beats GridSearchCV**
   50 random trials often finds better params
   than exhaustive search of 500 combinations!
   Because the search space is high-dimensional!

3. **CV std tells you about model stability**
   mean=0.89, std=0.01 → stable model ✅
   mean=0.89, std=0.08 → unstable — be careful! ⚠️

4. **More folds = more reliable but slower**
   5-fold: standard for large datasets
   10-fold: better for smaller datasets
   LOO: very small datasets only (slow!)

---

## 🎯 Next Goal

- Student Performance Predictor project starts!
- Apply EVERYTHING from Days 51-58
- Real product that helps teachers! 🎓

---

*Day 58 complete — Tuning mastered! 🎯🔥*


