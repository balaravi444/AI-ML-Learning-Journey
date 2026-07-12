# Day 46 — Data Preprocessing Pipeline 🚀

**Date:** 04 July 2026
**Time Spent:** (4 hours)
**Resource Used:** [Scikit-learn Docs](https://scikit-learn.org/) | [Kaggle Learn](https://www.kaggle.com/learn)

---

## 📚 Topics Covered

- What is a preprocessing pipeline
- Sklearn Pipeline class
- ColumnTransformer
- Custom transformers
- Handling train/test splits correctly
- Saving and loading pipelines
- Production-ready preprocessing
- Full end-to-end pipeline

---

## 🔑 What is a Preprocessing Pipeline?
Raw Data
↓
Clean (remove nulls, duplicates)
↓
Encode (categories → numbers)
↓
Scale (normalize features)
↓
Select (keep best features)
↓
ML Model
↓
Predictions

**Without Pipeline — danger!**
```python
# Manual preprocessing — BREAKS in production!
scaler.fit_transform(X_train)  # fit on train ✅
scaler.fit_transform(X_test)   # fit on test ❌ LEAKAGE!
encoder.fit_transform(X_train) # fit on train ✅
encoder.fit_transform(X_test)  # fit on test ❌ LEAKAGE!
```

**With Pipeline — safe!**
```python
# Pipeline handles everything correctly!
pipeline.fit(X_train, y_train)  # fits ALL steps
pipeline.predict(X_test)        # transforms correctly!
# NO leakage possible! ✅
```

**So what? Why does this matter?**
In production — your model sees new data every day!
Pipeline ensures same transformations
every single time — consistently and correctly!
Netflix, Uber, Google all use pipelines in production! 🔥

---

## 🔑 Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Chain steps in order!
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

# fit() runs ALL steps on training data!
pipeline.fit(X_train, y_train)

# predict() runs transformations then predicts!
predictions = pipeline.predict(X_test)
```

---

## 🔑 ColumnTransformer

Different preprocessing for different column types!

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder)
from sklearn.impute import SimpleImputer

numerical_features = ['experience', 'skills', 'rating']
categorical_features = ['city', 'job_title']

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine!
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])
```

---

## 🔑 Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom log transformer."""

    def fit(self, X, y=None):
        return self  # nothing to learn!

    def transform(self, X):
        return np.log1p(X)

class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap outliers using IQR."""

    def fit(self, X, y=None):
        self.Q1 = np.percentile(X, 25, axis=0)
        self.Q3 = np.percentile(X, 75, axis=0)
        self.IQR = self.Q3 - self.Q1
        self.lower = self.Q1 - 1.5 * self.IQR
        self.upper = self.Q3 + 1.5 * self.IQR
        return self

    def transform(self, X):
        return np.clip(X, self.lower, self.upper)
```

---

## 🔑 Saving and Loading Pipelines

```python
import joblib

# Save pipeline (model + all transformers!)
joblib.dump(pipeline, 'salary_predictor.pkl')

# Load pipeline later
loaded_pipeline = joblib.load('salary_predictor.pkl')

# Predict on new data — transformations happen automatically!
new_data = pd.DataFrame({
    'experience': [5],
    'city': ['Bangalore'],
    'skills_count': [7]
})
prediction = loaded_pipeline.predict(new_data)
```

**So what? Why does this matter?**
This is EXACTLY how ML models are deployed!
One .pkl file → entire ML system!
ArthAI could use a salary predictor pipeline! 🔥

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Basic Pipeline | Chain steps |
| 2 | ColumnTransformer | Different cols |
| 3 | Custom transformer | BaseEstimator |
| 4 | Full ML pipeline | Train + evaluate |
| 5 | Save/load pipeline | joblib |
| 6 | Production pipeline | Real-world ready |

---

## 🔗 How This Connects to AI/ML

```python
# This IS how production ML looks!
# Companies like Swiggy, Razorpay, Flipkart
# have pipelines exactly like this!

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Full production pipeline
production_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])),
    ('feature_selection', SelectKBest(k=10)),
    ('model', RandomForestRegressor(
        n_estimators=200))
])

# Train once
production_pipeline.fit(X_train, y_train)

# Save
joblib.dump(production_pipeline, 'model.pkl')

# Deploy — one line prediction!
prediction = production_pipeline.predict(new_data)
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Forgetting remainder='passthrough':**
```python
# ColumnTransformer drops columns not specified!
preprocessor = ColumnTransformer([
    ('num', scaler, num_features)
    # ❌ categorical columns DROPPED!
])

# Fix!
preprocessor = ColumnTransformer([
    ('num', scaler, num_features)
], remainder='passthrough')  # ✅ keeps rest!
# OR specify all columns explicitly
```

**Mistake 2 — Not using Pipeline for CV:**
```python
# Wrong — leakage in cross-validation!
X_scaled = scaler.fit_transform(X)  # ❌
cross_val_score(model, X_scaled, y)

# Correct — Pipeline handles CV correctly!
pipeline = Pipeline([('scaler', scaler),
                    ('model', model)])
cross_val_score(pipeline, X, y)  # ✅
```

---

## 💎 Important Realizations

1. **Pipeline = production-ready ML**
   Jupyter notebooks are for exploration.
   Pipelines are for production.
   The difference between a data scientist
   and an ML engineer!

2. **Custom transformers make pipelines flexible**
   Any transformation can be added to a pipeline!
   Log transform, domain-specific features,
   ArthAI financial ratios — all as transformers!

3. **joblib.dump is how models are deployed**
   One .pkl file = the entire ML system!
   Your FastAPI app loads it and predicts!
   This is how ArthAI salary predictor will work!

---

## 🎯 Next Goal

- Day 47 — Indian Job Market Analyzer!
- Apply EVERYTHING from Days 36-46!
- Real dataset. Real insights. Real product!

---

*Day 46 complete — Preprocessing Pipeline mastered! 🔧🔥*
