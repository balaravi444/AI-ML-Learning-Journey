# Day 44 — Feature Engineering 🚀

**Date:** 01 July 2026
**Time Spent:** (3 hours)
**Resource Used:** [Kaggle Learn](https://www.kaggle.com/learn) | [Scikit-learn Docs](https://scikit-learn.org/)

---

## 📚 Topics Covered

- What is Feature Engineering
- Why features matter more than algorithms
- Encoding categorical variables
- Feature scaling and normalization
- Creating new features from existing ones
- Feature selection
- Polynomial features
- Log transformations
- Real ML pipeline preparation

---

## 🔑 What is Feature Engineering?
Raw data → Feature Engineering → ML-ready featuresThe art of transforming raw data into
features that ML models can understand!"Better features beat better algorithms"
— Every Kaggle grandmaster ever!

**So what? Why does this matter?**
ML models see only NUMBERS — not text!
"Bangalore" → must become a number
"2024-01-15" → must become useful numbers
"salary_lpa" → might need log transformFeature engineering is how you TEACH
the model what's important in your data!80% of Kaggle competition success =
good feature engineering, not algorithm choice!

---

## 🔑 Encoding Categorical Variables

### Label Encoding — Ordinal categories
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])
# Bangalore=0, Delhi=1, Hyderabad=2, Mumbai=3, Pune=4
```

### One-Hot Encoding — Nominal categories
```python
# pd.get_dummies — simplest way!
df_encoded = pd.get_dummies(df, columns=['city'],
                             drop_first=True)

# Sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df[['city']])
```

### Target Encoding — Mean of target per category
```python
# Replace category with mean target value
city_mean_salary = df.groupby(
    'city')['salary_lpa'].mean()
df['city_target_encoded'] = df['city'].map(
    city_mean_salary)
# Best for high-cardinality categories!
```

**When to use which:**
Label Encoding → ordinal (Low, Mid, High)
One-Hot        → nominal, few categories (<10)
Target         → nominal, many categories (10+)

---

## 🔑 Feature Scaling

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler)

# StandardScaler — Z-score normalization
# Use for: Linear models, SVM, Neural Networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Mean=0, Std=1

# MinMaxScaler — Scale to 0-1 range
# Use for: Neural Networks, KNN
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Min=0, Max=1

# RobustScaler — Uses median and IQR
# Use for: Data with outliers!
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
# Robust to outliers!
```

**ML Connection:**
Linear Regression → StandardScaler (required!)
Random Forest     → No scaling needed!
Neural Network    → MinMaxScaler
Has outliers      → RobustScaler
---

## 🔑 Creating New Features

```python
# 1. Interaction features
df['exp_times_skills'] = (df['experience_years'] *
                           df['skills_count'])

# 2. Ratio features
df['salary_per_year_exp'] = (df['salary_lpa'] /
                              (df['experience_years'] + 1))

# 3. Log transform — fix skewness!
df['salary_log'] = np.log1p(df['salary_lpa'])

# 4. Binning — group continuous into categories
df['exp_level'] = pd.cut(
    df['experience_years'],
    bins=[0, 2, 5, 8, 12],
    labels=['Junior', 'Mid', 'Senior', 'Expert'])

# 5. Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,
                           include_bias=False)
X_poly = poly.fit_transform(X[['experience']])
```

---

## 🔑 Feature Selection

```python
# Method 1 — Correlation threshold
correlation = df.corr()['salary_lpa'].abs()
selected = correlation[correlation > 0.3].index

# Method 2 — Feature importance from tree model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)
importances = pd.Series(
    model.feature_importances_,
    index=X.columns).sort_values(ascending=False)

# Method 3 — SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X, y)
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Categorical encoding | OHE, Label, Target |
| 2 | Feature scaling | Standard, MinMax, Robust |
| 3 | New feature creation | Interactions, ratios, log |
| 4 | Feature selection | Correlation, importance |
| 5 | Complete pipeline | ML-ready dataset |
| 6 | ArthAI features | Financial feature engineering |

---

## 🔗 How This Connects to AI/ML

```python
# This IS the ML preprocessing pipeline!
# Every production ML system has this!

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define transformers
numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(
        handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

# Full pipeline!
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

full_pipeline.fit(X_train, y_train)
# This is how Netflix, Uber, Google do it!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Fitting scaler on test data:**
```python
# Wrong — data leakage!
scaler.fit_transform(X_train)  # ✅
scaler.fit_transform(X_test)   # ❌ never fit on test!

# Correct — fit only on train!
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ✅
```

**Mistake 2 — One-hot encoding without drop_first:**
```python
# Creates multicollinearity (dummy variable trap)!
pd.get_dummies(df, columns=['city'])  # ❌

# Fix — drop one category
pd.get_dummies(df, columns=['city'],
               drop_first=True)  # ✅
```

**Mistake 3 — Log transform with zeros:**
```python
np.log(df['salary'])    # ❌ fails on 0!
np.log1p(df['salary'])  # ✅ log(1+x) handles 0!
```

---

## 💎 Important Realizations

1. **Features matter more than algorithms**
   A Random Forest with good features beats
   a Neural Network with bad features — always!

2. **Never fit preprocessors on test data**
   This is called data leakage — it makes
   your model think it's better than it is!
   ALWAYS fit only on train, transform both!

3. **Target encoding is powerful but risky**
   Can cause data leakage if not done carefully!
   Always use cross-validation with target encoding!

4. **Sklearn pipelines prevent leakage automatically**
   This is why pipelines are used in production —
   they handle train/test split correctly!

---

## 🎯 Next Goal

- Statistics for ML
- Probability, hypothesis testing
- The math foundation we've been building toward!

---

*Day 44 complete — Feature Engineering mastered! 🔧🔥*
