# Day 39 — Pandas: Data Cleaning 🚀

**Date:** 26 June 2026
**Time Spent:** (4 hours)
**Resource Used:** [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas) | [Pandas Docs](https://pandas.pydata.org/docs/)

---

## 📚 Topics Covered

- Why data cleaning matters
- Handling missing values
- Removing duplicates
- Fixing data types
- Handling outliers
- String cleaning
- Renaming and reorganizing columns
- Real ML data preparation

---

## 🔑 Why Data Cleaning Matters

Real world data is MESSY!Problems you'll face every time:
→ Missing values (NaN, None, empty strings)
→ Duplicates (same row entered twice)
→ Wrong data types (salary stored as string!)
→ Inconsistent formats ("Bangalore" vs "bangalore")
→ Outliers (age = 999, salary = -50000)
→ Extra whitespace ("  Bala  " vs "Bala")"Data scientists spend 80% of time cleaning data"
— Every senior data scientist, ever!

**So what? Why does this matter?**

Dirty data = wrong ML model!
A model trained on messy data gives WRONG predictions!
Garbage In = Garbage Out (GIGO principle)
Data cleaning is the most important ML skill! 🔥

---

## 🔑 Handling Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Bala', 'Ravi', None, 'Priya'],
    'age':  [22, None, 21, 23],
    'salary': [35000, 45000, None, 60000]
})

# Detect missing values
df.isnull()          # True where NaN
df.isnull().sum()    # count per column
df.isnull().sum().sum()  # total missing

# Drop rows/columns with missing values
df.dropna()                    # drop rows with ANY null
df.dropna(how='all')           # drop rows with ALL null
df.dropna(subset=['age'])      # drop if 'age' is null
df.dropna(thresh=2)            # keep if at least 2 non-null

# Fill missing values
df.fillna(0)                   # fill all with 0
df['age'].fillna(df['age'].mean())   # fill with mean
df['name'].fillna('Unknown')   # fill with constant
df.fillna(method='ffill')      # forward fill
df.fillna(method='bfill')      # backward fill
```


**ML Strategy:**

Numerical missing → fill with mean/median
Categorical missing → fill with mode or "Unknown"
Too many missing (>50%) → drop the column
Few missing (<5%) → drop the rows

---

## 🔑 Removing Duplicates

```python
# Check duplicates
df.duplicated()              # True for duplicate rows
df.duplicated().sum()        # count of duplicates

# Remove duplicates
df.drop_duplicates()         # keep first occurrence
df.drop_duplicates(keep='last')  # keep last
df.drop_duplicates(
    subset=['email'])        # based on specific column
```

---

## 🔑 Fixing Data Types

```python
df = pd.DataFrame({
    'salary': ['35000', '45000', '60000'],  # should be int!
    'date': ['2026-01-01', '2026-02-15'],   # should be datetime!
    'is_remote': ['True', 'False', 'True']  # should be bool!
})

# Convert types
df['salary'] = df['salary'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['is_remote'] = df['is_remote'].map(
    {'True': True, 'False': False})

# Extract from datetime
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
```

**ML Connection:**
```python
# ML models need numbers — not strings!
# Categorical → numerical encoding
df['city_encoded'] = df['city'].map(
    {'Bangalore': 0, 'Mumbai': 1, 'Delhi': 2})

# Or use pandas get_dummies (one-hot encoding!)
df_encoded = pd.get_dummies(df, columns=['city'])
# This is EXACTLY what ML preprocessing does!
```

---

## 🔑 Handling Outliers

```python
# Detect outliers using IQR method
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['salary'] < lower) |
              (df['salary'] > upper)]

# Remove outliers
df_clean = df[(df['salary'] >= lower) &
              (df['salary'] <= upper)]

# Cap outliers (Winsorization)
df['salary'] = df['salary'].clip(lower, upper)
```

---

## 🔑 String Cleaning

```python
# Fix inconsistent strings
df['city'] = df['city'].str.strip()     # remove whitespace
df['city'] = df['city'].str.lower()     # lowercase
df['city'] = df['city'].str.title()     # Title Case
df['city'] = df['city'].str.replace(
    'Blr', 'Bangalore')                 # replace text

# Extract information
df['domain'] = df['email'].str.split('@').str[1]
df['first_name'] = df['name'].str.split().str[0]

# Check patterns
df['valid_email'] = df['email'].str.contains(
    r'@.*\.com', regex=True)
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Missing values | fillna, dropna strategies |
| 2 | Duplicates | drop_duplicates |
| 3 | Data types | astype, to_datetime |
| 4 | Outlier detection | IQR method |
| 5 | String cleaning | str methods |
| 6 | Complete pipeline | Indian job data |
| 7 | ML preparation | One-hot encoding |

---

## 🎯 LeetCode-style Data Problems Solved

| # | Problem | Method |
|---|---------|--------|
| 1 | Find % missing per column | isnull().sum()/len() |
| 2 | Salary as string → int | astype |
| 3 | "bangalore" and "Bangalore" → same | str.lower() |
| 4 | Remove salary outliers | IQR method |
| 5 | One-hot encode city column | get_dummies |

---

## 🔗 How This Connects to AI/ML

```python
# Complete ML data cleaning pipeline!
def clean_ml_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data for ML model training.
    This is what runs BEFORE every model.fit()!
    """
    df = df.copy()

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Fix data types
    df['salary'] = pd.to_numeric(
        df['salary'], errors='coerce')

    # 3. Handle missing values
    df['salary'].fillna(
        df['salary'].median(), inplace=True)
    df['city'].fillna('Unknown', inplace=True)

    # 4. Remove outliers
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['salary'] >= Q1 - 1.5*IQR) &
            (df['salary'] <= Q3 + 1.5*IQR)]

    # 5. Clean strings
    df['city'] = df['city'].str.strip().str.title()

    # 6. Encode categoricals
    df = pd.get_dummies(df, columns=['city'],
                         drop_first=True)

    return df
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Modifying original DataFrame:**
```python
# Wrong — modifies original!
df['salary'].fillna(0)  # ❌ doesn't save!

# Correct — use inplace or assign!
df['salary'].fillna(0, inplace=True)  # ✅
# OR
df['salary'] = df['salary'].fillna(0) # ✅
```

**Mistake 2 — Wrong missing value strategy:**
```python
# Wrong — filling numerical with 0 distorts mean!
df['salary'].fillna(0)  # ❌ 0 salary is misleading!

# Correct — fill with median (robust to outliers)!
df['salary'].fillna(df['salary'].median()) # ✅
```

**Mistake 3 — Not resetting index after dropping:**
```python
# After dropping rows, index has gaps!
df = df.dropna()
df.index  # 0, 2, 5, 7... (gaps!)

# Fix — reset index!
df = df.dropna().reset_index(drop=True)
df.index  # 0, 1, 2, 3... (clean!)
```

---

## 💎 Important Realizations

1. **Data cleaning is 80% of data science**
   Not glamorous. But without it — no ML!
   The best model on dirty data = wrong predictions!

2. **IQR outlier detection is industry standard**
   Used in finance, healthcare, ML everywhere!
   ArthAI uses it to flag unusual transactions!

3. **get_dummies = one-hot encoding**
   This is how every ML model handles categories!
   Scikit-learn's OneHotEncoder does the same thing!

4. **Always work on a copy**
   `df = df.copy()` before cleaning!
   Never mutate your original data!

---

## 🎯 Next Goal

- Pandas GroupBy and Aggregations
- This is how you find insights in data!
- Foundation of the Job Market Analyzer!

---

*Day 39 complete — Data Cleaning mastered! 🧹🔥*
