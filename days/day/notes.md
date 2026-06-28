# Day 40 — Pandas: GroupBy & Aggregations 🚀

**Date:** 27 June 2026
**Time Spent:** (3 hours)
**Resource Used:** [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas) | [Pandas Docs](https://pandas.pydata.org/docs/)

---

## 📚 Topics Covered

- GroupBy — split-apply-combine pattern
- Aggregation functions
- Multiple aggregations
- GroupBy with transform
- Pivot tables
- Cross-tabulation
- Real job market analysis

---

## 🔑 What is GroupBy?

The most powerful data analysis tool in Pandas!
Split → Apply → Combine
Split:   Group rows by category

Apply:   Calculate statistics per group
Combine: Put results back together

**Real world:**
```python
# "Average salary by job title" = GroupBy!
df.groupby('job_title')['salary'].mean()

# "Total sales by city per month" = GroupBy!
df.groupby(['city', 'month'])['sales'].sum()
```

**So what? Why does this matter?**
Every business question is a GroupBy!
"Which city has highest avg salary?"
"Which skills are most demanded?"
"How does salary change with experience?"
ALL answered with GroupBy in ONE line! 🔥

---

## 🔑 Basic GroupBy

```python
import pandas as pd

df = pd.read_csv("jobs.csv")

# Group by one column
df.groupby('city')['salary'].mean()

# Group by multiple columns
df.groupby(['city', 'role'])['salary'].mean()

# Multiple aggregations
df.groupby('city')['salary'].agg(['mean', 'min', 'max', 'count'])

# Named aggregations
df.groupby('city').agg(
    avg_salary=('salary', 'mean'),
    max_salary=('salary', 'max'),
    job_count=('salary', 'count')
)
```

---

## 🔑 Common Aggregation Functions

```python
group = df.groupby('city')['salary']

group.mean()    # average
group.sum()     # total
group.min()     # minimum
group.max()     # maximum
group.count()   # count (non-null)
group.size()    # count (all rows)
group.std()     # standard deviation
group.median()  # median
group.first()   # first value
group.last()    # last value
group.nunique() # unique count
```

---

## 🔑 Transform vs Agg

```python
# agg() — returns one row per group
avg_by_city = df.groupby('city')['salary'].mean()
# returns: city → avg_salary (fewer rows!)

# transform() — returns same shape as original
df['city_avg'] = df.groupby('city')['salary'].transform('mean')
# adds column with group average to EVERY row!

# Real use — normalize within groups!
df['salary_normalized'] = (
    df.groupby('city')['salary']
    .transform(lambda x: (x - x.mean()) / x.std())
)
```

**ML Connection:**
```python
# Group normalization is used in ML preprocessing!
# Normalize features within each category
# Removes category-level bias from features!
```

---

## 🔑 Pivot Tables

```python
# Excel-style pivot tables!
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='job_title',
    aggfunc='mean',
    fill_value=0
)
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Basic GroupBy | split-apply-combine |
| 2 | Multiple aggregations | named agg |
| 3 | Transform | same-shape output |
| 4 | Pivot tables | Excel-style analysis |
| 5 | Job market analysis | real GroupBy |
| 6 | Salary insights | business questions |

---

## 🔗 How This Connects to AI/ML

```python
# GroupBy IS feature engineering for ML!

# 1. Target encoding (encode categoricals with stats)
city_salary_map = df.groupby('city')['salary'].mean()
df['city_encoded'] = df['city'].map(city_salary_map)

# 2. Aggregated features — used in every Kaggle competition!
df['user_avg_purchase'] = (
    df.groupby('user_id')['purchase_amount']
    .transform('mean'))

# 3. Anomaly detection with GroupBy!
df['z_score'] = (
    df.groupby('city')['salary']
    .transform(lambda x: (x - x.mean()) / x.std()))

# Flag outliers within each city's salary distribution
df['is_outlier'] = df['z_score'].abs() > 3
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Forgetting reset_index():**
```python
# GroupBy returns Series with group as index!
result = df.groupby('city')['salary'].mean()
# result.index = ['Bangalore', 'Delhi', ...]

# If you want it as a regular column:
result = (df.groupby('city')['salary']
          .mean()
          .reset_index())  # ✅ now 'city' is a column
```

**Mistake 2 — agg vs transform confusion:**
```python
# agg — fewer rows (one per group)
df.groupby('city')['salary'].mean()
# → 5 rows for 5 cities

# transform — same rows as original
df.groupby('city')['salary'].transform('mean')
# → same number of rows as df!
```

---

## 💎 Important Realizations

1. **GroupBy is how businesses answer questions**
   Every analytics dashboard uses GroupBy!
   The Indian Job Market Analyzer = GroupBy queries!

2. **Transform enables powerful feature engineering**
   Adding group statistics as features = 10x better ML models!
   This is used in EVERY Kaggle winning solution!

3. **Pivot tables = GroupBy with reshape**
   Same data, different view — powerful for visualization!

---

## 🎯 Next Goal

- Matplotlib & Seaborn — Data Visualization
- Turn GroupBy results into beautiful charts!
- Visual proof for the Job Market Analyzer!

---

*Day 40 complete — GroupBy mastered! 🔥*
