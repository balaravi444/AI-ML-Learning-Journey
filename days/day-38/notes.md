# Day 38 — Pandas: DataFrames 🚀

**Date:** 25 June 2026
**Time Spent:** (3 hours)
**Resource Used:** [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas) | [Pandas Docs](https://pandas.pydata.org/docs/)

---

## 📚 Topics Covered

- What is Pandas and why it exists
- Series — 1D labeled data
- DataFrame — 2D labeled data
- Creating DataFrames
- Reading CSV/JSON files
- Selecting rows and columns
- Filtering data
- Basic statistics
- Real ML data loading

---

## 🔑 Why Pandas Exists

```python
# Without Pandas — loading data is painful!
import csv
data = []
with open("students.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Average marks? Need to write more code...
marks = [int(row['marks']) for row in data]
avg = sum(marks) / len(marks)

# With Pandas — ONE LINE!
import pandas as pd
df = pd.read_csv("students.csv")
avg = df['marks'].mean()  # done! 🔥
```

**So what? Why does this matter?**
Every Kaggle competition starts with Pandas!
Every data scientist uses Pandas daily!
Pandas is the BRIDGE between raw data and ML!
Without Pandas — you spend 80% of time on data,
with Pandas — you spend 20% and focus on ML! 🔥

---

## 🔑 Series — 1D Labeled Data

```python
import pandas as pd

# Creating a Series
scores = pd.Series([85, 92, 78, 95, 88],
                   index=['Bala', 'Ravi',
                          'Kumar', 'Priya', 'Hari'])
print(scores)
# Bala     85
# Ravi     92
# Kumar    78
# Priya    95
# Hari     88

# Accessing
scores['Bala']      # 85
scores[scores > 90] # Ravi 92, Priya 95
scores.mean()       # 87.6
scores.max()        # 95
```

---

## 🔑 DataFrame — 2D Labeled Data

```python
# DataFrame = collection of Series (like a spreadsheet!)
data = {
    'name':   ['Bala', 'Ravi', 'Kumar'],
    'age':    [22, 23, 21],
    'marks':  [85, 92, 78],
    'course': ['AI', 'ML', 'DS']
}
df = pd.DataFrame(data)

print(df)
#     name  age  marks course
# 0   Bala   22     85     AI
# 1   Ravi   23     92     ML
# 2  Kumar   21     78     DS
```

**Key properties:**
```python
df.shape    # (3, 4) — rows, columns
df.columns  # Index(['name', 'age', 'marks', 'course'])
df.dtypes   # data type of each column
df.index    # RangeIndex(start=0, stop=3, step=1)
```

---

## 🔑 The 3 Essential Commands

```python
# 1. First look at data
df.head()    # first 5 rows
df.tail()    # last 5 rows
df.info()    # column types and non-null counts
df.describe() # statistical summary

# 2. Shape and structure
df.shape     # (rows, cols)
df.columns   # column names
df.dtypes    # data types

# 3. Missing values
df.isnull().sum()  # count missing per column
df.isnull().any()  # which columns have missing
```

---

## 🔑 Selecting Data

```python
# Select column — returns Series
df['marks']
df.marks        # same thing!

# Select multiple columns — returns DataFrame
df[['name', 'marks']]

# Select rows by index — .iloc (integer position)
df.iloc[0]      # first row
df.iloc[0:3]    # first 3 rows
df.iloc[0, 2]   # row 0, column 2

# Select rows by label — .loc (label based)
df.loc[0]           # row with index 0
df.loc[0, 'marks']  # row 0, 'marks' column
df.loc[0:2, ['name', 'marks']]  # rows 0-2, 2 columns

# Boolean filtering
df[df['marks'] > 80]           # students who passed
df[df['course'] == 'AI']       # AI students only
df[(df['marks'] > 80) &
   (df['age'] < 23)]           # multiple conditions
```

**ML Connection:**
```python
# Split features and target — EVERY ML project!
X = df[['age', 'marks', 'attendance']]  # features
y = df['passed']                         # target label

# This is EXACTLY how you prepare data for Scikit-learn!
```

---

## 🔑 Reading Real Files

```python
# CSV — most common!
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", index_col=0)
df = pd.read_csv("data.csv", nrows=1000)

# JSON
df = pd.read_json("data.json")

# Excel
df = pd.read_excel("data.xlsx")

# From URL!
df = pd.read_csv("https://url/data.csv")

# Save back to file
df.to_csv("output.csv", index=False)
df.to_json("output.json")
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Series operations | Labeled 1D data |
| 2 | DataFrame creation | Multiple sources |
| 3 | Selecting data | iloc, loc, boolean |
| 4 | Reading CSV files | Real data loading |
| 5 | Basic statistics | describe, mean, std |
| 6 | Filtering | Boolean conditions |
| 7 | Indian Job Market data | Real application |

---

## 🔗 How This Connects to AI/ML

```python
import pandas as pd

# EVERY Kaggle notebook starts like this:
df = pd.read_csv("train.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()

# Feature/target split
X = df.drop('target', axis=1)
y = df['target']

# This is lines 1-10 of EVERY ML project! 🔥
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — SettingWithCopyWarning:**
```python
# Wrong — modifying a copy!
df[df['marks'] > 80]['grade'] = 'A'  # ❌ warning!

# Correct — use .loc!
df.loc[df['marks'] > 80, 'grade'] = 'A'  # ✅
```

**Mistake 2 — iloc vs loc confusion:**
```python
# iloc = integer position (like array index)
df.iloc[0]    # FIRST row (position 0)

# loc = label based
df.loc[0]     # row with INDEX LABEL 0

# When index is default (0,1,2,...) they look same!
# But with custom index they differ!
df = df.set_index('name')
df.iloc[0]          # first row (whatever name it is)
df.loc['Bala']      # row where name='Bala'
```

**Mistake 3 — Chained indexing:**
```python
# Wrong — unpredictable behavior!
df['col1']['col2']  # ❌

# Correct — use .loc or .iloc!
df.loc[:, 'col1']   # ✅
```

---

## 💎 Important Realizations

1. **Pandas is built on NumPy**
   Every DataFrame column is a NumPy array!
   `.values` gives you the NumPy array back!

2. **Boolean indexing is the same as NumPy**
   Day 36's boolean indexing → works the same in Pandas!
   Prior knowledge transfers directly!

3. **Head, info, describe = first 3 commands always**
   Professional data scientists run these on
   EVERY new dataset before touching anything!

4. **Shape matters as much as content**
   Know your (rows, cols) before doing anything!
   Most ML bugs come from wrong shapes!

---

## 🎯 Next Goal

- Pandas Data Cleaning
- Handle missing values, duplicates, wrong types
- This is 80% of real data science work!

---

*Day 38 complete — Pandas basics mastered! 🐼🔥*
