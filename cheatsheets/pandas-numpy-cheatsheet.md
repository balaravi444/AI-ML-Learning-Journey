# 📊 Pandas & NumPy Cheatsheet

Data manipulation reference for EDA and data science work.

---

## NumPy Basics

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
rng = np.arange(0, 10, 2)
lin = np.linspace(0, 1, 5)

arr.shape       # dimensions
arr.reshape(2, 2)
arr.mean(), arr.std(), arr.sum()

# Vectorized ops (no loops!)
arr2 = arr * 2
mask = arr[arr > 2]     # boolean indexing
```

## Pandas — Reading Data

```python
import pandas as pd

df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx")
df = pd.read_json("data.json")

df.head()
df.info()
df.describe()
df.shape
df.columns
df.dtypes
```

## Selecting & Filtering

```python
df["column"]                     # single column
df[["col1", "col2"]]             # multiple columns
df.loc[0, "col"]                 # by label
df.iloc[0, 1]                    # by position
df[df["age"] > 25]               # filter rows
df[(df["age"] > 25) & (df["city"] == "Bangalore")]
df.query("age > 25 and city == 'Bangalore'")
```

## Cleaning Data

```python
df.isnull().sum()                # count missing per column
df.dropna()                      # drop rows with NaN
df.fillna(0)                     # fill missing
df.fillna(df["col"].mean())      # fill with mean
df.drop_duplicates()
df.drop(columns=["col_to_drop"])
df.rename(columns={"old": "new"})
df["col"] = df["col"].astype(int)
```

## Transforming

```python
df["new_col"] = df["a"] + df["b"]
df["col"].apply(lambda x: x * 2)
df["col"].map({"yes": 1, "no": 0})
df["col"].str.lower()
df["col"].str.strip()
pd.to_datetime(df["date_col"])
```

## Grouping & Aggregating

```python
df.groupby("category")["value"].mean()
df.groupby("category").agg({"value": ["mean", "sum", "count"]})
df.pivot_table(values="sales", index="region", columns="month", aggfunc="sum")
df.sort_values("col", ascending=False)
```

## Merging & Joining

```python
pd.merge(df1, df2, on="id", how="left")   # left/right/inner/outer
pd.concat([df1, df2], axis=0)             # stack rows
pd.concat([df1, df2], axis=1)             # stack columns
```

## Quick Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

df["col"].hist()
sns.boxplot(x="category", y="value", data=df)
sns.heatmap(df.corr(), annot=True)
df.plot(kind="bar", x="category", y="value")
plt.show()
```

## Common Gotchas

- `df[df.col > 5]` returns a **view sometimes** — use `.copy()` to avoid `SettingWithCopyWarning`
- `groupby` + single column returns a Series, not DataFrame — use `.reset_index()`
- Always check `df.dtypes` before doing math — numbers stored as strings won't compute
- `inplace=True` modifies the original df — be careful, prefer reassignment
