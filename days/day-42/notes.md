# Day 42 — Seaborn: Statistical Visualization 🚀

**Date:** 29 June 2026
**Time Spent:** ( 3 hours)
**Resource Used:** [Seaborn Docs](https://seaborn.pydata.org/) | [Kaggle Data Visualization](https://www.kaggle.com/learn/data-visualization)

---

## 📚 Topics Covered

- What is Seaborn and why use it over Matplotlib
- Distribution plots — histplot, kdeplot
- Categorical plots — boxplot, violinplot, barplot
- Relational plots — scatterplot, lineplot
- Heatmaps — correlation matrices
- Pairplot — multi-feature relationships
- Styling themes
- Real ML/ArthAI applications

---

## 🔑 Why Seaborn Over Matplotlib?

```python
# Matplotlib — verbose, manual styling
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(df['salary'], bins=20, color='green',
        edgecolor='white', alpha=0.8)
ax.set_title('Salary Distribution')
plt.show()

# Seaborn — concise, beautiful defaults!
import seaborn as sns
sns.histplot(data=df, x='salary', kde=True)
```

**So what? Why does this matter?**
Seaborn is BUILT ON Matplotlib but adds:
→ Beautiful default styles
→ Built-in statistical calculations
→ Direct DataFrame integration
→ Automatic legends and color schemes

Every Kaggle notebook uses Seaborn for EDA!
It's the professional standard for statistical plots! 🔥

---

## 🔑 Distribution Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Histogram with KDE (smooth curve)
sns.histplot(data=df, x='salary_lpa', kde=True,
             color='#10b981')

# KDE plot alone — smooth distribution
sns.kdeplot(data=df, x='salary_lpa', fill=True)

# Compare distributions across categories
sns.histplot(data=df, x='salary_lpa',
             hue='city', kde=True)
```

**ML Connection:**
```python
# Check if feature is normally distributed!
sns.histplot(df['feature'], kde=True)
# Normal → use as-is
# Skewed → apply log transform before ML!
```

---
Box shows:
→ Median (middle line)
→ Q1, Q3 (box edges)
→ Whiskers (1.5 * IQR)
→ Outlier dots beyond whiskers!

### Violin Plot — Box plot + distribution shape!
```python
sns.violinplot(data=df, x='city', y='salary_lpa')
```

### Bar Plot — Shows mean with confidence interval!
```python
sns.barplot(data=df, x='city', y='salary_lpa',
            estimator='mean')
```

**So what? Why does this matter?**
Box plots are the FASTEST way to spot outliers!
ArthAI uses box plots to flag unusual transactions!
Violin plots show if data is bimodal — important
for choosing the right ML algorithm!

---

## 🔑 Relational Plots

```python
# Scatter with automatic regression line
sns.scatterplot(data=df, x='experience',
                 y='salary_lpa', hue='city')

sns.regplot(data=df, x='experience',
            y='salary_lpa')  # adds trend line!

sns.lmplot(data=df, x='experience',
           y='salary_lpa', hue='city',
           col='remote')  # multi-panel!
```

---

## 🔑 Heatmaps — Correlation Matrix

The MOST USED Seaborn plot in ML!

```python
correlation = df[['salary', 'experience',
                   'rating', 'skills_count']].corr()

sns.heatmap(correlation, annot=True,
            cmap='Greens', fmt='.2f',
            linewidths=0.5)
```

**So what? Why does this matter?**
This is THE FIRST chart every ML engineer makes!
Shows which features are correlated (redundant)
Shows which features predict the target!
Used to select features BEFORE training models!

---

## 🔑 Pairplot — See Everything at Once

```python
sns.pairplot(df[['salary', 'experience',
                  'rating', 'city']],
             hue='city')
```

This creates a GRID of every feature vs every
other feature — scatter plots + histograms!
One line shows the ENTIRE dataset's relationships!

---

## 💻 Programs Practiced

| # | Plot Type | ML Use Case |
|---|-----------|-------------|
| 1 | histplot + kde | Feature distribution check |
| 2 | boxplot | Outlier detection |
| 3 | violinplot | Distribution shape comparison |
| 4 | heatmap | Feature correlation |
| 5 | pairplot | Full EDA in one chart |
| 6 | regplot | Trend + correlation |
| 7 | ArthAI seaborn dashboard | Real application |

---

## 🔗 How This Connects to AI/ML

```python
import seaborn as sns

# 1. Correlation heatmap — ALWAYS first step in ML!
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# Shows multicollinearity — remove redundant features!

# 2. Box plot for outlier detection before training
sns.boxplot(data=df, y='salary')
# Visual IQR method — same math as Day 39!

# 3. Pairplot for complete feature relationship EDA
sns.pairplot(df, hue='target_class')
# See class separability — will this be easy to predict?

# 4. Distribution check before model selection
sns.histplot(df['feature'], kde=True)
# Normal distribution → Linear models work well
# Skewed → Tree-based models more robust
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Forgetting to import matplotlib too:**
```python
import seaborn as sns
sns.histplot(df['salary'])
# Nothing shows without:
import matplotlib.pyplot as plt
plt.show()  # ✅ Seaborn still needs this!
```

**Mistake 2 — Wrong data format for heatmap:**
```python
# Wrong — heatmap needs numeric data!
sns.heatmap(df)  # ❌ crashes on string columns!

# Correct — select numeric columns first!
sns.heatmap(df.select_dtypes(
    include='number').corr())  # ✅
```

**Mistake 3 — Too many categories in hue:**
```python
# With 20+ unique categories — unreadable!
sns.scatterplot(data=df, x='a', y='b',
                hue='city_with_50_values')  # ❌ messy!

# Fix — group rare categories or pick top N
top_cities = df['city'].value_counts().head(5).index
df_filtered = df[df['city'].isin(top_cities)]
sns.scatterplot(data=df_filtered, x='a',
                y='b', hue='city')  # ✅ clean!
```

---

## 💎 Important Realizations

1. **Seaborn = statistics + Matplotlib in one line**
   Box plots calculate quartiles automatically!
   Heatmaps calculate correlation automatically!
   No manual math needed — just visualize!

2. **Correlation heatmap is EDA step #1**
   Before training ANY model — check this chart!
   High correlation between features = redundancy!
   Low correlation with target = weak feature!

3. **Box plots are the fastest outlier detector**
   Same IQR math from Day 39 — now visual!
   One glance shows which city has salary outliers!

4. **Pairplot replaces 10 individual charts**
   For small datasets (< 10 features) — pairplot
   gives you the COMPLETE picture instantly!

---

## 🎯 Next Goal

- EDA — Exploratory Data Analysis (complete workflow!)
- Combine everything from Days 36-42 into one process!
- Direct preparation for Indian Job Market Analyzer!

---

*Day 42 complete — Statistical visualization mastered! 📊🔥*


## 🔑 Categorical Plots — The Power Trio

### Box Plot — Shows quartiles and outliers!
```python
sns.boxplot(data=df, x='city', y='salary_lpa')
```
