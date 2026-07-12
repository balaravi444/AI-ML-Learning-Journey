# Day 43 — EDA: Exploratory Data Analysis 🚀

**Date:** 30 June 2026
**Time Spent:** (4 hours)
**Resource Used:** [Kaggle Learn](https://www.kaggle.com/learn) | [Pandas Docs](https://pandas.pydata.org/docs/)

---

## 📚 Topics Covered

- What is EDA and why it matters
- The complete EDA workflow
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Data quality assessment
- Feature relationships
- Business insights from data
- EDA report generation

---

## 🔑 What is EDA?
EDA = Understanding your data BEFORE modeling!
The process:

Load data
Understand structure
Check data quality
Analyze distributions
Find relationships
Extract insights
Decide features for ML

**So what? Why does this matter?**
Without EDA:
→ Train model on dirty data
→ Include useless features
→ Miss important patterns
→ Model fails in production!
With EDA:
→ Know your data deeply
→ Choose right features
→ Spot problems early
→ Model works well!
EDA is NOT optional — it's STEP 1 of every ML project!

---

## 🔑 The Complete EDA Workflow

Step 1: Load & First Look
df.head(), df.info(), df.describe()
df.shape, df.dtypes, df.columns

Step 2: Data Quality Check
df.isnull().sum()     ← missing values
df.duplicated().sum() ← duplicates
df['col'].unique()    ← unique values
df['col'].nunique()   ← cardinality

Step 3: Univariate Analysis
Each feature individually:
Numerical → histplot, boxplot, describe()
Categorical → value_counts(), barplot

Step 4: Bivariate Analysis
Two features together:
Num vs Num → scatter, correlation
Num vs Cat → boxplot, violinplot
Cat vs Cat → crosstab, heatmap

Step 5: Multivariate Analysis
Multiple features together:
Correlation heatmap, pairplot

Step 6: Insights & Conclusions
Key findings → what drives the target?
Feature selection decisions
Data quality issues to fix

---

## 🔑 Univariate Analysis

```python
# Numerical feature
print(df['salary_lpa'].describe())
print(f"Skewness: {df['salary_lpa'].skew():.2f}")
print(f"Kurtosis: {df['salary_lpa'].kurtosis():.2f}")

# Distribution check
# Skewness > 1 → right skewed → log transform!
# Skewness < -1 → left skewed
# -1 to 1 → approximately normal

# Categorical feature
print(df['city'].value_counts())
print(df['city'].value_counts(normalize=True))  # percentages
```

---

## 🔑 Bivariate Analysis

```python
# Correlation
correlation = df['experience'].corr(df['salary_lpa'])
print(f"Correlation: {correlation:.2f}")

# Cross-tabulation for categories
pd.crosstab(df['city'], df['remote'])

# Group statistics
df.groupby('city')['salary_lpa'].agg(
    ['mean', 'median', 'std'])
```

---

## 🔑 Key EDA Questions for Job Market Data
What is the salary distribution? Normal or skewed?
Which city has highest salaries?
How much does experience affect salary?
Which skills are most demanded?
Is there a salary gap between remote and onsite?
Which companies hire most AI/ML talent?
What is the typical experience required?
Are there salary outliers? Who earns unusually high/low?

**These 8 questions = complete EDA for job market!**
**Same pattern used in the Job Market Analyzer project!**

---

## 💻 Programs Practiced

| # | Analysis Type | What It Reveals |
|---|---------------|-----------------|
| 1 | Data Quality Check | Missing, duplicates, types |
| 2 | Univariate Numerical | Salary distribution, skewness |
| 3 | Univariate Categorical | City/role distribution |
| 4 | Bivariate Num-Num | Experience vs salary |
| 5 | Bivariate Num-Cat | Salary by city/role |
| 6 | Multivariate | Full correlation analysis |
| 7 | Complete EDA Report | Job Market Analyzer preview! |

---

## 🔗 How This Connects to AI/ML

```python
# EDA directly informs ML decisions!

# 1. High skewness → log transform features
if df['salary'].skew() > 1:
    df['salary_log'] = np.log1p(df['salary'])

# 2. High correlation between features → remove one
# (avoids multicollinearity in linear models!)
correlation_matrix = df.corr()
high_corr = (correlation_matrix.abs() > 0.9)

# 3. Class imbalance → use oversampling!
class_counts = df['target'].value_counts()
imbalance_ratio = class_counts.min() / class_counts.max()
if imbalance_ratio < 0.1:
    # Use SMOTE or class weights!
    pass

# 4. Outliers → remove or cap before ML
Q1, Q3 = df['salary'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[df['salary'].between(
    Q1 - 1.5*IQR, Q3 + 1.5*IQR)]
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Skipping EDA before modeling:**
```python
# Wrong — direct to model without EDA!
model.fit(df[features], df['target'])  # ❌

# Correct — EDA first, THEN model!
# Step 1: EDA
# Step 2: Clean
# Step 3: Feature engineering
# Step 4: model.fit() ✅
```

**Mistake 2 — Ignoring skewness:**
```python
# Skewed salary → linear regression struggles!
df['salary'].skew()  # returns 2.3 → very skewed!

# Fix → log transform
df['salary_log'] = np.log1p(df['salary'])
# Now skewness ≈ 0 → linear models work better!
```

---

## 💎 Important Realizations

1. **EDA is the most underrated ML skill**
   Junior engineers rush to models.
   Senior engineers spend hours on EDA.
   Better EDA = better models, always!

2. **Every EDA question maps to a Pandas/Seaborn command**
   "Which city pays most?" → groupby().mean()
   "Is salary normal?" → histplot + skew()
   "Do features correlate?" → corr() + heatmap()

3. **EDA findings directly drive feature engineering**
   Skewed → log transform
   Outliers → cap or remove
   Correlated features → drop one
   These decisions happen in EDA!

4. **The Indian Job Market Analyzer IS just EDA**
   Every insight in the analyzer = one EDA step!
   Day 43 is the foundation of Day 47 project!

---

## 🎯 Next Goal

- Feature Engineering
- Transform raw features into ML-ready signals!
- Direct input to Job Market Analyzer model!

---

*Day 43 complete — EDA mastered! 🔍🔥*




"EDA is detective work on data" —
Every senior data scientist
