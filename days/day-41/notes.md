# Day 41 — Matplotlib: Data Visualization 🚀

**Date:** 28 June 2026
**Time Spent:** (4 hours)
**Resource Used:** [Matplotlib Docs](https://matplotlib.org/) | [Kaggle Data Visualization](https://www.kaggle.com/learn/data-visualization)

---

## 📚 Topics Covered

- Why visualization matters in ML
- Matplotlib architecture
- Line charts
- Bar charts
- Histograms
- Scatter plots
- Pie charts
- Subplots
- Styling and customization
- Saving figures

---

## 🔑 Why Visualization Matters
"A picture is worth a thousand numbers!"
Without visualization:

→ You see 1000 rows of salary data
→ Hard to spot patterns

With visualization:

→ One bar chart shows which city pays most
→ One scatter plot shows salary vs experience
→ Instantly understand complex patterns!

**So what? Why does this matter for ML?**
EDA (Exploratory Data Analysis) = visualization!

Before training ANY model you must visualize:
→ Distribution of features (histogram)
→ Correlation between features (scatter)
→ Class balance (bar chart)
→ Outliers (box plot)

"Plot first, model later" — every ML engineer! 🔥
---

## 🔑 Matplotlib Architecture

```python
import matplotlib.pyplot as plt

# Two ways to use Matplotlib:

# 1. Simple pyplot interface (quick plots)
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()

# 2. Object-oriented interface (recommended!)
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

**Key objects:**
Figure → the entire window/image
Axes   → a single plot inside the figure
plt    → shortcut interface to both
---

## 🔑 Line Chart

```python
import matplotlib.pyplot as plt
import numpy as np

# SIP growth over time
years = np.arange(1, 21)
corpus = [future_value_sip(5000, 12, y) for y in years]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years, corpus, color='#10b981',
        linewidth=2.5, marker='o', markersize=5)
ax.set_title('SIP Growth: ₹5000/month at 12%')
ax.set_xlabel('Years')
ax.set_ylabel('Corpus (₹)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sip_growth.png', dpi=150)
plt.show()
```

---

## 🔑 Bar Chart

```python
cities = ['Bangalore', 'Mumbai', 'Delhi',
          'Hyderabad', 'Pune']
salaries = [24.2, 22.8, 20.1, 21.5, 18.9]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(cities, salaries,
              color='#059669', edgecolor='white',
              linewidth=0.5)

# Add value labels on bars
for bar, salary in zip(bars, salaries):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.2,
            f'₹{salary}', ha='center',
            fontweight='bold')

ax.set_title('Average AI/ML Salary by City')
ax.set_ylabel('Average Salary (₹ LPA)')
plt.tight_layout()
plt.show()
```

---

## 🔑 Histogram

```python
# Salary distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['salary_lpa'], bins=20,
        color='#059669', edgecolor='white',
        alpha=0.8)
ax.axvline(df['salary_lpa'].mean(),
           color='red', linestyle='--',
           label=f"Mean: ₹{df['salary_lpa'].mean():.1f}")
ax.legend()
ax.set_title('Salary Distribution')
plt.show()
```

**ML Connection:**
```python
# ALWAYS check feature distributions before ML!
# Skewed distributions → need transformation!
# Bimodal → might have two clusters!
```

---

## 🔑 Scatter Plot

```python
# Experience vs Salary — check linear relationship
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['experience'], df['salary_lpa'],
           alpha=0.6, color='#059669', s=50)
ax.set_xlabel('Experience (years)')
ax.set_ylabel('Salary (₹ LPA)')
ax.set_title('Experience vs Salary')
plt.show()
```

**ML Connection:**
```python
# Scatter shows correlation between features!
# Strong correlation → one feature may be redundant
# No correlation → feature might not help model
# This is visual feature selection!
```

---

## 🔑 Subplots — Multiple Charts

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left
axes[0, 0].bar(cities, salaries)
axes[0, 0].set_title('Salary by City')

# Top-right
axes[0, 1].hist(df['salary_lpa'], bins=20)
axes[0, 1].set_title('Salary Distribution')

# Bottom-left
axes[1, 0].scatter(df['experience'], df['salary_lpa'])
axes[1, 0].set_title('Experience vs Salary')

# Bottom-right
axes[1, 1].pie(role_counts, labels=roles,
               autopct='%1.1f%%')
axes[1, 1].set_title('Jobs by Role')

plt.tight_layout()
plt.savefig('dashboard.png', dpi=150)
plt.show()
```

---

## 💻 Programs Practiced

| # | Chart Type | ML Use Case |
|---|------------|-------------|
| 1 | Line Chart | Training loss curves |
| 2 | Bar Chart | Feature importance |
| 3 | Histogram | Feature distribution |
| 4 | Scatter Plot | Feature correlation |
| 5 | Pie Chart | Class distribution |
| 6 | Subplots | EDA dashboard |
| 7 | ArthAI charts | Financial analysis |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Training loss visualization — every ML project!
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Model Training Progress')

# 2. Feature importance bar chart
importances = model.feature_importances_
plt.barh(feature_names, importances)
plt.title('Feature Importance')

# 3. Confusion matrix heatmap
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')

# 4. ROC curve
plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC Curve')
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Forgetting plt.show():**
```python
plt.plot([1,2,3], [4,5,6])
# Nothing happens without:
plt.show()  # ✅
```

**Mistake 2 — Forgetting tight_layout:**
```python
# Without tight_layout — labels overlap!
plt.tight_layout()  # ✅ always add before show/save
```

**Mistake 3 — Wrong figure size:**
```python
# Default is too small for subplots!
fig, axes = plt.subplots(2, 2)           # ❌ too small
fig, axes = plt.subplots(2, 2,
    figsize=(14, 10))  # ✅ proper size
```

---

## 💎 Important Realizations

1. **Visualization is ML debugging**
   When your model performs badly — plot the data!
   90% of ML bugs are visible in data plots!

2. **Always check distributions before modeling**
   Normal distribution → linear models work well
   Skewed → log transform first
   Bimodal → might need clustering!

3. **Subplots make professional reports**
   ArthAI's PDF report can embed matplotlib charts!
   Job Market Analyzer uses subplots for dashboard!

---

## 🎯 Next Goal

- Seaborn — Statistical Visualization
- Beautiful plots with less code!
- Built on Matplotlib but easier!

---

*Day 41 complete — Visualization unlocked! 📊🔥*
