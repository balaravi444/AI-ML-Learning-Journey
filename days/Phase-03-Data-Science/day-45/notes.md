# Day 45 — Statistics for ML 🚀

**Date:** 03 July 2026
**Time Spent:** (4 hours)
**Resource Used:** [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability) | [StatQuest YouTube](https://www.youtube.com/@statquest)

---

## 📚 Topics Covered

- Descriptive statistics
- Probability distributions
- Central Limit Theorem
- Hypothesis testing
- Confidence intervals
- Correlation vs causation
- Bayes theorem
- Statistical significance in ML

---

## 🔑 Why Statistics for ML?
ML is applied statistics.Every algorithm has statistics underneath:Linear Regression → Ordinary Least Squares
Naive Bayes       → Bayes Theorem
Decision Trees    → Information Entropy
Neural Networks   → Gradient Descent (calculus+stats)
A/B Testing       → Hypothesis Testing"Machine learning is statistics on steroids"
— Every statistician who learned ML

---

## 🔑 Descriptive Statistics

```python
import numpy as np
import pandas as pd

data = [23, 45, 67, 12, 89, 34, 56, 78, 90, 11]

# Central tendency
mean   = np.mean(data)    # 50.5 — affected by outliers
median = np.median(data)  # 51.0 — robust to outliers
mode   = pd.Series(data).mode()[0]

# Spread
std  = np.std(data)    # standard deviation
var  = np.var(data)    # variance = std²
iqr  = np.percentile(data, 75) - np.percentile(data, 25)

# Shape
skewness = pd.Series(data).skew()    # symmetry
kurtosis = pd.Series(data).kurtosis() # tail heaviness
```

**ML Connection:**
Mean vs Median → which to use for missing values?
→ Outliers present? → Use MEDIAN
→ No outliers? → Use MEANStd deviation → feature scaling uses this!
StandardScaler = (x - mean) / std
This IS statistics in ML!
---

## 🔑 Probability Distributions

### Normal (Gaussian) Distribution
```python
from scipy import stats
import numpy as np

# Generate normal distribution
data = np.random.normal(loc=25, scale=5, size=1000)
# loc = mean, scale = std

# Check if data is normal
stat, p_value = stats.normaltest(data)
print(f"p-value: {p_value:.4f}")
# p > 0.05 → data is likely normal!
```

**ML Connection:**
Why normal distribution matters:
→ LinearRegression assumes normally distributed errors
→ Many features in real datasets are approximately normal
→ Central Limit Theorem → sample means are normal!

### Other Important Distributions
Bernoulli  → binary outcomes (pass/fail)
Binomial   → n binary trials (5 out of 10)
Poisson    → count events (clicks per hour)
Uniform    → equal probability (random split)
---

## 🔑 Central Limit Theorem (CLT)

The most important theorem in statistics for ML!

No matter the original distribution,
the distribution of SAMPLE MEANS
approaches normal as sample size grows!
n ≥ 30 → sample mean is approximately normal!
**ML Connection:**
```python
# Why batch gradient descent works!
# Each batch mean ≈ true mean (CLT!)
# Stochastic gradient descent uses CLT!

# A/B testing relies on CLT
# "Is model A better than model B?"
# Uses sample means which are normal (CLT!)
```

---

## 🔑 Hypothesis Testing

```python
from scipy import stats

# T-test — compare two group means
# "Does remote work pay more than onsite?"
remote_salary = [24, 28, 22, 30, 26]
onsite_salary = [20, 18, 22, 19, 21]

t_stat, p_value = stats.ttest_ind(
    remote_salary, onsite_salary)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ Significant difference! "
          "Remote pays more!")
else:
    print("❌ No significant difference")
```

**Decision rule:**
p-value < 0.05 → Reject null hypothesis
(difference IS significant!)
p-value ≥ 0.05 → Fail to reject null
(difference is NOT significant)
**ML Connection:**
```python
# A/B Testing for ML models!
# "Is Model B significantly better than Model A?"

model_a_scores = [0.85, 0.82, 0.87, 0.83, 0.86]
model_b_scores = [0.89, 0.91, 0.88, 0.90, 0.92]

t, p = stats.ttest_ind(model_b_scores,
                        model_a_scores)
if p < 0.05:
    print("Model B is significantly better! 🚀")
```

---

## 🔑 Correlation vs Causation
Correlation ≠ Causation!
Classic examples:
→ Ice cream sales correlate with drowning rates
(Both increase in summer — not causal!)
→ Salary correlates with experience
(Experience likely CAUSES higher salary)
→ Features correlate with target
(May or may not be causal!)
In ML — we use correlation regardless!
"Correlation is enough for prediction,
causation is needed for intervention"

---

## 🔑 Bayes Theorem
P(A|B) = P(B|A) × P(A) / P(B)
"Update your belief when you get new evidence"
Example:
P(Spam | "Free money") = ?
P("Free money" | Spam) × P(Spam) / P("Free money")
= 0.8 × 0.3 / 0.1
= 2.4 → normalize → high probability of spam!

**ML Connection:**
```python
# Naive Bayes classifier IS Bayes Theorem!
# Used in:
# → Spam detection (email filtering)
# → Text classification (sentiment analysis)
# → Medical diagnosis
# → Document categorization

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()  # Bayes theorem inside!
```

---

## 🔑 Confidence Intervals

```python
import numpy as np
from scipy import stats

data = [24.2, 22.8, 20.1, 21.5, 18.9,
        23.1, 25.4, 19.8, 22.3, 24.7]
n = len(data)
mean = np.mean(data)
se = stats.sem(data)  # standard error

# 95% confidence interval
ci = stats.t.interval(0.95, df=n-1,
                       loc=mean, scale=se)
print(f"Mean: {mean:.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
```

**ML Connection:**
Model accuracy: 87% ± 2% (95% CI)
→ True accuracy is between 85% and 89%!
Always report confidence intervals
with model metrics — not just point estimates!
---

## 💻 Programs Practiced

| # | Topic | ML Application |
|---|-------|----------------|
| 1 | Descriptive stats | Feature analysis |
| 2 | Normal distribution | Feature check |
| 3 | Hypothesis testing | A/B model testing |
| 4 | Correlation analysis | Feature selection |
| 5 | Bayes theorem | Naive Bayes classifier |
| 6 | Confidence intervals | Model evaluation |
| 7 | CLT demonstration | Batch gradient descent |

---

## 🔗 How This Connects to AI/ML

```python
# Statistics powers EVERY ML algorithm!

# 1. Mean Squared Error → statistics!
mse = np.mean((y_true - y_pred) ** 2)

# 2. Standard Deviation → StandardScaler!
X_scaled = (X - X.mean()) / X.std()

# 3. Correlation → feature selection!
corr = np.corrcoef(X.T, y)

# 4. P-value → model comparison (A/B test)!
t, p = stats.ttest_ind(model_a_scores,
                        model_b_scores)

# 5. Bayes → Naive Bayes classifier!
# P(class|features) ∝ P(features|class) × P(class)

# 6. Normal dist → residual check!
residuals = y_true - y_pred
stat, p = stats.normaltest(residuals)
# p > 0.05 → residuals are normal → model OK!
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Confusing correlation with causation:**
Mistake: "Salary correlates with skills_count
→ more skills CAUSES higher salary!"
Reality: Both might be caused by experience!
Experienced people have more skills
AND higher salary.
Fix: Feature engineering — don't blindly
include all correlated features!

**Mistake 2 — Ignoring p-value threshold:**
```python
# Wrong — any correlation = significant!
if correlation > 0:
    print("significant!")  # ❌

# Correct — check p-value!
corr, p_value = stats.pearsonr(x, y)
if p_value < 0.05:
    print("statistically significant!")  # ✅
```

---

## 💎 Important Realizations

1. **Statistics IS machine learning**
   Every ML formula has a statistical derivation!
   Understanding statistics = understanding ML deeply!

2. **p-value < 0.05 is the golden rule**
   Used everywhere: feature selection, A/B testing,
   model comparison, hypothesis testing!

3. **CLT explains why ML works on large datasets**
   With enough data, distributions become normal,
   gradient descent converges, models generalize!

4. **Bayes theorem is the foundation of probabilistic ML**
   Naive Bayes, Bayesian Neural Networks,
   Gaussian Processes — all Bayes theorem!

---

## 🎯 Next Goal

- Data Preprocessing Pipeline
- Combine cleaning + features + stats into one!
- Foundation for Indian Job Market Analyzer!

---

*Day 45 complete — Statistics for ML mastered! 📊🔥*


