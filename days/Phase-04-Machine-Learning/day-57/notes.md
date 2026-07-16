# Day 57 — Model Evaluation & Metrics 🚀

**Date:** 14 July 2026
**Time Spent:** (4 hours)
**Resource Used:** [Scikit-learn Docs](https://scikit-learn.org/) | [StatQuest](https://www.youtube.com/@statquest)

---

## 📚 Topics Covered

- Why accuracy alone is misleading
- Confusion matrix deep dive
- Precision, Recall, F1 Score
- ROC Curve and AUC
- Precision-Recall Curve
- Regression metrics — R², MAE, RMSE, MAPE
- Multi-class metrics
- Choosing the right metric for your problem
- Real application — student predictor evaluation

---

## 🔑 Why Accuracy Alone is Misleading
Cancer detection dataset:
→ 99% healthy patients
→ 1% cancer patients
Dumb model: "predict healthy always"
→ Accuracy = 99% 🎉
But this model KILLS patients!
It catches 0% of cancer cases!
Accuracy = 99% → completely useless model ❌
F1 Score  = 0%  → reveals the truth ✅
**Rule:** Always check accuracy AND other metrics!
Especially on imbalanced datasets!

---

## 🔑 Confusion Matrix
**Rule:** Always check accuracy AND other metrics!
Especially on imbalanced datasets!

---

## 🔑 Confusion Matrix
Predicted
             FAIL    PASS
             
Actual  FAIL  [  TN   |  FP  ]
PASS  [  FN   |  TP  ]
TN = True Negative  → correctly predicted FAIL
TP = True Positive  → correctly predicted PASS
FP = False Positive → predicted PASS, actually FAIL (Type I error)
FN = False Negative → predicted FAIL, actually PASS (Type II error)
Everything comes from these 4 numbers!

---

## 🔑 Precision, Recall, F1
Precision = TP / (TP + FP)
"Of everyone I predicted PASS, how many actually passed?"
→ High precision = few false alarms
Recall = TP / (TP + FN)
"Of everyone who actually PASSED, how many did I catch?"
→ High recall = few misses
F1 = 2 * (Precision * Recall) / (Precision + Recall)
= Harmonic mean of precision and recall
→ Balance between the two!
**When to prioritize which:**
Medical diagnosis   → HIGH RECALL
(never miss a sick patient!)
Spam detection      → HIGH PRECISION
(never block real emails!)
Fraud detection     → HIGH RECALL
(never miss fraud!)
Search ranking      → HIGH PRECISION
(only show relevant results!)
Student intervention → HIGH RECALL
(catch all at-risk students!)

---

## 🔑 ROC Curve and AUC
ROC = Receiver Operating Characteristic
Plots: True Positive Rate vs False Positive Rate
at different decision thresholds
AUC = Area Under the ROC Curve
AUC = 1.0 → perfect model
AUC = 0.5 → random guessing
AUC = 0.0 → perfectly wrong (just flip predictions!)
Key insight:
AUC measures how well the model RANKS predictions
regardless of the threshold

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
```

---

## 🔑 Precision-Recall Curve
Better than ROC curve for IMBALANCED datasets!
Why?
ROC curve can look great even when
the model performs poorly on the minority class.
PR curve reveals the truth for imbalanced data!
Average Precision (AP) = area under PR curve
---

## 🔑 Regression Metrics
MAE  = mean(|y_true - y_pred|)
→ Average absolute error
→ Same units as target (₹ LPA)
→ Robust to outliers
RMSE = √mean((y_true - y_pred)²)
→ Penalizes large errors more!
→ Same units as target
→ Use when large errors are costly
R²   = 1 - SS_res/SS_tot
→ Fraction of variance explained
→ 1.0 = perfect, 0.0 = mean baseline
→ Can be negative (worse than mean!)
MAPE = mean(|y_true - y_pred| / |y_true|) × 100
→ Percentage error
→ Easy to explain to business!
→ "Off by 5% on average"

---

## 🔑 Multi-class Metrics

```python
# Macro: average metric across all classes (unweighted)
# Micro: pool predictions across all classes
# Weighted: average weighted by class support

from sklearn.metrics import f1_score

f1_macro    = f1_score(y_test, y_pred, average='macro')
f1_micro    = f1_score(y_test, y_pred, average='micro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# For imbalanced multi-class → use weighted!
# For equal class importance → use macro!
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Confusion matrix deep dive | TP, TN, FP, FN |
| 2 | Precision-Recall tradeoff | Threshold tuning |
| 3 | ROC + AUC | Model ranking quality |
| 4 | PR curve | Imbalanced data metric |
| 5 | Regression metrics | MAE, RMSE, R², MAPE |
| 6 | Multi-class evaluation | Macro vs weighted |
| 7 | Complete evaluation report | All metrics together |

---

## 🔗 How This Connects to AI/ML

```python
# 1. Choosing metric = choosing what to optimize!
# Student predictor → optimize Recall (catch failures)
# Spam detector → optimize Precision (no false blocks)
# ArthAI salary → optimize MAE (₹ error)

# 2. AUC used in Kaggle competitions!
# Many competitions use AUC as the scoring metric
# Because it's threshold-independent!

# 3. Business metrics matter more than ML metrics
# ML metric: F1 = 0.91
# Business metric: "We catch 94% of at-risk students
#                  with only 8% false interventions"
# Always translate to business language! 🔥
```

---

## ❌ Mistakes & Fixes

**Mistake 1 — Reporting only accuracy:**
```python
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# ❌ Not enough! Especially for imbalanced data!

# Correct — report everything!
print(classification_report(y_test, y_pred))  # ✅
```

**Mistake 2 — Using F1 macro on imbalanced data:**
```python
# Wrong — gives equal weight to rare class!
f1 = f1_score(y_test, y_pred, average='macro')  # ❌

# Correct — use weighted!
f1 = f1_score(y_test, y_pred, average='weighted')  # ✅
```

**Mistake 3 — Not using predict_proba for ROC:**
```python
# Wrong — uses hard predictions!
fpr, tpr, _ = roc_curve(y_test,
                          model.predict(X_test))  # ❌

# Correct — use probabilities!
fpr, tpr, _ = roc_curve(
    y_test,
    model.predict_proba(X_test)[:, 1])  # ✅
```

---

## 💎 Important Realizations

1. **Every metric answers a different question**
   Accuracy: "How often am I right overall?"
   Precision: "When I say yes, am I right?"
   Recall: "Do I catch all the yes cases?"
   AUC: "Can I rank predictions correctly?"
   Choose based on your BUSINESS problem!

2. **AUC is threshold-independent**
   You don't need to choose a threshold to evaluate!
   Compare models using AUC — then tune threshold!

3. **The confusion matrix is the source of truth**
   Every metric is derived from 4 numbers: TN/TP/FP/FN
   If something seems wrong — go back to the matrix!

4. **PR curve > ROC for imbalanced data**
   ROC can look great on imbalanced data
   while the model actually fails on minority class!
   Always use PR curve when classes are imbalanced!

---

## 🎯 Next Goal

- Cross Validation + Hyperparameter Tuning
- GridSearchCV, RandomizedSearchCV
- Finding the best model systematically!

---

*Day 57 complete — Evaluation mastered! 📊🔥*



