# 🤖 Machine Learning Cheatsheet

Quick reference for model building, evaluation, and Scikit-learn syntax.

---

## The Standard ML Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # never fit on test data!
```

## Choosing an Algorithm

| Problem Type | Try First | Then Try |
|---|---|---|
| Binary classification | Logistic Regression | Random Forest, XGBoost |
| Multi-class classification | Logistic Regression (OvR) | Random Forest, XGBoost |
| Regression | Linear Regression | Random Forest, XGBoost |
| Clustering | K-Means | DBSCAN, Hierarchical |
| Dimensionality reduction | PCA | t-SNE, UMAP |
| Text classification | TF-IDF + Logistic Regression | Fine-tuned transformer |

## Model Training (Scikit-learn pattern — same for every model)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)
```

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, mean_squared_error
)

accuracy_score(y_test, predictions)
classification_report(y_test, predictions)
confusion_matrix(y_test, predictions)
roc_auc_score(y_test, probabilities[:, 1])   # binary classification
mean_squared_error(y_test, predictions)       # regression
```

| Metric | When to Use |
|---|---|
| Accuracy | Balanced classes only |
| Precision | Cost of false positives is high (e.g. spam filter) |
| Recall | Cost of false negatives is high (e.g. disease detection) |
| F1 Score | Balance precision & recall, imbalanced classes |
| ROC-AUC | Overall ranking quality of classifier |
| RMSE / MAE | Regression error magnitude |

## Cross-Validation & Tuning

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="f1")
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
```

## Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# or use class_weight
model = RandomForestClassifier(class_weight="balanced")
```

## Feature Engineering Quick Wins

```python
# Encoding categorical variables
pd.get_dummies(df, columns=["category"], drop_first=True)

from sklearn.preprocessing import LabelEncoder
df["col_encoded"] = LabelEncoder().fit_transform(df["col"])

# Handling outliers (IQR method)
Q1, Q3 = df["col"].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df["col"] >= Q1 - 1.5*IQR) & (df["col"] <= Q3 + 1.5*IQR)]
```

## Bias-Variance Cheat Sheet

| Symptom | Problem | Fix |
|---|---|---|
| High train acc, low test acc | Overfitting (high variance) | More data, regularization, simpler model |
| Low train acc, low test acc | Underfitting (high bias) | More features, complex model, less regularization |
| Train ≈ test, both low | Underfitting | Try a different algorithm |
| Train ≈ test, both high | Good fit ✅ | Ship it |

## Explainability

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```
