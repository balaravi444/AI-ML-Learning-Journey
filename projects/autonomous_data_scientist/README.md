# 🤖 Autonomous Data Scientist

> Upload any CSV → Auto profile → AutoML → SHAP explanations → Live prediction API. Built in 4 days during Days 67–70 of my 90-day AI/ML journey.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![AutoML](https://img.shields.io/badge/AutoML-6%20Models-orange)
![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

🔗 **Live Demo:** [Add Render URL here]
📁 **Part of:** [AI/ML Learning Journey](https://github.com/balaravi444/AI-ML-Learning-Journey)

---

## 🎯 Problem

Companies have datasets. They want ML predictions.
They don't have a data scientist.
Hiring one takes 3 months and ₹30L+.

## 💡 Solution

Upload CSV → get live prediction API in minutes.
No ML knowledge required.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Auto Profiling** | Instant data quality report |
| ⚙️ **Auto Preprocessing** | Impute, scale, encode automatically |
| 🤖 **AutoML** | Try 6 models, tune top 3, pick best |
| 🔍 **SHAP Explainability** | Why did the model predict X? |
| 📡 **REST API** | FastAPI with auto docs |
| 📊 **Dashboard** | Chart.js visualization |
| 🚀 **Deploy** | One-click Render deployment |

---

## 🏗️ Architecture

```
CSV Upload
    ↓
Auto Data Profiler → quality score, issues, suggestions
    ↓
Auto Preprocessor → impute + scale + encode → sklearn Pipeline
    ↓
AutoML Engine → 6 models → tune top 3 → best model
    ↓
SHAP Explainer → global importance + local explanations
    ↓
FastAPI → REST endpoints
    ↓
Chart.js Dashboard → prediction + SHAP visualization
```

---

## 🤖 Models Tried Automatically

**Classification:** Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, SVM, KNN

**Regression:** Linear Regression, Ridge, Random Forest, Gradient Boosting, Extra Trees, SVR

---

## 🚀 Run Locally

```bash
git clone https://github.com/balaravi444/AI-ML-Learning-Journey
cd projects/autonomous_data_scientist
pip install -r requirements.txt
python -c "from data.generate_dataset import save_all_datasets; save_all_datasets()"
uvicorn app:app --reload --port 8003
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| POST | `/api/train` | Upload CSV + train |
| GET | `/api/status` | Training status |
| POST | `/api/predict` | Predict + SHAP |
| GET | `/api/explain/global` | Feature importance |
| GET | `/api/report` | AutoML report |
| GET | `/api/stats` | Model stats |

---

*Built Days 67–70 of 90-day AI/ML Learning Journey 🚀*
