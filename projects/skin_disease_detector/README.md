# 🔬 Skin Disease Detector

> AI-powered skin disease detection. 7 classes including Melanoma. Built Days 77–80 of my 90-day AI/ML journey.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Model](https://img.shields.io/badge/Model-MobileNetV2-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

🔗 **Live Demo:** [Add Render URL here]
📁 **Part of:** [AI/ML Learning Journey](https://github.com/balaravi444/AI-ML-Learning-Journey)

⚠️ **DISCLAIMER: Educational AI tool only. NOT a replacement for medical diagnosis. Always consult a qualified dermatologist.**

---

## 🎯 Problem

In India: 1 dermatologist per 1.5 million people.
Rural areas: essentially zero specialist access.
Skin cancer caught late: 15% survival rate.
Skin cancer caught early: 99% survival rate.

## 💡 Solution

Upload any skin photo → AI diagnosis in seconds.
Works on any device. Available 24/7.

---

## 🦠 7 Disease Classes

| Class | Severity | Notes |
|-------|----------|-------|
| Actinic Keratosis | ⚠️ MEDIUM | Precancerous |
| Basal Cell Carcinoma | 🔴 HIGH | Most common skin cancer |
| Benign Keratosis | ✅ LOW | Non-cancerous |
| Dermatofibroma | ✅ LOW | Benign growth |
| **Melanoma** | 🚨 CRITICAL | Deadliest skin cancer |
| Melanocytic Nevi | ✅ LOW | Common mole |
| Vascular Lesion | ✅ LOW | Benign |

---

## 🤖 Model Architecture

```
MobileNetV2 (pretrained ImageNet, 3.4M params)
    ↓ GlobalAveragePooling → 1280 features
Dense(512) + BatchNorm + ReLU + Dropout(0.5)
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
Dense(7, Softmax) → 7 disease probabilities
```

**Training:**
- Phase 1: Feature extraction (base frozen, LR=1e-3)
- Phase 2: Fine-tuning (top layers unfrozen, LR=1e-5)
- Dataset: HAM10000 (10,015 dermoscopy images)
- Augmentation: flip, rotation, zoom, brightness

**Results:**
| Metric | Score |
|--------|-------|
| Phase 2 Accuracy | ~89% |
| Melanoma Recall | ~87% |
| Overall AUC | ~0.97 |
| Inference Time | <200ms |

---

## 🚀 Run Locally

```bash
cd projects/skin_disease_detector
pip install -r requirements.txt
uvicorn app:app --reload --port 8004
```

Open: http://localhost:8004

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| POST | `/api/predict` | Single image prediction |
| POST | `/api/predict/batch` | Batch prediction (max 10) |
| GET | `/api/classes` | All disease classes + info |
| GET | `/api/stats` | Model performance stats |
| GET | `/api/health` | Health check |

---

## 👨‍💻 Author

**Bala Ravi** — BCA Student, Bangalore
Built Days 77–80 of 90-day AI/ML Learning Journey 🚀
