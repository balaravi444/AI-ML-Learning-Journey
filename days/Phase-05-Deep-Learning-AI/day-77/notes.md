# Day 77 — Skin Disease Detector 🚀
# Project Day 1 of 4 — Dataset + Model Architecture

**Date:** 03 August 2026
**Phase:** 5 — Deep Learning
**Time Spent:** (3 hours)
**Project:** Skin Disease Detector — Day 1

---

## 📚 What I Built Today

- HAM10000 dataset analysis + preparation
- Synthetic dataset generator (for local testing)
- MobileNetV2 transfer learning architecture
- Complete training pipeline with callbacks
- Phase 1 training completed
- Model evaluation framework

---

## 🔑 The Problem We're Solving
Skin cancer is the most common cancer worldwide.
Early detection = 99% survival rate.
Late detection = 15% survival rate.

In India:
→ 1 dermatologist per 1.5 million people
→ Most rural areas have ZERO specialists
→ Patients wait months for appointments

Our AI:
→ Upload photo → diagnosis in seconds
→ Works on any smartphone
→ Available 24/7 anywhere in India

7 disease classes:
0. Actinic Keratosis → precancerous
Basal Cell Carcinoma → cancer (common)
Benign Keratosis → benign
Dermatofibroma → benign
Melanoma → 🚨 deadly cancer
Melanocytic Nevi → common mole
Vascular Lesion → benign
---

## 🔑 HAM10000 Dataset
Human Against Machine with 10000 training images

Total: 10,015 dermoscopy images
Source: International Skin Imaging Collaboration
Format: JPEG, 450×600 pixels

Class distribution:
Melanocytic Nevi: 6705 (66.9%) ← imbalanced!
Melanoma: 1113 (11.1%)
Benign Keratosis: 1099 (11.0%)
Basal Cell Carcinoma: 514 (5.1%)
Actinic Keratosis: 327 (3.3%)
Vascular Lesion: 142 (1.4%)
Dermatofibroma: 115 (1.1%)
Challenge: Class imbalance!
Strategy: class_weight + augmentation
---

## 🔑 Model Architecture
Input (224×224×3)
↓
MobileNetV2 (pretrained ImageNet, frozen)
→ 154 layers
→ 3.4M parameters
→ Extracts 1280 features per image
↓
GlobalAveragePooling2D
→ 1280-dim vector
↓
Dense(512, ReLU) + BatchNorm + Dropout(0.5)
↓
Dense(256, ReLU) + Dropout(0.3)
↓
Dense(7, Softmax)
→ 7 disease probabilities

Total params: 4,104,775
Trainable: 662,279 (Phase 1)
Frozen: 3,442,496
---

## 💎 Important Realizations

1. **This could actually save lives**
   Not a toy project.
   Deploying this in rural India = real impact.
   AI for healthcare = highest purpose of ML!

2. **Class imbalance is the main challenge**
   66% of data is one class (Nevi).
   Without correction → misses all melanoma.
   class_weight + SMOTE + augmentation = solution!

3. **MobileNetV2 is perfect for mobile deployment**
   3.4M params → runs on smartphone CPU!
   Accuracy: 88-92% on HAM10000.
   Fast inference: <100ms on phone!

4. **Recall matters more than accuracy here**
   Missing a melanoma = death.
   False alarm = extra checkup (annoying but ok).
   Optimize for MELANOMA RECALL, not accuracy!

---

## 🎯 Next Goal (Day 78)

- Phase 2: Fine-tune MobileNetV2 top layers
- Class imbalance deep dive
- Multi-class evaluation (per-class metrics)
- Confusion matrix analysis

---

*Day 77 complete — Architecture built + Phase 1 done! 🏥🔥*
