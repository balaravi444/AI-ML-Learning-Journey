# Day 60 — Bug Priority Predictor: ML Pipeline 🚀
# Project Day 2 of 4 — Training + Evaluation

**Date:** 17 July 2026
**Time Spent:** (2 hours)
**Project:** Software Bug Priority Predictor — Day 2

---

## 📚 What I Built Today

- Full ML training pipeline
- TF-IDF + Random Forest classifier
- SMOTE for class imbalance
- Multi-class evaluation
- Model serialization with joblib

---

## 🔑 Why SMOTE?
Our dataset without balancing:
Critical → 150 samples (8.8%)
High → 400 samples (23.5%)
Medium → 700 samples (41.2%)
Low → 450 samples (26.5%)

Without SMOTE:
→ Model sees 700 Medium examples
→ Learns to predict Medium always
→ Gets 41% accuracy
→ Misses ALL Critical bugs → engineers hate it!

SMOTE = Synthetic Minority Oversampling Technique
→ Creates synthetic Critical + High examples
→ All classes balanced → model learns everything!

---

## 🔑 Multi-class Evaluation
Not binary anymore — 4 classes!
Critical, High, Medium, Low

Metrics per class:
→ Precision: when I say Critical, am I right?
→ Recall: do I catch all Critical bugs?
→ F1: balance of both

Overall:
→ Weighted F1: accounts for class size
→ Macro F1: equal weight to all classes

For us → Critical recall is MOST IMPORTANT!
Missing a Critical bug = production outage!

---

## 💎 Important Realizations

1. **TF-IDF + Random Forest = powerful NLP baseline**
   Before transformers existed, this combo won
   most text classification competitions!
   Still used in production today!

2. **SMOTE works on feature space not raw text**
   Can't SMOTE raw text — must convert to
   numbers first (TF-IDF), THEN SMOTE!

3. **Confusion matrix for 4 classes**
   4×4 matrix instead of 2×2
   Diagonal = correct predictions
   Off-diagonal = misclassifications
   Most errors should be adjacent classes
   (Critical→High, not Critical→Low)

4. **joblib saves the full pipeline**
   vectorizer + scaler + model saved together
   Load once → predict instantly!

---

## 🎯 Next Goal (Day 61)

- FastAPI backend — prediction endpoints
- Beautiful dashboard UI
- Real-time priority prediction demo

---

*Day 60 complete — Model trained! 🤖🔥*

