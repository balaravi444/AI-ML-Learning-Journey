# Day 78 — Skin Disease Detector 🚀
# Project Day 2 of 4 — Fine-tuning + Evaluation

**Date:** 04 August 2026
**Phase:** 5 — Deep Learning
**Time Spent:** (add your hours)
**Project:** Skin Disease Detector — Day 2

---

## 📚 What I Built Today

- Phase 2: Unfreeze top MobileNetV2 layers
- Fine-tuning with very low learning rate
- Multi-class confusion matrix
- Per-class precision, recall, F1
- Clinical metrics — melanoma recall
- ROC curves per disease class
- Model comparison Phase 1 vs Phase 2

---

## 🔑 Phase 2 — Fine-tuning Strategy
Phase 1 result: val_acc ≈ 0.76
Goal: val_acc ≈ 0.88-0.92

Strategy:

Unfreeze top 54 layers of MobileNetV2
(total 154 layers → freeze bottom 100)
Compile with very low LR (1e-5)
100× smaller than Phase 1!
Train again with same callbacks
EarlyStopping(patience=12) to protect

Why 100× smaller LR?
→ High LR = destroys pretrained weights
→ "Catastrophic forgetting"
→ Model unlearns ImageNet features
→ Accuracy DROPS below Phase 1!
Low LR = gentle adaptation
→ Slowly shifts features toward skin domain
→ Keeps general knowledge intact
→ Accuracy IMPROVES! 🔥
---

## 🔑 Which Layers to Unfreeze?
MobileNetV2 layer structure:
Layers 0-99: Early features (edges, textures)
→ Keep frozen! Universal!
Layers 100-153: High-level features
→ Unfreeze! Adapt to skin!

Why this split?
Early layers learn: lines, colors, gradients
These are universal — same for skin as for dogs!
No need to retrain these.

Late layers learn: complex patterns
These are task-specific.
Skin disease patterns differ from ImageNet cats!
These benefit from fine-tuning!
---

## 🔑 Multi-class Evaluation
7 classes → 7×7 confusion matrix!

Diagonal = correct predictions
Off-diagonal = misclassifications

Most important metric: MELANOMA RECALL
Missing melanoma = false sense of security = death

Target metrics:
Overall accuracy: > 85%
Melanoma recall: > 85% (non-negotiable!)
Melanoma precision: > 70% (acceptable FP rate)
Melanoma F1: > 77%
Overall AUC: > 0.94

Business metric:
"For every 100 melanoma cases,
how many does the model catch?"
Answer must be: 85+ of 100.
---

## 💎 Important Realizations

1. **Fine-tuning is surgical, not sledgehammer**
   Don't unfreeze everything.
   Don't use high LR.
   Gentle, targeted, slow = best results!

2. **Melanoma recall is the deployment gate**
   If melanoma recall < 80% → do NOT deploy!
   Better to have false alarms than miss cancer.
   Recall > precision for life-critical applications!

3. **Confusion matrix reveals failure modes**
   Which classes get confused with which?
   Melanoma vs Nevi confusion = most dangerous!
   Helps decide where to collect more data!

4. **Per-class AUC tells the full story**
   Overall AUC hides class-level performance.
   Melanoma AUC > 0.95 = model ready!
   Melanoma AUC < 0.90 = need more data!

---

## 🎯 Next Goal (Day 79)

- FastAPI backend
- Prediction endpoint with clinical output
- Confidence scores + recommendations
- "See a doctor immediately" alerts

---

*Day 78 complete — Fine-tuning + Evaluation done! 📊🔥*
