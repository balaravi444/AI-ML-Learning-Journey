# Day 76 — Model Optimization + Callbacks 🚀

**Date:** 02 August 2026
**Phase:** 5 — Deep Learning
**Time Spent:**(3 hours)
**Resource Used:** [Keras Callbacks](https://keras.io/api/callbacks/)

---

## 📚 Topics Covered

- Learning rate scheduling
- ReduceLROnPlateau deep dive
- Cosine annealing
- Model checkpointing strategy
- EarlyStopping patience tuning
- Mixed precision training
- Training diagnostics
- Full callback stack for production

---

## 🔑 Learning Rate is the Most Important Hyperparameter
Too high LR: loss oscillates, never converges
Too low LR: trains too slowly, gets stuck

Adaptive strategies:
→ Start high (explore loss surface)
→ Decrease over time (fine-tune)
→ This always beats fixed LR!

ReduceLROnPlateau:
Monitor val_loss
If no improvement for patience=5 epochs:
lr = lr × factor (e.g. 0.5)
Stop reducing at min_lr (e.g. 1e-7)

Cosine Annealing:
lr follows cosine curve
High → Low → (optionally restart)
Often finds better minima than fixed schedule!
---

## 🔑 Callbacks Stack for Production

```python
callbacks = [
    # 1. Stop training when stuck
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.001),

    # 2. Save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'),

    # 3. Reduce LR when plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=5,
        min_lr=1e-7,
        verbose=1),

    # 4. Custom training logger
    CSVLogger('training_log.csv')
]
```

---

## 🔑 Mixed Precision Training

```python
# Use float16 for computation, float32 for weights
# 2× faster on modern GPUs!
# Same accuracy as float32!

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Then build model normally
# Keras handles the rest automatically!

# For CPU (no GPU): use float32 (default)
# Only enable mixed_precision if you have GPU!
```

---

## 🔑 Training Diagnostics
Good training signs:
✅ train_loss steadily decreasing
✅ val_loss close to train_loss
✅ val_accuracy improving
✅ LR reducing smoothly

Warning signs:
⚠️ val_loss increasing (overfitting!)
⚠️ train_loss = 0 but val_loss = high (overfit!)
⚠️ loss oscillating (LR too high!)
⚠️ loss not moving (LR too low, dead neurons!)

Fix oscillating loss:
Reduce learning_rate by 10×

Fix no improvement:
Increase learning_rate by 10×
Check data pipeline
Check normalization!
---

## 💎 Important Realizations

1. **ReduceLROnPlateau saves most models**
   When training gets stuck → LR too high
   ReduceLROnPlateau finds this automatically!
   Usually adds 3-7% accuracy!

2. **patience must match your dataset**
   Small dataset: patience=5 (overfits fast)
   Large dataset: patience=15 (slower to plateau)
   Medical imaging: patience=10 (our choice)

3. **restore_best_weights = True always**
   Without it: EarlyStopping returns LAST weights
   Last weights ≠ best weights!
   Always restore_best_weights=True! 🔥

4. **Save model every epoch that improves**
   Never lose progress to crashes!
   ModelCheckpoint(save_best_only=True)
   Free insurance! Always use it!

---

## 🎯 Next Goal (Day 77)

- Skin Disease Detector starts!
- HAM10000 dataset
- MobileNetV2 + transfer learning
- All Days 71-76 skills applied!

---

*Day 76 complete — Optimization mastered! ⚙️🔥*



