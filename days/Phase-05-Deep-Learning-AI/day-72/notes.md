# Day 72 — TensorFlow & Keras 🚀

**Date:** 29 July 2026
**Phase:** 5 — Deep Learning
**Time Spent:** (3 hours)
**Resource Used:** [TensorFlow Docs](https://tensorflow.org) | [Keras Docs](https://keras.io)

---

## 📚 Topics Covered

- What is TensorFlow and Keras
- Sequential API — simple models
- Functional API — complex models
- model.compile, model.fit, model.evaluate
- Callbacks — EarlyStopping, ModelCheckpoint
- Batch training and epochs
- Saving and loading models
- TF vs our scratch NN (Day 71)

---

## 🔑 TensorFlow vs Keras
TensorFlow:
→ Google's ML framework
→ Low-level operations (tensors, graphs)
→ Deployed in production at scale
→ Powers Google Search, Gmail, Maps

Keras:
→ High-level API built ON TOP of TensorFlow
→ You write: model.fit(X, y)
→ TF handles: matrix math, GPU, gradients
→ 10x less code than raw TF or scratch!

From Day 71 scratch: ~150 lines
Keras equivalent: ~15 lines
Same result! 🔥
---

## 🔑 Sequential API

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu',
                        input_shape=(n_features,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1)

test_loss, test_acc = model.evaluate(
    X_test, y_test)
```

---

## 🔑 Adam Optimizer
Gradient Descent (Day 71):
W = W - lr * dW
Fixed learning rate — may be too slow or fast!

Adam (Adaptive Moment Estimation):
Adjusts learning rate PER PARAMETER!
Maintains moving averages of gradients.

m = β₁ * m + (1-β₁) * dW # 1st moment
v = β₂ * v + (1-β₂) * dW² # 2nd moment
W = W - lr * m / (√v + ε)

Faster convergence than plain SGD!
Default: lr=0.001, β₁=0.9, β₂=0.999
Usually works well without tuning! 🔥
---

## 🔑 Callbacks

```python
callbacks = [
    # Stop if val_loss doesn't improve for 10 epochs
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True),

    # Save best model automatically
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True),

    # Reduce LR if stuck
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    callbacks=callbacks,
    epochs=500)  # Early stopping saves us!
```

---

## 🔑 Batch Training
Full batch (all data at once):
→ Accurate gradient → slow for large datasets
→ Memory: must load entire dataset

Mini-batch (32-256 samples at a time):
→ Noisy gradient → trains faster!
→ Adds regularization effect!
→ Industry standard: batch_size=32 or 64

Stochastic (1 sample at a time):
→ Very noisy → rarely used

1 epoch = 1 pass over ALL training data
If n=1000 and batch=32 → 32 steps per epoch
---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Sequential model | Dense layers, compile, fit |
| 2 | Functional API | Multi-input models |
| 3 | Callbacks | EarlyStopping, ModelCheckpoint |
| 4 | Training visualization | Loss curves |
| 5 | Scratch NN vs Keras | Same result, 10x less code |
| 6 | Regression with Keras | Different loss functions |

---

## 💎 Important Realizations

1. **Keras hides the backprop we built in Day 71**
   model.fit() calls forward + backward + update
   automatically every epoch!
   All the gradient math from Day 71 = 1 line!

2. **Adam almost always beats plain SGD**
   Adam adapts learning rate per parameter.
   Start with Adam and lr=0.001 — works 90% of time!

3. **EarlyStopping is essential**
   Without it: 1000 epochs → overfit
   With it: stops at epoch 47 → perfect generalization!
   patience=10 is usually safe starting point!

4. **batch_size=32 is the industry default**
   Powers of 2 for GPU efficiency.
   32 → good balance of speed and stability.
   Try 16 (noisier) or 64 (smoother) if needed!

---

## 🎯 Next Goal (Day 73)

- CNN — Convolutional Neural Networks!
- Designed for images!
- Convolution, pooling, feature maps!
- Foundation for Skin Disease Detector (Day 77)!

---

*Day 72 complete — Keras mastered! ⚡🔥*
