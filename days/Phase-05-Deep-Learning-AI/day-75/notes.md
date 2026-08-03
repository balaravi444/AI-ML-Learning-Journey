# Day 75 — Image Preprocessing + Augmentation 🚀

**Date:** 01 August 2026
**Phase:** 5 — Deep Learning
**Time Spent:** (3 hours)
**Resource Used:** [TF Image Guide](https://tensorflow.org/tutorials/images)

---

## 📚 Topics Covered

- Image loading and resizing
- Normalization strategies
- Data augmentation deep dive
- ImageDataGenerator vs tf.data
- Class imbalance in image datasets
- Building efficient data pipelines
- Preparing HAM10000 for skin detection

---

## 🔑 Why Preprocessing Matters
Raw image → model = terrible results!
Preprocessed image → model = great results!

What preprocessing does:

Resize → all images same size (224×224)
Normalize → pixel values 0-1 or -1 to 1
Augment → artificially increase dataset
Balance → handle class imbalance
Batch → efficient GPU utilization

Bad preprocessing can hurt more than
bad model architecture!
---

## 🔑 Normalization Strategies
Raw pixels: 0 to 255 (uint8)

Strategy 1: Divide by 255
pixel / 255.0 → 0.0 to 1.0
Simple, works well

Strategy 2: ImageNet normalization
MobileNetV2 expects -1 to 1!
preprocess_input(img) → handles this
MUST use this with MobileNetV2!

Strategy 3: Per-channel normalization
(img - mean) / std per channel
Most rigorous for custom datasets

Wrong normalization → 20-30% accuracy drop! ⚠️
---

## 🔑 Data Augmentation Types
Geometric transforms:
RandomFlip → mirror left/right
RandomRotation → rotate ±20°
RandomZoom → zoom in/out ±15%
RandomTranslation → shift position

Color transforms:
RandomContrast → vary contrast
RandomBrightness → vary brightness
RandomSaturation → vary color intensity

Medical imaging specific:
Elastic deformation → subtle shape change
Grid distortion → texture variation

For skin diseases:
✅ Flip (lesions are symmetric)
✅ Rotation (any angle in real life)
✅ Zoom (different photo distances)
✅ Brightness (different lighting)
❌ Horizontal flip only (not vertical always)
---

## 🔑 tf.data Pipeline

```python
# Old way (slow):
ImageDataGenerator → fits in RAM only

# New way (fast, production):
tf.data.Dataset

dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, labels))

dataset = (dataset
    .map(load_and_preprocess,
         num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(buffer_size=1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE))

AUTOTUNE → TF decides optimal parallelism!
cache()   → after first epoch, data stays in RAM
prefetch() → GPU never waits for CPU! 🔥
```

---

## 💎 Important Realizations

1. **Augmentation only during training**
   Val and test: NO augmentation!
   Only apply to training set!
   Augmented val = unfair evaluation!

2. **preprocess_input is model-specific**
   MobileNetV2: preprocess_input → -1 to 1
   VGG16: different preprocess_input!
   Always match preprocessing to model!

3. **Class weights fix class imbalance**
   HAM10000: 67% Melanocytic Nevi, 2% Melanoma
   Without weights → model ignores Melanoma!
   class_weight = {0: 1.0, 4: 15.0}
   Tells model Melanoma is 15× more important!

4. **tf.data is production standard**
   ImageDataGenerator: deprecated, slow
   tf.data: fast, parallelized, GPU-optimized
   Always use tf.data for new projects!

---

## 🎯 Next Goal (Day 76)

- Model optimization — callbacks deep dive
- Learning rate scheduling
- Mixed precision training
- Model checkpointing strategy
- Everything ready for Skin Disease project!

---

*Day 75 complete — Image Pipeline mastered! 🖼️🔥*

