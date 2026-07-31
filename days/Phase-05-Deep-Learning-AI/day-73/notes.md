# Day 73 — CNN: Convolutional Neural Networks 🚀

**Date:** 30 July 2026
**Phase:** 5 — Deep Learning
**Time Spent:** (3 hours)
**Resource Used:** [CS231n](https://cs231n.github.io/) | [TF Docs](https://tensorflow.org)

---

## 📚 Topics Covered

- Why CNNs for images
- Convolution operation — filters + feature maps
- Pooling — MaxPool, AveragePool
- CNN architecture — Conv → Pool → Flatten → Dense
- Padding and Stride
- Multiple filters = multiple feature maps
- Build CNN from scratch with Keras
- Compare CNN vs Dense on image data

---

## 🔑 Why CNN for Images?
Dense NN on 224×224 image:
224 × 224 × 3 = 150,528 inputs
First Dense layer (512 neurons):
150,528 × 512 = 77M parameters!

Problems:
→ Too many parameters → slow + overfit
→ Ignores spatial structure of image
→ Pixel (0,0) treated same as pixel (112,112)
→ Image shifted 1 pixel = completely different!

CNN fixes all of this:
→ Shared weights (filters) → few parameters
→ Learns local patterns (edges, textures)
→ Translation invariant (pooling)
→ Spatial structure preserved!
---

## 🔑 Convolution Operation
Filter (kernel): small matrix e.g. 3×3

Image patch: Filter: Output:
[1, 0, 1] [1, 0, -1]
[0, 1, 0] × [2, 0, -2] = element-wise multiply + sum
[1, 0, 1] [1, 0, -1]

= (1×1 + 0×0 + 1×-1) +
(0×2 + 1×0 + 0×-2) +
(1×1 + 0×0 + 1×-1)
= (1 + 0 - 1) + (0 + 0 + 0) + (1 + 0 - 1)
= 0
Slide this filter across ENTIRE image:
→ One number per position = feature map!

Different filters learn different features:
Filter 1 → vertical edges
Filter 2 → horizontal edges
Filter 3 → diagonal edges
Filter 4 → color blobs
...
Filter 64 → complex textures
---

## 🔑 CNN Architecture
Input Image
↓
Conv2D(32 filters, 3×3) + ReLU → feature maps
↓
MaxPool2D(2×2) → reduce size
↓
Conv2D(64 filters, 3×3) + ReLU → deeper features
↓
MaxPool2D(2×2) → reduce more
↓
Conv2D(128 filters, 3×3) + ReLU → complex features
↓
GlobalAveragePooling2D() → flatten smartly
↓
Dense(256) + ReLU
↓
Dense(n_classes) + Softmax → predictions

Early layers: learn edges + textures
Deep layers: learn objects + concepts
---

## 🔑 MaxPooling
Takes MAX value in each 2×2 window.
Reduces spatial size by 2×.

Feature map (4×4): After MaxPool(2×2):
[1, 3, 2, 4] [3, 4]
[5, 6, 7, 8] → [6, 8]
[1, 2, 3, 4]
[9, 8, 7, 6]

Benefits:
→ Reduces parameters (faster training)
→ Translation invariance
→ Keeps most important features (max signal)
---

## 🔑 Key Parameters

```python
keras.layers.Conv2D(
    filters=32,        # number of feature maps
    kernel_size=(3,3), # filter size (usually 3×3)
    strides=(1,1),     # how much filter moves
    padding='same',    # 'same': keep size, 'valid': shrink
    activation='relu')

keras.layers.MaxPooling2D(
    pool_size=(2,2),   # window size
    strides=(2,2))     # usually = pool_size
```

---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Convolution from scratch | Manual filter sliding |
| 2 | Build CNN with Keras | Conv2D, MaxPool2D |
| 3 | Feature map visualization | What filters learn |
| 4 | Dense vs CNN on MNIST | Why CNN wins |
| 5 | Architecture patterns | VGG-style blocks |

---

## 💎 Important Realizations

1. **Filters are learned — not handcrafted**
   Before deep learning: engineers manually designed
   edge detectors, texture filters.
   CNN learns the best filters from data automatically!

2. **Parameter sharing is the key insight**
   Same 3×3 filter applied to EVERY position.
   1 filter = 9 parameters regardless of image size!
   Dense NN: each position gets its own weights.

3. **Depth > Width for CNNs**
   More Conv layers = learns more abstract features.
   Layer 1: edges, Layer 2: shapes, Layer 3: objects.
   This hierarchy mirrors how human vision works!

4. **GlobalAveragePooling > Flatten for images**
   Flatten: 7×7×512 = 25,088 → Dense(256) = 6.4M params!
   GAP: 512 averages → Dense(256) = 131K params!
   Much less overfit with GAP! 🔥

---

## 🎯 Next Goal (Day 74)

- Transfer Learning!
- Use MobileNetV2 pretrained on ImageNet
- Fine-tune for our skin disease detection!
- 90% accuracy with 100 training images!

---

*Day 73 complete — CNN mastered! 🖼️🔥*





