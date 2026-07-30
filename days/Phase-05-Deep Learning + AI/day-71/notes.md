# Day 71 — Neural Networks from Scratch 🚀

**Date:** 28 July 2026
**Phase:** 5 — Deep Learning + AI
**Time Spent:** (add your hours)
**Resource Used:** [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

## 📚 Topics Covered

- What is a Neural Network
- Neurons, layers, activations
- Forward pass — computing predictions
- Loss function
- Backpropagation — computing gradients
- Gradient descent — updating weights
- Implementing NN from scratch in NumPy
- Why deep learning beats shallow ML

---

## 🔑 What is a Neural Network?
Stacked logistic regression! (Day 53)

Input layer → Hidden layers → Output layer

Each neuron:
z = w₁x₁ + w₂x₂ + ... + b (linear)
a = activation(z) (non-linear)

Without activation → just linear regression!
With activation → can learn ANY function!

Architecture example:
Input: 5 features
Hidden 1: 64 neurons
Hidden 2: 32 neurons
Output: 1 neuron (binary classification)
---

## 🔑 Activation Functions
Sigmoid: 1/(1+e^-z) → 0 to 1
Use: output layer (binary classification)
Problem: vanishing gradients in deep nets!

ReLU: max(0, z) → 0 to ∞
Use: hidden layers (most common!)
Advantage: no vanishing gradient!
Disadvantage: dying ReLU (z always < 0 → dead neuron)

Leaky ReLU: max(0.01z, z) → solves dying ReLU
Use: hidden layers alternative

Tanh: (e^z - e^-z)/(e^z + e^-z) → -1 to 1
Use: hidden layers (better than sigmoid)

Softmax: e^zᵢ / Σe^zⱼ → probabilities sum to 1
Use: output layer (multi-class classification)

---

## 🔑 Forward Pass
Given weights W and biases b:

Layer 1: Z¹ = X @ W¹ + b¹
A¹ = ReLU(Z¹)

Layer 2: Z² = A¹ @ W² + b²
A² = ReLU(Z²)

Output: Z³ = A² @ W³ + b³
Ŷ = sigmoid(Z³)

Ŷ = prediction (probability 0-1)

This is just matrix multiplication +
non-linear squashing! 🔥
---

## 🔑 Backpropagation
Chain rule applied to neural networks!

Goal: compute ∂Loss/∂W for every weight

Output layer:
δ³ = Ŷ - Y (output error)
dW³ = A².T @ δ³
db³ = sum(δ³)

Hidden layer 2:
δ² = (δ³ @ W³.T) * ReLU'(Z²)
dW² = A¹.T @ δ²
db² = sum(δ²)

Hidden layer 1:
δ¹ = (δ² @ W².T) * ReLU'(Z¹)
dW¹ = X.T @ δ¹
db¹ = sum(δ¹)

Update weights:
W = W - learning_rate * dW
b = b - learning_rate * db
---

## 🔑 Why Deep Learning Beats Shallow ML
Shallow ML (Random Forest, SVM):
→ Manual feature engineering needed
→ Can't learn hierarchical features
→ Image: you must extract edges manually

Deep Learning:
→ Learns features automatically!
→ Layer 1: learns edges
→ Layer 2: learns shapes
→ Layer 3: learns objects
→ No manual feature engineering!

For tabular data: ML often wins (faster, less data)
For images/text/audio: Deep Learning always wins!
---

## 💻 Programs Practiced

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | Neuron from scratch | z = wx + b, activation |
| 2 | Forward pass | Matrix multiply + activate |
| 3 | Loss functions | BCE, MSE, Cross-entropy |
| 4 | Backpropagation | Chain rule gradients |
| 5 | Full NN training loop | Forward + backward + update |
| 6 | Compare NN vs sklearn | Deep vs shallow |

---

## 💎 Important Realizations

1. **Neural Networks = stacked matrix multiplications**
   Each layer: Z = X @ W + b → A = activation(Z)
   That's it. The math is surprisingly simple!

2. **Backprop = chain rule from calculus**
   Every calculus student learns chain rule.
   Backprop just applies it to neural nets!
   The hard part: keeping track of shapes!

3. **ReLU solved deep learning**
   Before ReLU: sigmoid + deep nets = vanishing gradients
   With ReLU: train 100+ layer networks easily!
   One activation function changed everything!

4. **More data beats better architecture**
   A simple NN with 1M examples beats
   a complex NN with 1000 examples.
   Data is the moat in deep learning!

---

## 🎯 Next Goal (Day 72)

- TensorFlow & Keras
- Same NN, 10x less code
- model.compile, model.fit, model.evaluate
- Industry standard deep learning!

---

*Day 71 complete — Neural Networks from scratch! 🧠🔥*