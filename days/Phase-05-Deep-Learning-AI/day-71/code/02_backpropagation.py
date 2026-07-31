"""
Day 71 — Neural Networks from Scratch
Topic: Backpropagation + Gradient Descent
Date: 28 July 2026
Author: Bala Ravi

The math that makes neural networks learn!
Chain rule applied to compute all gradients.
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def binary_cross_entropy(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> float:
    """
    Binary Cross-Entropy Loss.
    Same as Log Loss from Day 53!

    Loss = -mean(y*log(p) + (1-y)*log(1-p))
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred))


class NeuralNetworkScratch:
    """
    2-hidden-layer neural network from scratch.

    Architecture: input → hidden1 → hidden2 → output
    Activations:  ReLU → ReLU → Sigmoid

    Shows exactly what TensorFlow/Keras does!
    """

    def __init__(self,
                  n_inputs: int,
                  n_hidden1: int,
                  n_hidden2: int,
                  n_outputs: int = 1,
                  learning_rate: float = 0.01
                  ) -> None:
        """Initialize network with random weights."""
        self.lr = learning_rate
        self.loss_history = []

        # He initialization for ReLU
        scale1 = np.sqrt(2.0 / n_inputs)
        scale2 = np.sqrt(2.0 / n_hidden1)
        scale3 = np.sqrt(2.0 / n_hidden2)

        self.W1 = np.random.randn(
            n_inputs, n_hidden1) * scale1
        self.b1 = np.zeros((1, n_hidden1))

        self.W2 = np.random.randn(
            n_hidden1, n_hidden2) * scale2
        self.b2 = np.zeros((1, n_hidden2))

        self.W3 = np.random.randn(
            n_hidden2, n_outputs) * scale3
        self.b3 = np.zeros((1, n_outputs))

    def forward(self,
                 X: np.ndarray) -> dict:
        """
        Forward pass — compute predictions.

        Returns:
            Cache with all intermediate values
            (needed for backprop!)
        """
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)

        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)

        Z3 = A2 @ self.W3 + self.b3
        A3 = sigmoid(Z3)

        return {
            'X': X, 'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2,
            'Z3': Z3, 'A3': A3}

    def backward(self,
                  cache: dict,
                  y: np.ndarray) -> dict:
        """
        Backward pass — compute all gradients.

        Chain rule applied layer by layer!
        Goes BACKWARD from output to input.

        Args:
            cache: Values from forward pass
            y: True labels

        Returns:
            Dictionary of gradients
        """
        m = len(y)
        y = y.reshape(-1, 1)

        A3 = cache['A3']
        A2 = cache['A2']
        A1 = cache['A1']
        Z2 = cache['Z2']
        Z1 = cache['Z1']
        X = cache['X']

        # Output layer gradient
        # dL/dZ3 = Ŷ - Y (for BCE + sigmoid!)
        dZ3 = A3 - y
        dW3 = (1/m) * A2.T @ dZ3
        db3 = (1/m) * dZ3.sum(axis=0,
                                keepdims=True)

        # Hidden layer 2 gradient
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = (1/m) * A1.T @ dZ2
        db2 = (1/m) * dZ2.sum(axis=0,
                                keepdims=True)

        # Hidden layer 1 gradient
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = (1/m) * X.T @ dZ1
        db1 = (1/m) * dZ1.sum(axis=0,
                                keepdims=True)

        return {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3}

    def update_weights(self,
                        grads: dict) -> None:
        """Gradient descent weight update."""
        self.W1 -= self.lr * grads['dW1']
        self.b1 -= self.lr * grads['db1']
        self.W2 -= self.lr * grads['dW2']
        self.b2 -= self.lr * grads['db2']
        self.W3 -= self.lr * grads['dW3']
        self.b3 -= self.lr * grads['db3']

    def fit(self,
             X: np.ndarray,
             y: np.ndarray,
             epochs: int = 1000,
             verbose: bool = True) -> None:
        """
        Full training loop.

        For each epoch:
        1. Forward pass → predictions
        2. Compute loss
        3. Backward pass → gradients
        4. Update weights
        """
        for epoch in range(epochs):
            # Forward
            cache = self.forward(X)
            loss = binary_cross_entropy(
                y, cache['A3'].flatten())
            self.loss_history.append(loss)

            # Backward
            grads = self.backward(cache, y)

            # Update
            self.update_weights(grads)

            if verbose and (
                    epoch % 200 == 0 or
                    epoch == epochs - 1):
                acc = self.score(X, y)
                print(f"  Epoch {epoch:>4}: "
                      f"loss={loss:.4f}, "
                      f"acc={acc:.4f}")

    def predict_proba(self,
                       X: np.ndarray
                       ) -> np.ndarray:
        """Predict probabilities."""
        cache = self.forward(X)
        return cache['A3'].flatten()

    def predict(self,
                 X: np.ndarray,
                 threshold: float = 0.5
                 ) -> np.ndarray:
        """Predict binary labels."""
        return (self.predict_proba(X) >=
                threshold).astype(int)

    def score(self,
               X: np.ndarray,
               y: np.ndarray) -> float:
        """Accuracy score."""
        return (self.predict(X) == y).mean()


def train_and_evaluate() -> None:
    """Train NN from scratch and compare to sklearn."""
    print("=== Neural Network from Scratch ===\n")

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    np.random.seed(42)
    X, y = make_classification(
        n_samples=600,
        n_features=8,
        n_informative=5,
        random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Our scratch NN
    print("Training Neural Network from scratch:")
    nn = NeuralNetworkScratch(
        n_inputs=8,
        n_hidden1=32,
        n_hidden2=16,
        n_outputs=1,
        learning_rate=0.05)

    nn.fit(X_train_s, y_train,
            epochs=1000, verbose=True)

    scratch_test_acc = nn.score(X_test_s, y_test)

    # Sklearn MLP (same architecture)
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        learning_rate_init=0.05,
        max_iter=1000,
        random_state=42)
    mlp.fit(X_train_s, y_train)
    sklearn_acc = mlp.score(X_test_s, y_test)

    print(f"\n{'='*45}")
    print(f"Scratch NN test accuracy: "
          f"{scratch_test_acc:.4f}")
    print(f"Sklearn MLP test accuracy: "
          f"{sklearn_acc:.4f}")
    print(f"\n✅ Same architecture → similar results!")
    print(f"   Sklearn uses C optimizations → faster")
    print(f"   but the MATH is identical! 🔥")

    print(f"\nLoss history (first to last):")
    hist = nn.loss_history
    for i in [0, 200, 400, 600, 800, 999]:
        print(f"  Epoch {i:>4}: {hist[i]:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
