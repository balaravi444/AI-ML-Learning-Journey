"""
Day 53 — Logistic Regression
Topic: Logistic Regression from Scratch
Date: 10 July 2026
Author: Bala Ravi

Understanding the MATH behind classification!
Sigmoid + Log Loss + Gradient Descent
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class LogisticRegressionScratch:
    """
    Logistic Regression from scratch.
    Shows exactly what sklearn does internally!

    Steps:
    1. z = X @ weights + bias     (linear part)
    2. p = sigmoid(z)             (probability)
    3. Loss = -log(p) if y=1      (log loss)
             -log(1-p) if y=0
    4. Update weights via gradient descent
    """

    def __init__(self,
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000) -> None:
        """
        Initialize Logistic Regression.

        Args:
            learning_rate: Gradient descent step size
            n_iterations: Number of training iterations
        """
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.
        Maps any real number to (0, 1).

        σ(z) = 1 / (1 + e^(-z))
        """
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self,
             X: np.ndarray,
             y: np.ndarray) -> 'LogisticRegressionScratch':
        """
        Train logistic regression using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)

        Returns:
            self
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            p = self.sigmoid(z)

            # Log Loss
            epsilon = 1e-15  # avoid log(0)
            p_clipped = np.clip(p, epsilon, 1 - epsilon)
            loss = -np.mean(
                y * np.log(p_clipped) +
                (1 - y) * np.log(1 - p_clipped))
            self.loss_history.append(loss)

            # Gradients
            error = p - y
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            # Update (gradient descent)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self,
                       X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Returns:
            Probability of class 1 for each sample
        """
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self,
                 X: np.ndarray,
                 threshold: float = 0.5
                 ) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            X: Feature matrix
            threshold: Decision boundary (default 0.5)

        Returns:
            Binary predictions (0 or 1)
        """
        return (self.predict_proba(X) >=
                threshold).astype(int)

    def score(self,
               X: np.ndarray,
               y: np.ndarray) -> float:
        """Calculate accuracy score."""
        return accuracy_score(y, self.predict(X))


def demonstrate_sigmoid() -> None:
    """Show sigmoid function behavior."""
    print("=== Sigmoid Function ===\n")

    test_values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]

    print(f"{'z':>6} | {'sigmoid(z)':>12} | "
          f"{'Interpretation':>25}")
    print("-" * 50)

    for z in test_values:
        sig = 1 / (1 + np.exp(-z))
        if sig >= 0.9:
            interp = "Very confident → class 1"
        elif sig >= 0.5:
            interp = "Leaning → class 1"
        elif sig >= 0.1:
            interp = "Leaning → class 0"
        else:
            interp = "Very confident → class 0"
        print(f"{z:>6} | {sig:>12.4f} | {interp:>25}")

    print(f"\n💡 Key: sigmoid(0) = 0.5 (decision boundary!)")


def demonstrate_log_loss() -> None:
    """Show why log loss is used."""
    print("\n=== Log Loss (Binary Cross-Entropy) ===\n")

    print("When actual = 1 (positive class):")
    print(f"{'Predicted p':>12} | "
          f"{'Loss':>8} | {'Quality':>15}")
    print("-" * 40)

    for p in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:
        loss = -np.log(p + 1e-15)
        quality = ("✅ Great" if p >= 0.9 else
                   "⚡ OK" if p >= 0.5 else
                   "⚠️ Bad" if p >= 0.2 else
                   "❌ Terrible")
        print(f"{p:>12.2f} | {loss:>8.4f} | "
              f"{quality:>15}")

    print("\nWhen actual = 0 (negative class):")
    print(f"{'Predicted p':>12} | "
          f"{'Loss':>8} | {'Quality':>15}")
    print("-" * 40)

    for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        loss = -np.log(1 - p + 1e-15)
        quality = ("✅ Great" if p <= 0.1 else
                   "⚡ OK" if p <= 0.5 else
                   "⚠️ Bad" if p <= 0.8 else
                   "❌ Terrible")
        print(f"{p:>12.2f} | {loss:>8.4f} | "
              f"{quality:>15}")


def train_and_compare() -> None:
    """Train scratch vs sklearn and compare."""
    print("\n=== Scratch vs Sklearn ===\n")

    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # From scratch
    scratch = LogisticRegressionScratch(
        learning_rate=0.1,
        n_iterations=500)
    scratch.fit(X_train_s, y_train)
    scratch_acc = scratch.score(X_test_s, y_test)

    # Sklearn
    from sklearn.linear_model import LogisticRegression
    sk_model = LogisticRegression(random_state=42)
    sk_model.fit(X_train_s, y_train)
    sk_acc = sk_model.score(X_test_s, y_test)

    print(f"Scratch accuracy:  {scratch_acc:.4f}")
    print(f"Sklearn accuracy:  {sk_acc:.4f}")
    print(f"\nScratch weights: {scratch.weights.round(3)}")
    print(f"Sklearn weights: {sk_model.coef_[0].round(3)}")
    print(f"\n✅ Very similar! Same algorithm!")

    # Loss curve
    print(f"\nTraining loss (first 5 iterations):")
    for i, loss in enumerate(
            scratch.loss_history[:5], 1):
        print(f"  Iteration {i}: {loss:.4f}")
    print(f"  ...")
    print(f"  Final loss: "
          f"{scratch.loss_history[-1]:.4f}")


if __name__ == "__main__":
    demonstrate_sigmoid()
    demonstrate_log_loss()
    train_and_compare()
