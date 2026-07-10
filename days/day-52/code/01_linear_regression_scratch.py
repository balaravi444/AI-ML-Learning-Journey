"""
Day 52 — Linear Regression
Topic: Linear Regression from Scratch
Date: 09 July 2026
Author: Bala Ravi

Understanding the MATH behind fit()!
OLS = Ordinary Least Squares
w = (X.T @ X)⁻¹ @ X.T @ y
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch!
    Shows exactly what sklearn's fit() does internally.

    Uses two methods:
    1. OLS (closed form) — exact solution
    2. Gradient Descent — iterative solution
    """

    def __init__(self,
                  method: str = 'ols',
                  learning_rate: float = 0.01,
                  n_iterations: int = 1000) -> None:
        """
        Initialize Linear Regression.

        Args:
            method: 'ols' or 'gradient_descent'
            learning_rate: Step size for GD
            n_iterations: GD iterations
        """
        self.method = method
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self,
             X: np.ndarray,
             y: np.ndarray) -> 'LinearRegressionScratch':
        """
        Train linear regression model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)

        Returns:
            self
        """
        n_samples, n_features = X.shape

        if self.method == 'ols':
            self._fit_ols(X, y)
        else:
            self._fit_gradient_descent(
                X, y, n_samples, n_features)

        return self

    def _fit_ols(self,
                  X: np.ndarray,
                  y: np.ndarray) -> None:
        """
        OLS closed form solution.
        w = (X.T @ X)⁻¹ @ X.T @ y

        This is EXACT — not iterative!
        sklearn uses this internally!
        """
        # Add bias column (column of 1s)
        X_b = np.c_[np.ones(X.shape[0]), X]

        # Normal equation — the OLS formula!
        theta = np.linalg.solve(
            X_b.T @ X_b,
            X_b.T @ y)

        self.bias = theta[0]
        self.weights = theta[1:]

    def _fit_gradient_descent(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_samples: int,
            n_features: int) -> None:
        """
        Gradient Descent solution.
        Same as neural network training!

        For each iteration:
        1. Forward pass: y_pred = X @ w + b
        2. Compute loss: MSE
        3. Compute gradients
        4. Update weights: w = w - lr * gradient
        """
        # Initialize weights to zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iter):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Compute loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # Compute gradients
            error = y_pred - y
            dw = (2 / n_samples) * X.T @ error
            db = (2 / n_samples) * np.sum(error)

            # Update weights (gradient descent!)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,
                 X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        y = X @ w + b
        """
        return X @ self.weights + self.bias

    def score(self,
               X: np.ndarray,
               y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


def demonstrate_ols_math() -> None:
    """
    Show OLS math step by step!
    """
    print("=== OLS Math — What fit() Does! ===\n")

    np.random.seed(42)
    n = 100

    # Simple case: salary = 2 * experience + 10
    experience = np.random.randint(0, 12, n)
    salary = 10 + 2 * experience + np.random.normal(0, 2, n)

    X = experience.reshape(-1, 1)
    y = salary

    # Manual OLS
    X_b = np.c_[np.ones(n), X]

    print("OLS Formula: w = (X.T @ X)⁻¹ @ X.T @ y")
    print()

    XtX = X_b.T @ X_b
    Xty = X_b.T @ y
    theta = np.linalg.solve(XtX, Xty)

    print(f"Bias (intercept): {theta[0]:.3f}")
    print(f"Weight (exp):     {theta[1]:.3f}")
    print(f"\nTrue values:")
    print(f"  Bias:   10.0")
    print(f"  Weight: 2.0")
    print(f"\n✅ OLS recovered true values!")

    # Compare with sklearn
    from sklearn.linear_model import LinearRegression
    sk_model = LinearRegression()
    sk_model.fit(X, y)
    print(f"\nSklearn verification:")
    print(f"  Intercept: {sk_model.intercept_:.3f}")
    print(f"  Coef:      {sk_model.coef_[0]:.3f}")
    print(f"\n✅ Identical results! "
          f"sklearn uses same OLS!")


def compare_ols_gd() -> None:
    """Compare OLS vs Gradient Descent."""
    print("\n=== OLS vs Gradient Descent ===\n")

    np.random.seed(42)
    n = 300

    X = np.random.randn(n, 3)
    y = (2*X[:, 0] + 3*X[:, 1] -
         X[:, 2] + np.random.randn(n))

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Scale for GD
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # OLS
    ols = LinearRegressionScratch(method='ols')
    ols.fit(X_train_s, y_train)
    ols_r2 = ols.score(X_test_s, y_test)

    # Gradient Descent
    gd = LinearRegressionScratch(
        method='gradient_descent',
        learning_rate=0.01,
        n_iterations=1000)
    gd.fit(X_train_s, y_train)
    gd_r2 = gd.score(X_test_s, y_test)

    print(f"OLS R²:              {ols_r2:.6f}")
    print(f"Gradient Descent R²: {gd_r2:.6f}")
    print(f"\nOLS weights:  {ols.weights.round(3)}")
    print(f"GD weights:   {gd.weights.round(3)}")

    print(f"\n💡 OLS gives exact solution!")
    print(f"   GD converges to same answer iteratively!")
    print(f"\n   GD is used when:")
    print(f"   → Dataset too large to invert matrix!")
    print(f"   → Neural networks (OLS not possible)")


if __name__ == "__main__":
    demonstrate_ols_math()
    compare_ols_gd()
