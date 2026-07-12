"""
Day 51 — ML Fundamentals + Scikit-learn
Topic: Bias-Variance Tradeoff
Date: 08 July 2026
Author: Bala Ravi

The most fundamental concept in ML!
Understanding WHY models succeed or fail!
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso)
from sklearn.model_selection import (
    cross_val_score, train_test_split)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def bias_variance_demonstration() -> None:
    """
    Demonstrate bias-variance tradeoff
    with different model complexities.
    """
    print("=== Bias-Variance Tradeoff ===\n")

    np.random.seed(42)
    n_experiments = 20

    def true_function(x):
        return np.sin(x) * 3 + x * 0.5

    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_test_true = true_function(X_test.flatten())

    results = {}

    for degree in [1, 3, 7, 15]:
        predictions = []

        for _ in range(n_experiments):
            X_train = np.random.uniform(
                0, 10, 30).reshape(-1, 1)
            y_train = (true_function(
                X_train.flatten()) +
                np.random.normal(0, 1, 30))

            model = Pipeline([
                ('poly', PolynomialFeatures(
                    degree=degree)),
                ('lr', LinearRegression())
            ])
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.append(pred)

        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)

        bias_sq = np.mean(
            (mean_pred - y_test_true) ** 2)
        variance = np.mean(predictions.var(axis=0))
        total_error = bias_sq + variance + 1.0

        results[degree] = {
            'bias_sq': bias_sq,
            'variance': variance,
            'total': total_error
        }

    print(f"{'Degree':<8} | {'Bias²':>8} | "
          f"{'Variance':>10} | {'Total Error':>12} | "
          f"{'Status':>15}")
    print("-" * 65)

    for degree, r in results.items():
        if r['bias_sq'] > 5:
            status = "❌ Underfitting"
        elif r['variance'] > 3:
            status = "⚠️ Overfitting"
        else:
            status = "✅ Good balance"

        print(f"{degree:<8} | {r['bias_sq']:>8.3f} | "
              f"{r['variance']:>10.3f} | "
              f"{r['total']:>12.3f} | "
              f"{status:>15}")

    print("\n💡 Key Insight:")
    print("   degree=1:  High bias (too simple)")
    print("   degree=15: High variance (too complex)")
    print("   degree=3:  Sweet spot! ✅")


def regularization_demo() -> None:
    """
    Demonstrate regularization to fix overfitting.
    Ridge and Lasso add penalty to prevent overfitting!
    """
    print("\n=== Regularization ===\n")

    np.random.seed(42)
    n = 50

    # True relationship: y = 3x + 2
    X = np.random.randn(n, 1)
    y = 3 * X.flatten() + 2 + np.random.randn(n)

    # Add many noise features (causes overfitting!)
    X_noisy = np.hstack([X] + [
        np.random.randn(n, 1)
        for _ in range(20)])

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X_noisy, y,
            test_size=0.3,
            random_state=42))

    models = {
        "Linear (no reg)": LinearRegression(),
        "Ridge (L2 reg)": Ridge(alpha=1.0),
        "Lasso (L1 reg)": Lasso(alpha=0.1)
    }

    print("Effect of regularization on overfitting:\n")
    print(f"{'Model':<22} | {'Train R²':>9} | "
          f"{'Test R²':>9} | {'Gap':>8}")
    print("-" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        gap = train_r2 - test_r2

        flag = "⚠️ Overfit" if gap > 0.2 else "✅"
        print(f"{name:<22} | {train_r2:>9.4f} | "
              f"{test_r2:>9.4f} | {gap:>6.4f} {flag}")

    print("\n💡 Ridge/Lasso reduces overfitting!")
    print("   They add a penalty for large weights.")
    print("   Forces model to use fewer/smaller weights!")


def learning_curves_demo() -> None:
    """
    Learning curves show if more data helps!
    """
    print("\n=== Learning Curves ===\n")

    from sklearn.model_selection import learning_curve
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    n = 500

    X = np.random.randn(n, 5)
    y = (2*X[:, 0] + 3*X[:, 1] -
         X[:, 2] + np.random.randn(n))

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42)

    train_sizes, train_scores, val_scores = (
        learning_curve(
            model, X, y,
            train_sizes=np.linspace(0.1, 1.0, 8),
            cv=5, scoring='r2',
            n_jobs=-1))

    print("Learning Curve Analysis:")
    print(f"\n{'Train Size':>12} | "
          f"{'Train R²':>10} | "
          f"{'Val R²':>10} | "
          f"{'Status':>15}")
    print("-" * 55)

    for size, tr, val in zip(
            train_sizes,
            train_scores.mean(axis=1),
            val_scores.mean(axis=1)):
        gap = tr - val
        if gap > 0.15:
            status = "⚠️ Overfit"
        elif val < 0.6:
            status = "❌ Underfit"
        else:
            status = "✅ Good"

        print(f"{int(size):>12} | {tr:>10.4f} | "
              f"{val:>10.4f} | {status:>15}")

    print("\n💡 Learning curve insights:")
    print("   If val score keeps improving with data")
    print("   → GET MORE DATA!")
    print("   If gap stays large with more data")
    print("   → Model is TOO COMPLEX (reduce!)")


if __name__ == "__main__":
    bias_variance_demonstration()
    regularization_demo()
    learning_curves_demo()
