"""
Day 52 — Linear Regression
Topic: Ridge and Lasso Regularization
Date: 09 July 2026
Author: Bala Ravi

Regularization = prevent overfitting!
Ridge (L2) → shrink weights
Lasso (L1) → zero out weights (feature selection!)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso,
    ElasticNet)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


def create_high_dimensional_dataset(
        n: int = 200) -> tuple:
    """
    Create dataset with many features
    (some relevant, some noise!)
    """
    np.random.seed(42)

    n_real = 5     # relevant features
    n_noise = 25   # noise features

    # Real features
    X_real = np.random.randn(n, n_real)
    y = (2*X_real[:, 0] + 3*X_real[:, 1] -
         X_real[:, 2] + 1.5*X_real[:, 3] +
         np.random.randn(n))

    # Noise features
    X_noise = np.random.randn(n, n_noise)

    X = np.hstack([X_real, X_noise])

    feat_names = (
        [f'real_{i}' for i in range(n_real)] +
        [f'noise_{i}' for i in range(n_noise)])

    return X, y, feat_names


def compare_regularization() -> None:
    """Compare Linear, Ridge, Lasso effects."""
    print("=== Regularization Comparison ===\n")

    X, y, feat_names = (
        create_high_dimensional_dataset())

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Linear (no reg)': LinearRegression(),
        'Ridge (L2, α=1)': Ridge(alpha=1.0),
        'Ridge (L2, α=10)': Ridge(alpha=10.0),
        'Lasso (L1, α=0.1)': Lasso(alpha=0.1),
        'Lasso (L1, α=1)': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(
            alpha=0.1, l1_ratio=0.5)
    }

    print(f"{'Model':<22} | {'Train R²':>9} | "
          f"{'Test R²':>9} | {'Gap':>7} | "
          f"{'Non-zero W':>10}")
    print("-" * 70)

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        tr = r2_score(
            y_train, model.predict(X_train_s))
        te = r2_score(
            y_test, model.predict(X_test_s))
        gap = tr - te
        nonzero = np.sum(
            np.abs(model.coef_) > 0.01)

        flag = "⚠️" if gap > 0.2 else "✅"
        print(f"{name:<22} | {tr:>9.4f} | "
              f"{te:>9.4f} | {gap:>7.4f} | "
              f"{nonzero:>10} {flag}")

    print(f"\n💡 Key Observations:")
    print(f"   Linear: High gap → overfitting!")
    print(f"   Ridge:  Reduces gap, keeps all weights")
    print(f"   Lasso:  Reduces gap, ZEROS noise weights!")


def lasso_feature_selection() -> None:
    """
    Show Lasso's automatic feature selection!
    Zeros out noise features!
    """
    print("\n=== Lasso Feature Selection ===\n")

    X, y, feat_names = (
        create_high_dimensional_dataset())

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_s, y_train)

    print("Feature coefficients after Lasso:")
    print(f"\n{'Feature':<15} | {'Coefficient':>12} | "
          f"{'Status':>15}")
    print("-" * 47)

    real_kept = 0
    noise_kept = 0

    for name, coef in zip(
            feat_names, lasso.coef_):
        is_zero = abs(coef) < 0.01
        is_real = name.startswith('real')

        if is_real and not is_zero:
            real_kept += 1
            status = "✅ KEPT (real)"
        elif is_real and is_zero:
            status = "⚠️ ZEROED (real!)"
        elif not is_real and is_zero:
            noise_kept += 0
            status = "✅ ZEROED (noise)"
        else:
            noise_kept += 1
            status = "⚠️ KEPT (noise)"

        if abs(coef) > 0.01 or is_real:
            print(f"{name:<15} | {coef:>12.4f} | "
                  f"{status:>15}")

    print(f"\nSummary:")
    print(f"  Original features: {len(feat_names)}")
    print(f"  Non-zero after Lasso: "
          f"{np.sum(np.abs(lasso.coef_) > 0.01)}")
    print(f"  Noise features zeroed: "
          f"{25 - noise_kept}")
    print(f"\n✅ Lasso automatically eliminated "
          f"most noise features!")


def alpha_tuning() -> None:
    """
    Find best alpha using cross-validation.
    """
    print("\n=== Alpha Tuning (Ridge) ===\n")

    X, y, _ = create_high_dimensional_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    print(f"{'Alpha':>8} | {'Ridge CV R²':>12} | "
          f"{'Lasso CV R²':>12}")
    print("-" * 38)

    best_ridge_alpha = None
    best_ridge_score = -np.inf
    best_lasso_alpha = None
    best_lasso_score = -np.inf

    for alpha in alphas:
        ridge_scores = cross_val_score(
            Ridge(alpha=alpha),
            X_scaled, y, cv=5,
            scoring='r2')
        lasso_scores = cross_val_score(
            Lasso(alpha=alpha),
            X_scaled, y, cv=5,
            scoring='r2')

        r_mean = ridge_scores.mean()
        l_mean = lasso_scores.mean()

        print(f"{alpha:>8} | {r_mean:>12.4f} | "
              f"{l_mean:>12.4f}")

        if r_mean > best_ridge_score:
            best_ridge_score = r_mean
            best_ridge_alpha = alpha
        if l_mean > best_lasso_score:
            best_lasso_score = l_mean
            best_lasso_alpha = alpha

    print(f"\nBest Ridge alpha: {best_ridge_alpha} "
          f"(R²={best_ridge_score:.4f})")
    print(f"Best Lasso alpha: {best_lasso_alpha} "
          f"(R²={best_lasso_score:.4f})")
    print(f"\n💡 In production: use RidgeCV or LassoCV")
    print(f"   for automatic alpha selection!")


if __name__ == "__main__":
    compare_regularization()
    lasso_feature_selection()
    alpha_tuning()
