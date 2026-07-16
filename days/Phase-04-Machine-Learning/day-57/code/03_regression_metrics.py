"""
Day 57 — Model Evaluation & Metrics
Topic: Regression Metrics Deep Dive
Date: 14 July 2026
Author: Bala Ravi

MAE, RMSE, R², MAPE — when to use which!
Real application: salary prediction evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def regression_metrics_from_scratch(
        y_true: np.ndarray,
        y_pred: np.ndarray) -> dict:
    """
    Calculate all regression metrics from scratch.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary of all metrics
    """
    n = len(y_true)

    # MAE — Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # MSE — Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)

    # RMSE — Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R² — Coefficient of Determination
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # MAPE — Mean Absolute Percentage Error
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(
        np.abs((y_true[mask] - y_pred[mask]) /
                y_true[mask])) * 100

    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))

    # Median Absolute Error (robust to outliers)
    median_ae = np.median(np.abs(y_true - y_pred))

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape,
        'Max Error': max_error,
        'Median AE': median_ae
    }


def demonstrate_metrics_differences() -> None:
    """Show when each metric tells a different story."""
    print("=== When Metrics Tell Different Stories ===\n")

    np.random.seed(42)
    y_true = np.array([10, 20, 30, 40, 50,
                        60, 70, 80, 90, 100])

    predictions = {
        'Consistent errors': y_true + 3,
        'One big outlier': np.concatenate([
            y_true[:9] + 1, [y_true[9] + 50]]),
        'Percentage errors': y_true * 1.1,
        'Perfect': y_true.copy()
    }

    print(f"{'Model':<25} | {'MAE':>7} | "
          f"{'RMSE':>7} | {'R²':>6} | "
          f"{'MAPE%':>7} | {'Notes'}")
    print("-" * 80)

    for name, y_pred in predictions.items():
        metrics = regression_metrics_from_scratch(
            y_true.astype(float),
            y_pred.astype(float))

        note = ""
        if 'outlier' in name:
            note = "← RMSE penalizes outlier!"
        elif 'Consistent' in name:
            note = "← MAE=RMSE (uniform errors)"
        elif 'Percentage' in name:
            note = "← MAPE reveals % pattern"

        print(f"{name:<25} | {metrics['MAE']:>7.2f} | "
              f"{metrics['RMSE']:>7.2f} | "
              f"{metrics['R²']:>6.4f} | "
              f"{metrics['MAPE (%)']:>7.2f} | {note}")

    print(f"\n💡 Key Insights:")
    print(f"   RMSE punishes big errors MORE than MAE!")
    print(f"   Use RMSE when large errors are costly.")
    print(f"   Use MAE when all errors are equally bad.")
    print(f"   Use MAPE to explain to business (%).")
    print(f"   Use R² to compare across datasets.")


def create_salary_dataset(
        n: int = 800) -> tuple:
    """Create salary dataset."""
    np.random.seed(42)

    experience = np.random.randint(0, 12, n).astype(float)
    skills = np.random.randint(2, 10, n).astype(float)
    remote = np.random.choice([0, 1], n, p=[0.7, 0.3])
    rating = np.round(
        np.random.uniform(3.0, 5.0, n), 1)

    salary = np.clip(
        10 + experience * 1.8 + skills * 0.5 +
        remote * 2.5 + rating * 0.5 +
        np.random.normal(0, 2, n), 5, 55)

    X = np.column_stack([experience, skills,
                          remote, rating])
    y = salary

    return train_test_split(
        X, y, test_size=0.2, random_state=42)


def compare_regression_models() -> None:
    """Compare regression models with all metrics."""
    print("\n=== Regression Model Comparison ===\n")

    X_train, X_test, y_train, y_test = (
        create_salary_dataset())

    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge (α=1)': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'Decision Tree (d=5)': DecisionTreeRegressor(
            max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42, n_jobs=-1)
    }

    print(f"{'Model':<22} | {'R²':>7} | "
          f"{'MAE':>7} | {'RMSE':>7} | "
          f"{'MAPE%':>7}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = regression_metrics_from_scratch(
            y_test, y_pred)

        print(f"{name:<22} | "
              f"{metrics['R²']:>7.4f} | "
              f"{metrics['MAE']:>7.2f} | "
              f"{metrics['RMSE']:>7.2f} | "
              f"{metrics['MAPE (%)']:>7.2f}")

    print(f"\n💡 Metrics in business language:")
    print(f"   MAE = ₹2.3L → average prediction error")
    print(f"   RMSE = ₹2.8L → sensitive to big errors")
    print(f"   R² = 0.89 → explains 89% of salary variation")
    print(f"   MAPE = 11% → off by 11% on average")


def residual_analysis() -> None:
    """
    Analyze residuals to check model assumptions.
    Good model → residuals should be random noise!
    """
    print("\n=== Residual Analysis ===\n")

    X_train, X_test, y_train, y_test = (
        create_salary_dataset())

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    residuals = y_test - y_pred

    print("Residual Statistics:")
    print(f"  Mean:     {residuals.mean():.4f} "
          f"(should be ≈ 0)")
    print(f"  Std:      {residuals.std():.4f}")
    print(f"  Min:      {residuals.min():.4f}")
    print(f"  Max:      {residuals.max():.4f}")
    print(f"  Skewness: "
          f"{pd.Series(residuals).skew():.4f} "
          f"(should be ≈ 0)")

    # Check for systematic errors
    # Bin predictions and check average residual
    print(f"\nResiduals by prediction range:")
    print(f"{'Salary Range':>15} | "
          f"{'Avg Residual':>13} | {'Count':>6}")
    print("-" * 40)

    bins = [(0, 15), (15, 25), (25, 35), (35, 55)]
    for low, high in bins:
        mask = (y_pred >= low) & (y_pred < high)
        if mask.sum() > 0:
            avg_res = residuals[mask].mean()
            count = mask.sum()
            bias = ("← underpredict" if avg_res > 1
                    else "← overpredict" if avg_res < -1
                    else "← good")
            print(f"₹{low:>5}-{high:>3}L | "
                  f"{avg_res:>13.3f} | "
                  f"{count:>6} {bias}")

    print(f"\n✅ Good model: residuals ≈ 0 across all ranges")
    print(f"   Systematic bias → model needs improvement!")


if __name__ == "__main__":
    demonstrate_metrics_differences()
    compare_regression_models()
    residual_analysis()
