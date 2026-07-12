"""
Day 45 — Statistics for ML
Topic: Complete Statistics → ML Connection
Date: 02 July 2026
Author: Bala Ravi

Every ML formula has statistics underneath!
This file connects ALL statistics to ML!
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    cross_val_score, train_test_split)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def show_stats_in_ml() -> None:
    """
    Demonstrate how statistics appears in ML!
    """
    np.random.seed(42)
    n = 300

    experience = np.random.randint(0, 12, n)
    salary = np.clip(
        10 + experience * 1.8 +
        np.random.normal(0, 2, n), 5, 55)

    X = experience.reshape(-1, 1)
    y = salary

    print("=== Statistics Inside ML ===\n")

    # 1. Mean and Std in StandardScaler
    mean = X.mean()
    std = X.std()
    X_scaled = (X - mean) / std
    print("1. StandardScaler = (x - mean) / std")
    print(f"   Mean of experience: {mean:.2f}")
    print(f"   Std of experience:  {std:.2f}")
    print(f"   After scaling - mean: "
          f"{X_scaled.mean():.4f} (≈0)")
    print(f"   After scaling - std:  "
          f"{X_scaled.std():.4f} (≈1)")

    # 2. Correlation in feature selection
    corr = np.corrcoef(experience, salary)[0, 1]
    print(f"\n2. Pearson Correlation")
    print(f"   experience vs salary: {corr:.3f}")
    print(f"   → Strong positive! ✅ Keep feature!")

    # 3. Linear Regression = OLS (stats!)
    model = LinearRegression()
    model.fit(X, y)
    print(f"\n3. Linear Regression = OLS Statistics")
    print(f"   Coefficient: {model.coef_[0]:.3f}")
    print(f"   Intercept:   {model.intercept_:.3f}")
    print(f"   Formula: salary = "
          f"{model.coef_[0]:.2f} × exp + "
          f"{model.intercept_:.2f}")

    # 4. R-squared = statistical measure
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"\n4. R² Score = Statistical Measure")
    print(f"   R² = {r2:.4f}")
    print(f"   → Model explains "
          f"{r2*100:.1f}% of variance!")
    print(f"   → From statistics: "
          f"1 - SS_res/SS_tot")

    # 5. Confidence Interval for model prediction
    residuals = y - y_pred
    se = np.std(residuals)
    ci_width = 1.96 * se

    print(f"\n5. Confidence Interval for Predictions")
    print(f"   Standard error: ₹{se:.2f} LPA")
    sample_pred = model.predict([[5]])[0]
    print(f"   5yr exp prediction: "
          f"₹{sample_pred:.1f} LPA")
    print(f"   95% CI: ₹{sample_pred-ci_width:.1f} "
          f"to ₹{sample_pred+ci_width:.1f} LPA")

    # 6. Cross-validation = statistical sampling!
    rf = RandomForestRegressor(
        n_estimators=50, random_state=42)
    cv_scores = cross_val_score(
        rf, X, y, cv=5, scoring='r2')

    print(f"\n6. Cross-Validation = Statistical Sampling")
    print(f"   5-fold CV scores: {cv_scores.round(3)}")
    print(f"   Mean R²: {cv_scores.mean():.3f} "
          f"± {cv_scores.std():.3f}")
    print(f"   95% CI: ({cv_scores.mean()-1.96*cv_scores.std():.3f}, "
          f"{cv_scores.mean()+1.96*cv_scores.std():.3f})")

    # 7. p-value for model evaluation
    print(f"\n7. Statistical Significance Check")
    t_stat, p_value = stats.pearsonr(experience, salary)
    print(f"   Pearson r: {t_stat:.3f}, "
          f"p-value: {p_value:.6f}")
    print(f"   p < 0.001 → HIGHLY significant!")
    print(f"   Experience is a statistically")
    print(f"   significant predictor of salary! ✅")


def mse_loss_stats() -> None:
    """MSE loss function = statistics concept."""
    print("\n=== MSE Loss = Statistics ===\n")

    np.random.seed(42)
    y_true = np.array([25, 30, 18, 35, 22,
                       28, 40, 15, 32, 27])
    y_pred = np.array([24, 31, 19, 33, 23,
                       29, 38, 16, 31, 28])

    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    print(f"Predictions vs Actual:")
    for actual, pred in zip(y_true, y_pred):
        error = pred - actual
        print(f"  Actual: ₹{actual}L → "
              f"Predicted: ₹{pred}L "
              f"(error: {error:+.0f}L)")

    print(f"\nMetrics (all from statistics!):")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: ₹{rmse:.2f} LPA average error")
    print(f"  MAE:  ₹{mae:.2f} LPA average abs error")
    print(f"\n  RMSE = standard deviation of residuals!")
    print(f"  Pure statistics applied to ML! 🔥")


if __name__ == "__main__":
    show_stats_in_ml()
    mse_loss_stats()

    print("\n" + "=" * 50)
    print("  Statistics → ML Connection Map")
    print("=" * 50)
    connections = [
        ("Mean/Std", "StandardScaler, Batch Norm"),
        ("Normal Distribution", "Linear Regression assumptions"),
        ("Correlation", "Feature selection"),
        ("Hypothesis Test", "A/B model testing"),
        ("Bayes Theorem", "Naive Bayes classifier"),
        ("CLT", "Batch gradient descent"),
        ("Confidence Interval", "Model uncertainty"),
        ("OLS", "Linear Regression solver"),
        ("R-squared", "Model evaluation metric"),
        ("MSE/RMSE", "Loss functions"),
    ]
    for stat, ml in connections:
        print(f"  {stat:<22} → {ml}")
