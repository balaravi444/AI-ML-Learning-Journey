"""
Day 52 — Linear Regression
Topic: Sklearn Linear Regression - Full Usage
Date: 09 July 2026
Author: Bala Ravi

Production-ready Linear Regression with sklearn!
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures)
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def create_salary_dataset(
        n: int = 500) -> pd.DataFrame:
    """Create salary prediction dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai',
              'Delhi', 'Hyderabad', 'Pune']
    experience = np.random.randint(0, 12, n)
    skills = np.random.randint(2, 10, n)
    city = np.random.choice(cities, n)
    remote = np.random.choice([0, 1], n,
                               p=[0.7, 0.3])
    rating = np.round(
        np.random.uniform(3.0, 5.0, n), 1)

    city_bonus = {
        'Bangalore': 3, 'Mumbai': 2,
        'Delhi': 1, 'Hyderabad': 0.5, 'Pune': 0}

    salary = np.clip(
        10 + experience * 1.8 + skills * 0.5 +
        np.array([city_bonus[c] for c in city]) +
        remote * 2.5 + rating * 1.2 +
        np.random.normal(0, 2, n), 5, 55).round(1)

    return pd.DataFrame({
        'experience_years': experience,
        'skills_count': skills,
        'remote': remote,
        'rating': rating,
        'city_encoded': [
            city_bonus[c] for c in city],
        'salary_lpa': salary
    })


def simple_linear_regression_demo(
        df: pd.DataFrame) -> None:
    """Simple linear regression — one feature."""
    print("=== Simple Linear Regression ===\n")

    X = df[['experience_years']].values
    y = df['salary_lpa'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Formula: salary = "
          f"{model.coef_[0]:.2f} × experience + "
          f"{model.intercept_:.2f}")
    print(f"\nInterpretation:")
    print(f"  Each year of experience adds "
          f"₹{model.coef_[0]:.2f} LPA")
    print(f"  Starting salary (0 exp): "
          f"₹{model.intercept_:.2f} LPA")

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\nModel Performance:")
    print(f"  R²:  {r2:.4f}")
    print(f"  MAE: ₹{mae:.2f} LPA")

    # Predictions
    exp_range = np.array([[0], [2], [5], [8], [12]])
    preds = model.predict(exp_range)
    print(f"\nPredictions:")
    for exp, pred in zip(
            exp_range.flatten(), preds):
        print(f"  {exp} years → ₹{pred:.1f} LPA")


def multiple_linear_regression_demo(
        df: pd.DataFrame) -> None:
    """Multiple linear regression — all features."""
    print("\n=== Multiple Linear Regression ===\n")

    feature_cols = ['experience_years',
                    'skills_count', 'remote',
                    'rating', 'city_encoded']
    X = df[feature_cols].values
    y = df['salary_lpa'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Feature Importance (coefficients):")
    for feat, coef in sorted(
            zip(feature_cols, model.coef_),
            key=lambda x: abs(x[1]),
            reverse=True):
        direction = "↑" if coef > 0 else "↓"
        print(f"  {feat:<20}: "
              f"{coef:>7.3f} {direction}")

    print(f"\nModel Performance:")
    print(f"  R²:  {r2:.4f}")
    print(f"  MAE: ₹{mae:.2f} LPA")
    print(f"\nWith all features: "
          f"R² improved significantly!")


def polynomial_regression_demo(
        df: pd.DataFrame) -> None:
    """
    Polynomial regression — captures non-linearity!
    """
    print("\n=== Polynomial Regression ===\n")

    X = df[['experience_years']].values
    y = df['salary_lpa'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"{'Degree':<8} | {'R²':>8} | "
          f"{'MAE':>8} | {'Train R²':>9}")
    print("-" * 40)

    for degree in [1, 2, 3, 5]:
        pipe = Pipeline([
            ('poly', PolynomialFeatures(
                degree=degree,
                include_bias=False)),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        pipe.fit(X_train, y_train)

        train_r2 = r2_score(
            y_train, pipe.predict(X_train))
        test_r2 = r2_score(
            y_test, pipe.predict(X_test))
        mae = mean_absolute_error(
            y_test, pipe.predict(X_test))

        flag = ("⚠️ Overfit"
                if train_r2 - test_r2 > 0.1
                else "✅")
        print(f"{degree:<8} | {test_r2:>8.4f} | "
              f"{mae:>8.2f} | {train_r2:>9.4f} "
              f"{flag}")

    print("\n💡 degree=2 usually best for salary!")
    print("   Higher degrees → overfitting!")


def assumption_checking(
        df: pd.DataFrame) -> None:
    """Check Linear Regression assumptions."""
    print("\n=== Assumption Checking ===\n")

    from scipy import stats

    X = df[['experience_years',
             'skills_count']].values
    y = df['salary_lpa'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred

    print("1. Linearity Check:")
    for col in ['experience_years',
                'skills_count']:
        corr = df[col].corr(df['salary_lpa'])
        print(f"   {col} corr with salary: "
              f"{corr:.3f} "
              f"{'✅ Linear' if abs(corr) > 0.3 else '⚠️ Weak'}")

    print("\n2. Residual Normality (Shapiro-Wilk):")
    sample_residuals = residuals[:50]
    stat, p = stats.shapiro(sample_residuals)
    print(f"   p-value: {p:.4f}")
    print(f"   Normal: {p > 0.05} "
          f"{'✅' if p > 0.05 else '⚠️'}")

    print("\n3. Multicollinearity Check:")
    num_cols = ['experience_years',
                'skills_count', 'remote',
                'rating']
    corr_matrix = df[num_cols].corr()
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.8:
                print(f"   ⚠️ {num_cols[i]} & "
                      f"{num_cols[j]}: "
                      f"{corr:.3f} HIGH!")
            else:
                print(f"   ✅ {num_cols[i]} & "
                      f"{num_cols[j]}: "
                      f"{corr:.3f} OK")

    print("\n4. Residual Stats:")
    print(f"   Mean: {residuals.mean():.4f} "
          f"(should be ~0)")
    print(f"   Std:  {residuals.std():.4f}")


if __name__ == "__main__":
    df = create_salary_dataset()
    print(f"Dataset: {df.shape}\n")

    simple_linear_regression_demo(df)
    multiple_linear_regression_demo(df)
    polynomial_regression_demo(df)
    assumption_checking(df)
