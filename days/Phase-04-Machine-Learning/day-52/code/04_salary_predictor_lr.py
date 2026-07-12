"""
Day 52 — Linear Regression
Topic: Salary Prediction with Linear Regression
Date: 09 July 2026
Author: Bala Ravi

Complete salary predictor using Linear Regression!
Connects back to ArthAI and Job Market Analyzer!
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder,
    PolynomialFeatures)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')


def create_comprehensive_dataset(
        n: int = 1000) -> pd.DataFrame:
    """Create comprehensive salary dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune', 'Chennai']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']

    experience = np.random.randint(0, 12, n)
    city = np.random.choice(cities, n)
    role = np.random.choice(roles, n)
    skills = np.random.randint(2, 10, n)
    remote = np.random.choice([0, 1], n,
                               p=[0.7, 0.3])
    rating = np.round(
        np.random.uniform(3.0, 5.0, n), 1)

    city_b = {
        'Bangalore': 3, 'Mumbai': 2,
        'Delhi': 1, 'Hyderabad': 0.5,
        'Pune': 0, 'Chennai': 0.3}
    role_b = {
        'Data Scientist': 3,
        'ML Engineer': 4,
        'Data Analyst': -2,
        'AI Engineer': 5,
        'Data Engineer': 1}

    salary = np.clip(
        10 + experience * 1.8 +
        np.array([city_b[c] for c in city]) +
        np.array([role_b[r] for r in role]) +
        skills * 0.4 + remote * 2.5 +
        np.random.normal(0, 2, n), 5, 55).round(1)

    return pd.DataFrame({
        'experience_years': experience,
        'skills_count': skills,
        'remote': remote,
        'rating': rating,
        'city': city,
        'job_title': role,
        'salary_lpa': salary
    })


def build_lr_pipeline(
        df: pd.DataFrame) -> dict:
    """
    Build and compare Linear Regression models.
    """
    print("=== Salary Prediction with LR ===\n")

    num_features = ['experience_years',
                    'skills_count',
                    'remote', 'rating']
    cat_features = ['city', 'job_title']

    X = df[num_features + cat_features]
    y = df['salary_lpa']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(
            strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(
            strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (α=1)': Ridge(alpha=1.0),
        'Lasso (α=0.01)': Lasso(alpha=0.01),
    }

    results = {}

    print(f"{'Model':<22} | {'Test R²':>8} | "
          f"{'MAE (₹L)':>9} | {'CV R²':>8}")
    print("-" * 55)

    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cv = cross_val_score(
            Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)]),
            X, y, cv=5,
            scoring='r2').mean()

        results[name] = {
            'pipe': pipe,
            'r2': r2,
            'mae': mae,
            'cv': cv}

        print(f"{name:<22} | {r2:>8.4f} | "
              f"{mae:>9.2f} | {cv:>8.4f}")

    return results


def interpret_coefficients(
        df: pd.DataFrame) -> None:
    """
    Interpret Linear Regression coefficients.
    Business insights from ML model!
    """
    print("\n=== Coefficient Interpretation ===\n")

    # Use only numerical features for clarity
    num_features = ['experience_years',
                    'skills_count',
                    'remote', 'rating']
    X = df[num_features].values
    y = df['salary_lpa'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    print("What drives salary? (standardized coefs)")
    print("-" * 45)

    coefs = sorted(
        zip(num_features, model.coef_),
        key=lambda x: abs(x[1]),
        reverse=True)

    for feat, coef in coefs:
        direction = "↑ adds" if coef > 0 else "↓ reduces"
        print(f"  {feat:<22}: {coef:>7.3f} "
              f"({direction} salary)")

    print(f"\n📊 Salary Formula (approximate):")
    print(f"  salary = {model.intercept_:.1f}")
    for feat, coef in zip(
            num_features, model.coef_):
        sign = "+" if coef > 0 else ""
        print(f"         {sign}{coef:.2f} × {feat}")


def predict_my_salary(pipe: Pipeline) -> None:
    """Interactive salary prediction."""
    print("\n=== ArthAI Integration Preview ===\n")
    print("Predicting salaries for candidates:\n")

    candidates = pd.DataFrame({
        'experience_years': [1, 3, 5, 8, 10],
        'skills_count': [3, 5, 7, 8, 9],
        'remote': [0, 0, 1, 1, 0],
        'rating': [3.5, 4.0, 4.2, 4.5, 4.8],
        'city': ['Hyderabad', 'Bangalore',
                 'Mumbai', 'Bangalore',
                 'Bangalore'],
        'job_title': ['Data Analyst',
                      'Data Scientist',
                      'ML Engineer',
                      'AI Engineer',
                      'AI Engineer']
    })

    preds = pipe.predict(candidates)

    print(f"{'Role':<22} | {'City':<12} | "
          f"{'Exp':>4} | {'Predicted':>12}")
    print("-" * 55)

    for i, (_, row) in enumerate(
            candidates.iterrows()):
        print(f"{row['job_title']:<22} | "
              f"{row['city']:<12} | "
              f"{int(row['experience_years']):>4}yr | "
              f"₹{preds[i]:>9.1f} LPA")


if __name__ == "__main__":
    df = create_comprehensive_dataset()
    print(f"Dataset: {df.shape}\n")

    results = build_lr_pipeline(df)
    interpret_coefficients(df)

    best_pipe = results[
        'Ridge (α=1)']['pipe']
    predict_my_salary(best_pipe)

    print("\n💡 Linear Regression Takeaways:")
    print("   Best for: Linear relationships")
    print("   Strength: Interpretable coefficients!")
    print("   Weakness: Non-linear patterns")
    print("   Next: Decision Trees (Day 54)")
    print("         handle non-linear patterns! 🔥")
