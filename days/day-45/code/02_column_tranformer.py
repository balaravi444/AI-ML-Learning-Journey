"""
Day 46 — Data Preprocessing Pipeline
Topic: ColumnTransformer — Different preprocessing
       for different column types!
Date: 03 July 2026
Author: Bala Ravi

Numerical columns → impute + scale
Categorical columns → impute + encode
Different treatment for different types!
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, MinMaxScaler)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import r2_score


def create_mixed_dataset(n: int = 400) -> pd.DataFrame:
    """Create dataset with mixed column types."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']

    experience = np.random.randint(0, 12, n).astype(float)
    skills = np.random.randint(2, 10, n).astype(float)
    city = np.random.choice(cities, n).astype(object)
    role = np.random.choice(roles, n).astype(object)
    remote = np.random.choice([True, False], n)

    city_bonus = {'Bangalore': 3, 'Mumbai': 2,
                  'Delhi': 1, 'Hyderabad': 0.5,
                  'Pune': 0}
    role_bonus = {'Data Scientist': 3,
                  'ML Engineer': 4,
                  'Data Analyst': -2,
                  'AI Engineer': 5,
                  'Data Engineer': 1}

    salary = np.clip(
        10 + experience * 1.8 +
        np.array([city_bonus[c] for c in city]) +
        np.array([role_bonus[r] for r in role]) +
        np.random.normal(0, 2, n), 5, 55).round(1)

    # Add missing values
    exp_missing = np.random.choice(
        n, int(n * 0.05), replace=False)
    city_missing = np.random.choice(
        n, int(n * 0.03), replace=False)
    experience[exp_missing] = np.nan
    city[city_missing] = None

    return pd.DataFrame({
        'experience_years': experience,
        'skills_count': skills,
        'city': city,
        'job_title': role,
        'remote': remote.astype(int),
        'salary_lpa': salary
    })


def build_column_transformer_pipeline(
        num_features: list,
        cat_features: list) -> Pipeline:
    """
    Build complete pipeline with ColumnTransformer.
    Handles numerical and categorical separately!
    """
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(
            strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])

    # Combine with ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ], remainder='passthrough')

    # Full pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            random_state=42))
    ])

    return full_pipeline


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Train and evaluate the complete pipeline."""
    print("=== ColumnTransformer Pipeline ===\n")

    num_features = ['experience_years',
                    'skills_count', 'remote']
    cat_features = ['city', 'job_title']

    X = df[num_features + cat_features]
    y = df['salary_lpa']

    print(f"Features:")
    print(f"  Numerical ({len(num_features)}): "
          f"{num_features}")
    print(f"  Categorical ({len(cat_features)}): "
          f"{cat_features}")
    print(f"  Missing values: {X.isnull().sum().sum()}")

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    pipeline = build_column_transformer_pipeline(
        num_features, cat_features)

    print(f"\nTraining pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"Test R²: {r2:.4f}")
    print(f"({r2*100:.1f}% of variance explained!)")

    # CV
    cv_scores = cross_val_score(
        pipeline, X, y, cv=5, scoring='r2')
    print(f"\nCV R²: {cv_scores.mean():.4f} "
          f"± {cv_scores.std():.4f}")

    # Sample predictions
    print(f"\nSample predictions:")
    sample = X_test.head(3)
    preds = pipeline.predict(sample)
    actual = y_test.head(3)
    for i, (pred, act) in enumerate(
            zip(preds, actual)):
        print(f"  {i+1}. Predicted: ₹{pred:.1f}L "
              f"| Actual: ₹{act:.1f}L "
              f"| Error: ₹{abs(pred-act):.1f}L")

    return pipeline


def predict_new_candidate(pipeline: Pipeline) -> None:
    """Predict salary for new job candidates."""
    print("\n=== Salary Predictions for New Candidates ===\n")

    new_candidates = pd.DataFrame({
        'experience_years': [2, 5, 8, 1, 10],
        'skills_count': [4, 7, 9, 3, 8],
        'remote': [0, 1, 0, 0, 1],
        'city': ['Bangalore', 'Mumbai',
                 'Bangalore', 'Hyderabad', 'Delhi'],
        'job_title': ['Data Analyst',
                      'ML Engineer',
                      'AI Engineer',
                      'Data Scientist',
                      'Data Engineer']
    })

    predictions = pipeline.predict(new_candidates)

    print(f"{'Role':<25} {'City':<12} "
          f"{'Exp':>4} {'Skills':>6} "
          f"{'Predicted':>12}")
    print("-" * 65)

    for i, (_, row) in enumerate(
            new_candidates.iterrows()):
        print(f"{row['job_title']:<25} "
              f"{row['city']:<12} "
              f"{row['experience_years']:>4.0f} "
              f"{row['skills_count']:>6.0f} "
              f"₹{predictions[i]:>10.1f}L")


if __name__ == "__main__":
    df = create_mixed_dataset()
    print(f"Dataset: {df.shape}\n")

    pipeline = train_and_evaluate(df)
    predict_new_candidate(pipeline)
