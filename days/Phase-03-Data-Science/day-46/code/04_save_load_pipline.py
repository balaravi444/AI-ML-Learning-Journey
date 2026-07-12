"""
Day 46 — Data Preprocessing Pipeline
Topic: Saving and Loading Pipelines
Date: 03 July 2026
Author: Bala Ravi

joblib.dump() → save entire ML system
joblib.load() → load and predict instantly!

This is HOW ML models are deployed in production!
ArthAI salary predictor will use this! 🔥
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def create_dataset(n: int = 500) -> pd.DataFrame:
    """Create comprehensive job dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']

    experience = np.random.randint(0, 12, n).astype(float)
    city = np.random.choice(cities, n)
    role = np.random.choice(roles, n)

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

    exp_missing = np.random.choice(
        n, int(n * 0.05), replace=False)
    experience[exp_missing] = np.nan

    return pd.DataFrame({
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n),
        'city': city,
        'job_title': role,
        'remote': np.random.choice([0, 1], n,
                                    p=[0.7, 0.3]),
        'salary_lpa': salary
    })


def train_and_save_pipeline(
        df: pd.DataFrame,
        save_path: str = 'salary_predictor.pkl'
        ) -> Pipeline:
    """
    Train complete pipeline and save to disk.
    This is how ML models are deployed!
    """
    print("=== Train & Save Pipeline ===\n")

    num_features = ['experience_years',
                    'skills_count', 'remote']
    cat_features = ['city', 'job_title']

    X = df[num_features + cat_features]
    y = df['salary_lpa']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Build pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(
            strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1))
    ])

    print("Training pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test R²:  {r2:.4f}")
    print(f"Test MAE: ₹{mae:.2f} LPA")

    # Save pipeline
    joblib.dump(pipeline, save_path)
    file_size = os.path.getsize(save_path) / 1024
    print(f"\n✅ Pipeline saved: {save_path}")
    print(f"   File size: {file_size:.1f} KB")
    print(f"   Contains: preprocessor + model!")

    return pipeline


def load_and_predict(
        save_path: str = 'salary_predictor.pkl'
        ) -> None:
    """
    Load saved pipeline and make predictions.
    Simulates a production ML API!
    """
    print("\n=== Load & Predict (Production!) ===\n")

    # Load pipeline
    loaded_pipeline = joblib.load(save_path)
    print(f"✅ Pipeline loaded from: {save_path}")

    # New candidate data — simulates API request!
    new_candidates = pd.DataFrame({
        'experience_years': [1, 3, 6, 10, 0],
        'skills_count': [3, 6, 8, 9, 2],
        'remote': [0, 1, 0, 1, 0],
        'city': ['Hyderabad', 'Bangalore',
                 'Mumbai', 'Bangalore', 'Pune'],
        'job_title': ['Data Analyst',
                      'Data Scientist',
                      'ML Engineer',
                      'AI Engineer',
                      'Data Analyst']
    })

    # ONE LINE prediction — pipeline handles everything!
    predictions = loaded_pipeline.predict(new_candidates)

    print(f"\nSalary Predictions:")
    print(f"{'Role':<25} {'City':<12} "
          f"{'Exp':>4} {'Predicted':>12}")
    print("-" * 56)

    for i, (_, row) in enumerate(
            new_candidates.iterrows()):
        print(f"{row['job_title']:<25} "
              f"{row['city']:<12} "
              f"{int(row['experience_years']):>4} yrs "
              f"₹{predictions[i]:>8.1f} LPA")

    print(f"\n🚀 This is how ArthAI salary")
    print(f"   predictor will work!")
    print(f"   Load pipeline → predict → return!")


def simulate_api_endpoint(
        save_path: str = 'salary_predictor.pkl'
        ) -> None:
    """Simulate a FastAPI salary prediction endpoint."""
    print("\n=== Simulated FastAPI Endpoint ===\n")

    pipeline = joblib.load(save_path)

    def predict_salary(
            experience: float,
            skills: int,
            city: str,
            role: str,
            remote: bool) -> dict:
        """
        Simulate FastAPI endpoint for salary prediction.
        This is EXACTLY how ArthAI would expose
        the ML model as an API!
        """
        input_data = pd.DataFrame({
            'experience_years': [experience],
            'skills_count': [skills],
            'remote': [int(remote)],
            'city': [city],
            'job_title': [role]
        })

        prediction = pipeline.predict(input_data)[0]
        confidence_range = 2.5  # ±₹2.5L (MAE)

        return {
            'predicted_salary': round(prediction, 1),
            'range_low': round(
                prediction - confidence_range, 1),
            'range_high': round(
                prediction + confidence_range, 1),
            'currency': 'LPA (₹ Lakhs Per Annum)',
            'model_version': '1.0'
        }

    # Test the endpoint
    test_cases = [
        (3, 6, 'Bangalore',
         'Data Scientist', False),
        (7, 9, 'Mumbai', 'ML Engineer', True),
        (1, 3, 'Hyderabad',
         'Data Analyst', False),
    ]

    print("API Responses:")
    for exp, skills, city, role, remote in test_cases:
        result = predict_salary(
            exp, skills, city, role, remote)
        print(f"\n  Request: {role}, {city}, "
              f"{exp}yr exp, remote={remote}")
        print(f"  Response:")
        print(f"    Salary: "
              f"₹{result['predicted_salary']} LPA")
        print(f"    Range: ₹{result['range_low']} - "
              f"₹{result['range_high']} LPA")

    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"\n🧹 Cleaned up: {save_path}")


if __name__ == "__main__":
    df = create_dataset()
    print(f"Dataset: {df.shape}\n")

    train_and_save_pipeline(df)
    load_and_predict()
    simulate_api_endpoint()
