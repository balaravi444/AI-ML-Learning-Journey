"""
Day 44 — Feature Engineering
Topic: Complete ML-Ready Feature Pipeline
Date: 01 July 2026
Author: Bala Ravi

The complete feature engineering pipeline
that feeds into the ML model!

This is EXACTLY how production ML systems
prepare data before training!
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score)


def create_raw_dataset(n: int = 500) -> pd.DataFrame:
    """Create comprehensive job market dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']

    experience = np.random.randint(0, 12, n)
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

    return pd.DataFrame({
        'job_title': role,
        'city': city,
        'salary_lpa': salary,
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1),
        'remote': np.random.choice(
            [True, False], n, p=[0.3, 0.7])
    })


def engineer_features(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.

    Args:
        df: Raw DataFrame

    Returns:
        Engineered DataFrame ready for ML
    """
    df = df.copy()

    # 1. New numerical features
    df['exp_x_skills'] = (df['experience_years'] *
                          df['skills_count'])
    df['salary_per_exp'] = (
        df['salary_lpa'] /
        (df['experience_years'] + 1))
    df['exp_rating'] = (df['experience_years'] *
                        df['rating'])

    # 2. Log transform
    df['salary_log'] = np.log1p(df['salary_lpa'])

    # 3. Binning
    df['exp_level_num'] = pd.cut(
        df['experience_years'],
        bins=[0, 2, 5, 8, 12],
        labels=[0, 1, 2, 3],
        include_lowest=True).astype(int)

    # 4. Target encoding for city and role
    city_mean = (df.groupby('city')['salary_lpa']
                .mean())
    df['city_target'] = df['city'].map(city_mean)

    role_mean = (df.groupby('job_title')['salary_lpa']
                .mean())
    df['role_target'] = df['job_title'].map(role_mean)

    # 5. Boolean to int
    df['is_remote'] = df['remote'].astype(int)

    return df


def build_ml_pipeline(
        df: pd.DataFrame) -> dict:
    """
    Build complete sklearn ML pipeline.
    Handles preprocessing + model training!

    Args:
        df: Engineered DataFrame

    Returns:
        Results dictionary
    """
    print("=== Complete ML Pipeline ===\n")

    # Feature engineering
    df_eng = engineer_features(df)

    # Define features
    numerical_features = [
        'experience_years', 'skills_count',
        'rating', 'exp_x_skills',
        'exp_rating', 'exp_level_num',
        'city_target', 'role_target', 'is_remote'
    ]
    categorical_features = ['city', 'job_title']

    # Target
    X = df_eng[numerical_features + categorical_features]
    y = df_eng['salary_lpa']

    # Train/test split
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Features used:    {len(X.columns)}\n")

    # Build sklearn pipeline
    numerical_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer,
         numerical_features),
        ('cat', categorical_transformer,
         categorical_features)
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100,
            random_state=42))
    ])

    # Train
    print("Training Random Forest model...")
    full_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_train = full_pipeline.predict(X_train)
    y_pred_test = full_pipeline.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"\n📊 Model Performance:")
    print(f"  Train R²:    {train_r2:.4f}")
    print(f"  Test R²:     {test_r2:.4f}")
    print(f"  Test MAE:    ₹{test_mae:.2f} LPA")
    print(f"\n  R² = {test_r2:.2f} means the model")
    print(f"  explains {test_r2*100:.0f}% of "
          f"salary variance!")

    # Sample predictions
    sample_X = X_test.head(5)
    sample_y = y_test.head(5)
    sample_pred = full_pipeline.predict(sample_X)

    print(f"\n🎯 Sample Predictions:")
    print(f"  {'Actual':>10} | {'Predicted':>10} | "
          f"{'Error':>8}")
    print(f"  {'-'*35}")
    for actual, pred in zip(
            sample_y.values, sample_pred):
        error = abs(actual - pred)
        print(f"  ₹{actual:>8.1f}L | "
              f"₹{pred:>8.1f}L | "
              f"₹{error:>6.1f}L")

    return {
        'pipeline': full_pipeline,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae
    }


def feature_importance_analysis(
        df: pd.DataFrame,
        pipeline: Pipeline) -> None:
    """Show which features matter most."""
    print("\n=== Feature Importance ===\n")

    df_eng = engineer_features(df)

    numerical_features = [
        'experience_years', 'skills_count',
        'rating', 'exp_x_skills',
        'exp_rating', 'exp_level_num',
        'city_target', 'role_target', 'is_remote'
    ]

    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    ohe_features = (preprocessor
                   .named_transformers_['cat']
                   .named_steps['onehot']
                   .get_feature_names_out(
                       ['city', 'job_title']))

    all_features = numerical_features + list(
        ohe_features)
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("Top 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f"  {row['feature']:<25}: "
              f"{row['importance']:.4f} {bar}")

    print(f"\n💡 Key insight: "
          f"'{importance_df.iloc[0]['feature']}' "
          f"is most important for predicting salary!")


if __name__ == "__main__":
    df = create_raw_dataset()
    print(f"Raw dataset: {df.shape}\n")

    results = build_ml_pipeline(df)
    feature_importance_analysis(
        df, results['pipeline'])

    print(f"\n✅ Complete ML pipeline built!")
    print(f"   This is Day 47 Job Market "
          f"Analyzer — preview! 🚀")
