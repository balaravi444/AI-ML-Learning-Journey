"""
Day 38 — Pandas: DataFrames
Topic: Advanced Filtering and Selection
Date: 25 June 2026
Author: Bala Ravi

Filtering is the most used Pandas operation!
Boolean indexing, query(), isin(), between()...
"""
import pandas as pd
import numpy as np


def create_job_dataframe() -> pd.DataFrame:
    """Create a richer job market dataset."""
    np.random.seed(42)
    n = 50

    companies = ['TCS', 'Infosys', 'Wipro',
                 'Amazon', 'Google', 'Microsoft',
                 'Flipkart', 'Swiggy', 'Zomato',
                 'Paytm']
    locations = ['Bangalore', 'Hyderabad', 'Pune',
                 'Mumbai', 'Delhi', 'Chennai']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Python Developer', 'Data Engineer']

    return pd.DataFrame({
        'job_title': np.random.choice(roles, n),
        'company': np.random.choice(companies, n),
        'location': np.random.choice(locations, n),
        'salary_lpa': np.random.randint(6, 45, n),
        'experience': np.random.randint(0, 10, n),
        'remote': np.random.choice(
            [True, False], n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1)
    })


def filtering_demo(df: pd.DataFrame) -> None:
    """Demonstrate all filtering methods."""
    print("=== Advanced Filtering ===\n")
    print(f"Total jobs: {len(df)}\n")

    # 1. Single condition
    high_pay = df[df['salary_lpa'] >= 25]
    print(f"High paying (25+ LPA): {len(high_pay)}")

    # 2. Multiple conditions
    senior_ds = df[
        (df['job_title'] == 'Data Scientist') &
        (df['experience'] >= 3)
    ]
    print(f"Senior Data Scientists: {len(senior_ds)}")

    # 3. OR condition
    top_cities = df[
        (df['location'] == 'Bangalore') |
        (df['location'] == 'Hyderabad')
    ]
    print(f"Bangalore or Hyderabad: {len(top_cities)}")

    # 4. isin() — multiple values
    big_tech = df[df['company'].isin(
        ['Google', 'Microsoft', 'Amazon'])]
    print(f"Big Tech companies: {len(big_tech)}")

    # 5. between()
    mid_salary = df[
        df['salary_lpa'].between(15, 30)]
    print(f"Mid-range salary (15-30 LPA): "
          f"{len(mid_salary)}")

    # 6. str methods
    ml_jobs = df[
        df['job_title'].str.contains('ML|AI',
                                      case=False,
                                      regex=True)]
    print(f"ML/AI jobs: {len(ml_jobs)}")

    # 7. Remote jobs
    remote_high = df[
        df['remote'] & (df['salary_lpa'] >= 20)]
    print(f"Remote + 20+ LPA: {len(remote_high)}")

    # 8. query() method — SQL-like!
    result = df.query(
        "salary_lpa >= 20 and experience <= 3")
    print(f"\nquery() — high pay, low exp:")
    print(f"  {len(result)} jobs found")

    # 9. nlargest/nsmallest
    print(f"\nTop 5 highest paying jobs:")
    print(df.nlargest(5, 'salary_lpa')
          [['job_title', 'company',
            'salary_lpa', 'location']])


def ml_data_preparation(
        df: pd.DataFrame) -> tuple:
    """
    Prepare data for ML model.
    This is EXACTLY how Kaggle competitors
    prepare data for submission!

    Args:
        df: Raw DataFrame

    Returns:
        X (features) and y (target) ready for ML
    """
    print("\n=== ML Data Preparation ===\n")
    print(f"Original shape: {df.shape}")

    # Select features
    X = df[['experience', 'remote', 'rating']].copy()
    y = df['salary_lpa']

    # Convert boolean to int
    X['remote'] = X['remote'].astype(int)

    # Add engineered feature
    X['exp_rating'] = X['experience'] * X['rating']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature preview:")
    print(X.head())
    print(f"\nTarget preview:")
    print(y.head())
    print(f"\nTarget stats:")
    print(y.describe())

    return X, y


if __name__ == "__main__":
    df = create_job_dataframe()
    print(f"Job Market Dataset: {df.shape}\n")
    print(df.head())

    filtering_demo(df)
    X, y = ml_data_preparation(df)

    print("\n✅ Data ready for ML model!")
    print("Next step: sklearn.fit(X, y) — Day 51!")
