"""
Day 40 — Pandas: GroupBy & Aggregations
Topic: Transform and Pivot Tables
Date: 27 June 2026
Author: Bala Ravi

transform() = same shape as original DataFrame
pivot_table() = Excel-style cross-tabulation
"""
import pandas as pd
import numpy as np


def create_dataset() -> pd.DataFrame:
    """Create job dataset."""
    np.random.seed(42)
    n = 80

    return pd.DataFrame({
        'job_title': np.random.choice(
            ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer'], n),
        'city': np.random.choice(
            ['Bangalore', 'Hyderabad',
             'Mumbai', 'Delhi'], n),
        'salary_lpa': np.round(
            np.random.normal(20, 7, n), 1),
        'experience': np.random.randint(0, 10, n),
        'skills_count': np.random.randint(3, 10, n)
    })


def transform_demo(df: pd.DataFrame) -> None:
    """Demonstrate GroupBy transform."""
    print("=== GroupBy Transform ===\n")

    df = df.copy()

    # Add city average salary to each row
    df['city_avg_salary'] = (
        df.groupby('city')['salary_lpa']
        .transform('mean')
        .round(1))

    # Salary relative to city average
    df['salary_vs_city_avg'] = (
        df['salary_lpa'] - df['city_avg_salary']).round(1)

    # Z-score within each city
    df['city_z_score'] = (
        df.groupby('city')['salary_lpa']
        .transform(lambda x: (x - x.mean()) / x.std())
        .round(2))

    print("Sample rows with transform features:")
    print(df[['job_title', 'city', 'salary_lpa',
              'city_avg_salary', 'salary_vs_city_avg',
              'city_z_score']].head(8).to_string(
        index=False))

    # Find above-average earners per city
    above_avg = df[df['salary_vs_city_avg'] > 0]
    print(f"\nAbove city average: {len(above_avg)} "
          f"({len(above_avg)/len(df)*100:.0f}%)")

    # Flag outliers (z_score > 2)
    outliers = df[df['city_z_score'].abs() > 2]
    print(f"Salary outliers: {len(outliers)}")


def pivot_table_demo(df: pd.DataFrame) -> None:
    """Demonstrate pivot tables."""
    print("\n=== Pivot Tables ===\n")

    # Average salary: city vs job title
    pivot = df.pivot_table(
        values='salary_lpa',
        index='city',
        columns='job_title',
        aggfunc='mean',
        fill_value=0
    ).round(1)

    print("Average Salary (₹ LPA) by City × Role:")
    print(pivot.to_string())

    # Count pivot
    count_pivot = df.pivot_table(
        values='salary_lpa',
        index='city',
        columns='job_title',
        aggfunc='count',
        fill_value=0
    )

    print("\nJob Count by City × Role:")
    print(count_pivot.to_string())


def feature_engineering_demo(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering using GroupBy transform.
    This is used in Kaggle competitions!

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with engineered features
    """
    print("\n=== ML Feature Engineering ===\n")

    df = df.copy()

    # Group-level features
    for col in ['city', 'job_title']:
        df[f'{col}_avg_salary'] = (
            df.groupby(col)['salary_lpa']
            .transform('mean').round(1))
        df[f'{col}_job_count'] = (
            df.groupby(col)['salary_lpa']
            .transform('count'))

    # Interaction features
    df['exp_salary_ratio'] = (
        df['salary_lpa'] /
        (df['experience'] + 1)).round(2)

    print("Engineered features added:")
    new_cols = ['city_avg_salary', 'city_job_count',
                'job_title_avg_salary',
                'job_title_job_count',
                'exp_salary_ratio']

    print(df[new_cols].describe().round(1))
    print(f"\nOriginal features: 5")
    print(f"After engineering: {len(df.columns)}")
    print("These features improve ML model accuracy! 🔥")

    return df


if __name__ == "__main__":
    df = create_dataset()
    transform_demo(df)
    pivot_table_demo(df)
    df_engineered = feature_engineering_demo(df)
