"""
Day 39 — Pandas: Data Cleaning
Topic: Handling Missing Values
Date: 26 June 2026
Author: Bala Ravi

Missing values = the most common data problem!
Every real dataset has them.
Handling them correctly = better ML models!
"""
import pandas as pd
import numpy as np


def create_messy_dataset() -> pd.DataFrame:
    """Create a realistic messy job market dataset."""
    np.random.seed(42)

    data = {
        'name': ['Bala', 'Ravi', None, 'Priya',
                 'Kumar', 'Hari', None, 'Meera'],
        'age': [22, 23, None, 25, 21, None, 28, 24],
        'salary_lpa': [12.0, 18.0, None, 25.0,
                       None, 15.0, 22.0, None],
        'city': ['Bangalore', None, 'Mumbai',
                 'Bangalore', 'Hyderabad', None,
                 'Delhi', 'Chennai'],
        'experience': [2, 3, None, 5, 1, 4, None, 3],
        'job_title': ['Data Scientist', 'ML Engineer',
                      'Data Analyst', None,
                      'Python Dev', 'Data Engineer',
                      'AI Engineer', None]
    }
    return pd.DataFrame(data)


def detect_missing(df: pd.DataFrame) -> None:
    """Detect and summarize missing values."""
    print("=== Missing Value Detection ===\n")

    print("Missing count per column:")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() /
                   len(df) * 100).round(1)

    summary = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })
    print(summary[summary['Missing Count'] > 0])

    print(f"\nTotal missing values: "
          f"{df.isnull().sum().sum()}")
    print(f"Total cells: {df.size}")
    print(f"Overall missing %: "
          f"{df.isnull().sum().sum()/df.size*100:.1f}%")


def handle_missing_strategies(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply different strategies for different columns.

    ML Strategy:
    - Numerical → fill with median (robust to outliers)
    - Categorical → fill with mode or 'Unknown'
    - Too many missing → drop column
    """
    print("\n=== Handling Missing Values ===\n")
    df = df.copy()

    print("Before cleaning:")
    print(f"Shape: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()}")

    # Numerical — fill with median
    df['age'] = df['age'].fillna(
        df['age'].median())
    df['salary_lpa'] = df['salary_lpa'].fillna(
        df['salary_lpa'].median())
    df['experience'] = df['experience'].fillna(
        df['experience'].median())

    # Categorical — fill with mode or Unknown
    df['name'] = df['name'].fillna('Unknown')
    df['city'] = df['city'].fillna(
        df['city'].mode()[0])
    df['job_title'] = df['job_title'].fillna('Not Specified')

    print("\nAfter cleaning:")
    print(f"Shape: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()}")
    print("\nCleaned DataFrame:")
    print(df)

    return df


def dropna_strategies(df: pd.DataFrame) -> None:
    """Show different dropna strategies."""
    print("\n=== dropna Strategies ===\n")

    print(f"Original rows: {len(df)}")

    # Drop rows with any missing
    df1 = df.dropna()
    print(f"Drop any missing: {len(df1)} rows left")

    # Drop rows where specific column is missing
    df2 = df.dropna(subset=['salary_lpa'])
    print(f"Drop if salary missing: "
          f"{len(df2)} rows left")

    # Drop columns with too many missing
    df3 = df.dropna(axis=1, thresh=len(df)*0.7)
    print(f"Drop cols <70% complete: "
          f"{df3.columns.tolist()}")


if __name__ == "__main__":
    df = create_messy_dataset()

    print("Original messy dataset:")
    print(df)
    print()

    detect_missing(df)
    df_clean = handle_missing_strategies(df)
    dropna_strategies(df)
