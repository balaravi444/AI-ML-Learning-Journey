"""
Day 39 — Pandas: Data Cleaning
Topic: Duplicates and Data Types
Date: 26 June 2026
Author: Bala Ravi

Wrong data types = broken ML models!
Duplicates = biased training data!
"""
import pandas as pd
import numpy as np


def duplicates_demo() -> None:
    """Handle duplicate records."""
    print("=== Handling Duplicates ===\n")

    data = {
        'name': ['Bala', 'Ravi', 'Bala',
                 'Kumar', 'Ravi', 'Priya'],
        'email': ['bala@email.com', 'ravi@email.com',
                  'bala@email.com', 'kumar@email.com',
                  'ravi@email.com', 'priya@email.com'],
        'salary': [35000, 45000, 35000,
                   28000, 50000, 55000]
    }
    df = pd.DataFrame(data)

    print(f"Original records: {len(df)}")
    print(df)

    print(f"\nDuplicate rows: "
          f"{df.duplicated().sum()}")
    print(f"Duplicate emails: "
          f"{df.duplicated(subset=['email']).sum()}")

    # Remove duplicates
    df_clean = df.drop_duplicates()
    print(f"\nAfter removing duplicates: "
          f"{len(df_clean)} records")
    print(df_clean)

    # Keep last occurrence
    df_last = df.drop_duplicates(
        subset=['email'], keep='last')
    print(f"\nKeep last (for salary update): "
          f"{len(df_last)} records")
    print(df_last)


def data_types_demo() -> None:
    """Fix wrong data types."""
    print("\n=== Fixing Data Types ===\n")

    data = {
        'name': ['Bala', 'Ravi', 'Kumar'],
        'salary': ['35,000', '45,000', '28,000'],
        'join_date': ['2024-01-15', '2023-06-20',
                      '2025-03-01'],
        'is_remote': ['True', 'False', 'True'],
        'experience': ['2 years', '3 years', '1 year']
    }
    df = pd.DataFrame(data)

    print("Before fixing types:")
    print(df.dtypes)
    print()

    # Fix salary (string → int)
    df['salary'] = (df['salary']
                    .str.replace(',', '')
                    .astype(int))

    # Fix date (string → datetime)
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['year_joined'] = df['join_date'].dt.year
    df['months_employed'] = ((pd.Timestamp.now() -
                               df['join_date'])
                              .dt.days // 30)

    # Fix boolean
    df['is_remote'] = df['is_remote'].map(
        {'True': True, 'False': False})

    # Fix experience (extract number)
    df['exp_years'] = (df['experience']
                       .str.extract(r'(\d+)')
                       .astype(int))

    print("After fixing types:")
    print(df.dtypes)
    print()
    print(df)


def encoding_demo() -> None:
    """
    Encode categorical variables for ML.
    This is EXACTLY what ML preprocessing does!
    """
    print("\n=== Encoding for ML ===\n")

    df = pd.DataFrame({
        'name': ['Bala', 'Ravi', 'Kumar', 'Priya'],
        'city': ['Bangalore', 'Mumbai',
                 'Bangalore', 'Delhi'],
        'experience': [2, 3, 1, 5],
        'salary': [35, 45, 28, 60]
    })

    print("Original:")
    print(df)

    # Label encoding (ordinal)
    city_map = {'Bangalore': 0, 'Mumbai': 1, 'Delhi': 2}
    df['city_label'] = df['city'].map(city_map)

    print("\nLabel encoded:")
    print(df[['city', 'city_label']])

    # One-hot encoding (nominal) — for ML!
    df_encoded = pd.get_dummies(
        df, columns=['city'], drop_first=True)
    print("\nOne-hot encoded (ML ready!):")
    print(df_encoded)

    print("\nML features:")
    X = df_encoded.drop(['name', 'salary'], axis=1)
    y = df_encoded['salary']
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("Ready for model.fit(X, y)! 🚀")


if __name__ == "__main__":
    duplicates_demo()
    data_types_demo()
    encoding_demo()
