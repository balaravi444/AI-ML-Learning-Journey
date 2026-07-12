"""
Day 39 — Pandas: Data Cleaning
Topic: Outlier Detection and String Cleaning
Date: 26 June 2026
Author: Bala Ravi

Outliers can destroy ML model performance!
Messy strings cause wrong groupings and joins!
"""
import pandas as pd
import numpy as np


def outlier_detection_demo() -> None:
    """Detect and handle outliers using IQR method."""
    print("=== Outlier Detection (IQR Method) ===\n")

    np.random.seed(42)

    # Realistic salary data with outliers
    salaries = np.concatenate([
        np.random.normal(25, 8, 95),  # normal salaries
        [150, 200, -5, 0.5, 180]      # outliers!
    ])
    df = pd.DataFrame({'salary_lpa': salaries})
    df['salary_lpa'] = df['salary_lpa'].round(1)

    print(f"Dataset size: {len(df)}")
    print(f"Salary stats before cleaning:")
    print(df['salary_lpa'].describe())

    # IQR method
    Q1 = df['salary_lpa'].quantile(0.25)
    Q3 = df['salary_lpa'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print(f"\nIQR Analysis:")
    print(f"  Q1: {Q1:.1f}")
    print(f"  Q3: {Q3:.1f}")
    print(f"  IQR: {IQR:.1f}")
    print(f"  Lower bound: {lower:.1f}")
    print(f"  Upper bound: {upper:.1f}")

    outliers = df[(df['salary_lpa'] < lower) |
                  (df['salary_lpa'] > upper)]
    print(f"\nOutliers detected: {len(outliers)}")
    print(f"Outlier values: "
          f"{sorted(outliers['salary_lpa'].tolist())}")

    # Strategy 1 — Remove outliers
    df_removed = df[(df['salary_lpa'] >= lower) &
                    (df['salary_lpa'] <= upper)]
    print(f"\nStrategy 1 — Remove outliers:")
    print(f"  Remaining: {len(df_removed)} rows")

    # Strategy 2 — Cap outliers (Winsorization)
    df['salary_capped'] = df['salary_lpa'].clip(
        lower, upper)
    print(f"\nStrategy 2 — Cap outliers:")
    print(f"  Max after capping: "
          f"{df['salary_capped'].max():.1f}")
    print(f"  Min after capping: "
          f"{df['salary_capped'].min():.1f}")


def string_cleaning_demo() -> None:
    """Clean messy string data."""
    print("\n=== String Cleaning ===\n")

    data = {
        'name': ['  bala ravi  ', 'RAVI KUMAR',
                 'priya sharma', '  Kumar S.  '],
        'city': ['bangalore', 'MUMBAI',
                 'Hyderabad ', ' delhi'],
        'email': ['Bala@Gmail.COM', 'ravi@yahoo.com',
                  'PRIYA@OUTLOOK.COM', 'kumar@gmail.com'],
        'phone': ['+91-9876543210', '9876543211',
                  '+919876543212', '98-765-43213'],
        'skills': ['Python, ML, Deep Learning',
                   'python,ml,nlp',
                   'PYTHON; SQL; TABLEAU',
                   'Python | R | Statistics']
    }
    df = pd.DataFrame(data)

    print("Messy data:")
    print(df)

    # Clean names
    df['name'] = (df['name']
                  .str.strip()
                  .str.title())

    # Clean cities
    df['city'] = (df['city']
                  .str.strip()
                  .str.title())

    # Normalize emails
    df['email'] = df['email'].str.lower()

    # Normalize phone numbers
    df['phone'] = (df['phone']
                   .str.replace(r'[^0-9]', '',
                                 regex=True)
                   .str[-10:])  # keep last 10 digits

    # Extract first skill
    df['primary_skill'] = (df['skills']
                           .str.split(r'[,;|]')
                           .str[0]
                           .str.strip()
                           .str.title())

    print("\nCleaned data:")
    print(df[['name', 'city', 'email',
              'phone', 'primary_skill']])


def complete_cleaning_pipeline() -> pd.DataFrame:
    """
    Complete data cleaning pipeline.
    Ready to plug into any ML project!
    """
    print("\n=== Complete Cleaning Pipeline ===\n")

    np.random.seed(42)
    n = 20

    df = pd.DataFrame({
        'name': ['Bala', 'Ravi', None,
                 'Kumar', 'Bala'] + [f'Person{i}'
                 for i in range(15)],
        'city': (['bangalore', 'MUMBAI', 'delhi',
                  None, 'bangalore'] +
                  ['Hyderabad'] * 15),
        'salary': (['35000', '45000', '28000',
                    'NA', '35000'] +
                    [str(s) for s in
                     np.random.randint(
                         10000, 80000, 15)]),
        'experience': ([2, 3, None, 1, 2] +
                        list(np.random.randint(0, 10, 15)))
    })

    print(f"Before cleaning: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()}")

    df = df.copy()

    # Step 1 — Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Step 2 — Fix data types
    df['salary'] = pd.to_numeric(
        df['salary'], errors='coerce')

    # Step 3 — Handle missing values
    df['salary'].fillna(
        df['salary'].median(), inplace=True)
    df['experience'].fillna(
        df['experience'].median(), inplace=True)
    df['name'].fillna('Unknown', inplace=True)
    df['city'].fillna('Unknown', inplace=True)

    # Step 4 — Clean strings
    df['city'] = df['city'].str.strip().str.title()

    # Step 5 — Remove outliers from salary
    Q1 = df['salary'].quantile(0.25)
    Q3 = df['salary'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['salary'] >= Q1 - 1.5 * IQR) &
            (df['salary'] <= Q3 + 1.5 * IQR)]

    # Step 6 — Reset index
    df = df.reset_index(drop=True)

    print(f"\nAfter cleaning: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()}")
    print("\nClean data ready for ML!")
    print(df.head())

    return df


if __name__ == "__main__":
    outlier_detection_demo()
    string_cleaning_demo()
    df_clean = complete_cleaning_pipeline()
