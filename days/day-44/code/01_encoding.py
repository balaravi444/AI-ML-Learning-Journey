"""
Day 44 — Feature Engineering
Topic: Encoding Categorical Variables
Date: 01 July 2026
Author: Bala Ravi

ML models need numbers — not text!
Encoding converts text categories to numbers.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder)


def create_job_dataset(n: int = 200) -> pd.DataFrame:
    """Create job market dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']
    levels = ['Junior', 'Mid', 'Senior', 'Expert']

    experience = np.random.randint(0, 12, n)
    city = np.random.choice(cities, n)
    role = np.random.choice(roles, n)
    level = np.random.choice(levels, n)

    city_bonus = {'Bangalore': 3, 'Mumbai': 2,
                  'Delhi': 1, 'Hyderabad': 0.5,
                  'Pune': 0}

    salary = np.clip(
        10 + experience * 1.8 +
        np.array([city_bonus[c] for c in city]) +
        np.random.normal(0, 2, n), 5, 55).round(1)

    return pd.DataFrame({
        'job_title': role,
        'city': city,
        'level': level,
        'salary_lpa': salary,
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n)
    })


def label_encoding_demo(df: pd.DataFrame) -> None:
    """
    Label Encoding — best for ORDINAL categories.
    Ordinal = has natural order (Junior < Mid < Senior)
    """
    print("=== Label Encoding (Ordinal) ===\n")

    df = df.copy()

    # Manual mapping for ordinal — controls order!
    level_map = {'Junior': 0, 'Mid': 1,
                 'Senior': 2, 'Expert': 3}
    df['level_encoded'] = df['level'].map(level_map)

    print("Level → Encoded mapping:")
    for level, code in level_map.items():
        print(f"  {level:<8} → {code}")

    print("\nSample data:")
    print(df[['level', 'level_encoded',
              'salary_lpa']].head(8))

    # Correlation check — ordinal encoding preserves order!
    corr = df['level_encoded'].corr(df['salary_lpa'])
    print(f"\nCorrelation level_encoded vs salary: "
          f"{corr:.3f}")
    print("Higher level → higher salary ✅")

    # Sklearn LabelEncoder (no order control)
    le = LabelEncoder()
    df['city_label'] = le.fit_transform(df['city'])
    print(f"\nSKLearn LabelEncoder for city:")
    print(f"Classes: {le.classes_}")
    print(f"Encoded: {le.transform(le.classes_)}")
    print("⚠️ No order control — use carefully!")


def one_hot_encoding_demo(df: pd.DataFrame) -> None:
    """
    One-Hot Encoding — best for NOMINAL categories.
    Nominal = no natural order (city, job_title)
    """
    print("\n=== One-Hot Encoding (Nominal) ===\n")

    df = df.copy()

    # Pandas get_dummies — simplest!
    df_encoded = pd.get_dummies(
        df[['city', 'salary_lpa', 'experience_years']],
        columns=['city'], drop_first=True)

    print("Original columns: ['city', 'salary_lpa', "
          "'experience_years']")
    print(f"After OHE columns: "
          f"{list(df_encoded.columns)}")

    print("\nSample encoded data:")
    print(df_encoded.head(5).to_string())

    print(f"\n✅ 5 cities → 4 binary columns")
    print(f"   drop_first=True prevents "
          f"multicollinearity!")


def target_encoding_demo(df: pd.DataFrame) -> None:
    """
    Target Encoding — best for HIGH CARDINALITY.
    Replace category with mean target value.
    """
    print("\n=== Target Encoding ===\n")

    df = df.copy()

    # City target encoding
    city_mean = (df.groupby('city')['salary_lpa']
                .mean()
                .round(2))
    df['city_target'] = df['city'].map(city_mean)

    # Role target encoding
    role_mean = (df.groupby('job_title')['salary_lpa']
                .mean()
                .round(2))
    df['role_target'] = df['job_title'].map(role_mean)

    print("City target encoding (mean salary):")
    for city, mean_sal in city_mean.sort_values(
            ascending=False).items():
        print(f"  {city:<12} → ₹{mean_sal} LPA")

    print("\nRole target encoding:")
    for role, mean_sal in role_mean.sort_values(
            ascending=False).items():
        print(f"  {role:<25} → ₹{mean_sal} LPA")

    # Correlation after encoding
    city_corr = df['city_target'].corr(
        df['salary_lpa'])
    role_corr = df['role_target'].corr(
        df['salary_lpa'])

    print(f"\nCorrelation after target encoding:")
    print(f"  city_target vs salary: {city_corr:.3f}")
    print(f"  role_target vs salary: {role_corr:.3f}")
    print(f"✅ High correlation → useful features!")


if __name__ == "__main__":
    df = create_job_dataset()
    print(f"Dataset: {df.shape}\n")

    label_encoding_demo(df)
    one_hot_encoding_demo(df)
    target_encoding_demo(df)
