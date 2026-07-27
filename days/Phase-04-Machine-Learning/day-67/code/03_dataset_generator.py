"""
Day 67 — Autonomous Data Scientist
Topic: Sample Dataset Generator for Testing
Date: 24 July 2026
Author: Bala Ravi

Generate various datasets to test AutoDS system.
Employee churn, house prices, customer segmentation.
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


def generate_employee_churn(
        n: int = 1000) -> pd.DataFrame:
    """
    Employee churn dataset.
    Classification: predict who will quit.

    Args:
        n: Number of rows

    Returns:
        DataFrame with churn label
    """
    np.random.seed(42)

    age = np.random.normal(35, 8, n).clip(22, 60)
    years_exp = np.random.exponential(5, n).clip(0, 30)
    salary = np.clip(
        300000 + years_exp * 50000 +
        np.random.normal(0, 80000, n), 200000, 3000000)
    dept = np.random.choice(
        ['Engineering', 'Sales', 'HR',
         'Marketing', 'Finance', 'Operations'], n)
    performance = np.random.uniform(1, 5, n)
    satisfaction = np.random.uniform(1, 5, n)
    projects = np.random.randint(1, 10, n)
    overtime = np.random.choice([0, 1], n, p=[0.6, 0.4])
    distance_km = np.random.exponential(15, n).clip(1, 80)

    # Churn probability based on features
    churn_prob = (
        0.1 +
        0.1 * (salary < 400000) +
        0.15 * (satisfaction < 2.5) +
        0.1 * (performance < 2.0) +
        0.1 * overtime +
        0.05 * (distance_km > 40) +
        0.1 * (years_exp > 8))
    churn_prob = np.clip(churn_prob, 0, 0.8)
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        'age': age.round(0).astype(int),
        'years_experience': years_exp.round(1),
        'salary': salary.round(0).astype(int),
        'department': dept,
        'performance_score': performance.round(2),
        'satisfaction_score': satisfaction.round(2),
        'num_projects': projects,
        'overtime': overtime,
        'distance_km': distance_km.round(1),
        'churn': churn
    })

    # Add missing values (realistic)
    df.loc[
        np.random.choice(n, int(n * 0.03),
                         replace=False),
        'satisfaction_score'] = np.nan
    df.loc[
        np.random.choice(n, int(n * 0.02),
                         replace=False),
        'distance_km'] = np.nan

    return df


def generate_house_prices(
        n: int = 800) -> pd.DataFrame:
    """
    House price dataset.
    Regression: predict sale price.
    """
    np.random.seed(42)

    area_sqft = np.random.normal(1500, 500, n).clip(
        500, 5000)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.randint(1, 4, n)
    age_years = np.random.exponential(15, n).clip(0, 80)
    city = np.random.choice(
        ['Bangalore', 'Mumbai', 'Delhi',
         'Hyderabad', 'Pune'], n)
    locality_type = np.random.choice(
        ['Premium', 'Standard', 'Economy'], n,
        p=[0.2, 0.5, 0.3])
    floors = np.random.randint(1, 25, n)
    parking = np.random.choice([0, 1], n, p=[0.3, 0.7])
    gym = np.random.choice([0, 1], n, p=[0.5, 0.5])

    city_mult = {
        'Mumbai': 2.5, 'Delhi': 2.0,
        'Bangalore': 1.8, 'Hyderabad': 1.4,
        'Pune': 1.2}
    locality_mult = {
        'Premium': 1.8, 'Standard': 1.0,
        'Economy': 0.6}

    price = (
        area_sqft * 8000 *
        np.array([city_mult[c] for c in city]) *
        np.array([locality_mult[l]
                  for l in locality_type]) +
        bedrooms * 500000 +
        bathrooms * 300000 -
        age_years * 50000 +
        parking * 400000 +
        gym * 300000 +
        np.random.normal(0, 1000000, n))

    price = np.clip(price, 2000000, 50000000)

    df = pd.DataFrame({
        'area_sqft': area_sqft.round(0).astype(int),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age_years.round(0).astype(int),
        'city': city,
        'locality_type': locality_type,
        'floor': floors,
        'parking': parking,
        'gym': gym,
        'price': price.round(0).astype(int)
    })

    # Add missing values
    df.loc[
        np.random.choice(n, int(n * 0.04),
                         replace=False),
        'age_years'] = np.nan

    return df


def generate_loan_default(
        n: int = 1200) -> pd.DataFrame:
    """
    Loan default dataset.
    Binary classification with class imbalance.
    """
    np.random.seed(42)

    income = np.random.lognormal(12, 0.5, n)
    loan_amount = np.random.lognormal(13, 0.6, n)
    loan_term = np.random.choice(
        [12, 24, 36, 48, 60], n)
    credit_score = np.random.normal(700, 80, n).clip(
        300, 850)
    debt_to_income = np.random.beta(2, 5, n)
    employment_type = np.random.choice(
        ['Salaried', 'Self-employed',
         'Business', 'Freelance'], n,
        p=[0.5, 0.25, 0.15, 0.1])
    num_dependents = np.random.randint(0, 5, n)
    prev_defaults = np.random.choice(
        [0, 1, 2], n, p=[0.7, 0.2, 0.1])

    default_prob = (
        0.05 +
        0.15 * (credit_score < 600) +
        0.1 * (debt_to_income > 0.5) +
        0.1 * (prev_defaults > 0) +
        0.05 * (employment_type == 'Freelance') +
        0.05 * (loan_amount / income > 5))
    default_prob = np.clip(default_prob, 0, 0.7)
    default = np.random.binomial(1, default_prob)

    df = pd.DataFrame({
        'annual_income': income.round(0).astype(int),
        'loan_amount': loan_amount.round(0).astype(int),
        'loan_term_months': loan_term,
        'credit_score': credit_score.round(0).astype(int),
        'debt_to_income_ratio': debt_to_income.round(3),
        'employment_type': employment_type,
        'num_dependents': num_dependents,
        'previous_defaults': prev_defaults,
        'default': default
    })

    # Missing values
    df.loc[
        np.random.choice(n, int(n * 0.05),
                         replace=False),
        'credit_score'] = np.nan

    return df


def save_all_datasets(
        save_dir: str = "projects/autonomous_data_scientist/data"
) -> None:
    """Generate and save all sample datasets."""
    os.makedirs(save_dir, exist_ok=True)

    datasets = {
        'employee_churn.csv': (
            generate_employee_churn(), 'churn'),
        'house_prices.csv': (
            generate_house_prices(), 'price'),
        'loan_default.csv': (
            generate_loan_default(), 'default')
    }

    print("Generating sample datasets...\n")
    for filename, (df, target) in datasets.items():
        path = os.path.join(save_dir, filename)
        df.to_csv(path, index=False)
        print(f"✅ {filename}")
        print(f"   Shape: {df.shape}")
        print(f"   Target: '{target}'")
        print(f"   Task: "
              f"{'classification' if df[target].nunique() <= 10 else 'regression'}")
        print()

    print(f"All datasets saved to {save_dir}/")


if __name__ == "__main__":
    save_all_datasets()

    # Quick demo
    df = generate_employee_churn()
    print("\nEmployee Churn Dataset Preview:")
    print(df.head(3).to_string())
    print(f"\nChurn rate: {df['churn'].mean()*100:.1f}%")
