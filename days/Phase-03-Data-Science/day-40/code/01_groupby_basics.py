"""
Day 40 — Pandas: GroupBy & Aggregations
Topic: GroupBy Basics
Date: 27 June 2026
Author: Bala Ravi

GroupBy = Split → Apply → Combine
Most powerful analysis tool in Pandas!
"""
import pandas as pd
import numpy as np


def create_job_dataset() -> pd.DataFrame:
    """Create rich job market dataset."""
    np.random.seed(42)
    n = 100

    cities = ['Bangalore', 'Hyderabad', 'Mumbai',
              'Delhi', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']
    companies = ['TCS', 'Infosys', 'Amazon',
                 'Google', 'Microsoft', 'Flipkart',
                 'Swiggy', 'Paytm']
    levels = ['Junior', 'Mid', 'Senior']

    return pd.DataFrame({
        'job_title': np.random.choice(roles, n),
        'city': np.random.choice(cities, n),
        'company': np.random.choice(companies, n),
        'salary_lpa': np.round(
            np.random.normal(20, 8, n), 1),
        'experience': np.random.randint(0, 12, n),
        'level': np.random.choice(levels, n),
        'remote': np.random.choice([True, False], n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1)
    })


def basic_groupby_demo(df: pd.DataFrame) -> None:
    """Demonstrate basic GroupBy operations."""
    print("=== Basic GroupBy ===\n")

    # Average salary by city
    print("Average Salary by City (₹ LPA):")
    city_salary = (df.groupby('city')['salary_lpa']
                   .mean()
                   .sort_values(ascending=False)
                   .round(1))
    for city, salary in city_salary.items():
        bar = '█' * int(salary // 2)
        print(f"  {city:<12}: ₹{salary:<5} LPA {bar}")

    # Job count by role
    print("\nJob Count by Role:")
    role_count = (df.groupby('job_title')
                  .size()
                  .sort_values(ascending=False))
    for role, count in role_count.items():
        print(f"  {role:<25}: {count} openings")

    # Multiple stats by city
    print("\nSalary Stats by City:")
    stats = (df.groupby('city')['salary_lpa']
             .agg(['mean', 'min', 'max', 'count'])
             .round(1))
    stats.columns = ['Avg', 'Min', 'Max', 'Count']
    print(stats.to_string())


def named_aggregations_demo(df: pd.DataFrame) -> None:
    """Show named aggregations — cleaner output."""
    print("\n=== Named Aggregations ===\n")

    insights = df.groupby('city').agg(
        avg_salary=('salary_lpa', 'mean'),
        max_salary=('salary_lpa', 'max'),
        min_salary=('salary_lpa', 'min'),
        total_jobs=('salary_lpa', 'count'),
        avg_experience=('experience', 'mean'),
        avg_rating=('rating', 'mean')
    ).round(1).reset_index()

    print("City-wise Job Market Insights:")
    print(insights.to_string(index=False))


def multi_column_groupby(df: pd.DataFrame) -> None:
    """GroupBy on multiple columns."""
    print("\n=== Multi-Column GroupBy ===\n")

    salary_by_role_city = (
        df.groupby(['city', 'job_title'])['salary_lpa']
        .mean()
        .round(1)
        .reset_index()
        .sort_values('salary_lpa', ascending=False)
    )

    print("Top 10 Role-City Combinations by Salary:")
    print(salary_by_role_city.head(10).to_string(
        index=False))

    # Remote vs onsite salary
    print("\nRemote vs Onsite Salary:")
    remote_stats = (
        df.groupby('remote')['salary_lpa']
        .agg(['mean', 'count'])
        .round(1)
    )
    remote_stats.index = ['Onsite', 'Remote']
    remote_stats.columns = ['Avg Salary', 'Count']
    print(remote_stats.to_string())


if __name__ == "__main__":
    df = create_job_dataset()
    print(f"Dataset: {df.shape[0]} jobs\n")

    basic_groupby_demo(df)
    named_aggregations_demo(df)
    multi_column_groupby(df)
