"""
Day 38 — Pandas: DataFrames
Topic: Reading and Writing Files
Date: 25 June 2026
Author: Bala Ravi

Pandas can read almost any file format!
CSV, JSON, Excel, SQL, HTML...

Real World Connection:
    Every Kaggle dataset = CSV file!
    read_csv() is the MOST USED Pandas function!
"""
import pandas as pd
import json
import os


def create_sample_data() -> None:
    """Create sample CSV and JSON files for practice."""

    # Create sample Indian job market data
    jobs_data = {
        'job_title': [
            'Data Scientist', 'ML Engineer',
            'Data Analyst', 'AI Researcher',
            'Python Developer', 'Data Engineer',
            'MLOps Engineer', 'NLP Engineer',
            'Computer Vision Engineer',
            'Business Analyst'
        ],
        'company': [
            'TCS', 'Infosys', 'Wipro',
            'Amazon', 'Google', 'Microsoft',
            'Flipkart', 'Zomato', 'Swiggy', 'HDFC'
        ],
        'location': [
            'Bangalore', 'Hyderabad', 'Pune',
            'Bangalore', 'Hyderabad', 'Bangalore',
            'Bangalore', 'Mumbai', 'Delhi', 'Mumbai'
        ],
        'salary_lpa': [
            12, 18, 8, 25, 35, 15,
            20, 22, 24, 10
        ],
        'experience_years': [
            2, 3, 1, 5, 7, 4, 4, 3, 3, 2
        ],
        'skills': [
            'Python,ML,Statistics',
            'Python,TensorFlow,MLOps',
            'SQL,Excel,Tableau',
            'Python,Research,NLP',
            'Python,TensorFlow,ML',
            'Python,Spark,SQL',
            'Python,Docker,Kubernetes',
            'Python,NLP,Transformers',
            'Python,OpenCV,Deep Learning',
            'SQL,Excel,Power BI'
        ]
    }

    df = pd.DataFrame(jobs_data)
    df.to_csv("indian_jobs.csv", index=False)
    df.to_json("indian_jobs.json",
               orient='records', indent=2)

    print("✅ Created: indian_jobs.csv")
    print("✅ Created: indian_jobs.json")
    return df


def read_csv_demo() -> pd.DataFrame:
    """Demonstrate reading CSV files."""
    print("\n=== Reading CSV Files ===\n")

    df = pd.read_csv("indian_jobs.csv")

    print(f"Shape: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nInfo:")
    df.info()
    print(f"\nNull values:")
    print(df.isnull().sum())

    return df


def read_json_demo() -> None:
    """Demonstrate reading JSON files."""
    print("\n=== Reading JSON Files ===\n")

    df = pd.read_json("indian_jobs.json")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))


def save_filtered_data(df: pd.DataFrame) -> None:
    """Demonstrate saving filtered data."""
    print("\n=== Saving Data ===\n")

    # Filter high-salary jobs
    high_salary = df[df['salary_lpa'] >= 20]

    high_salary.to_csv(
        "high_salary_jobs.csv", index=False)
    print(f"Saved {len(high_salary)} high-salary "
          f"jobs to high_salary_jobs.csv")

    # Filter Bangalore jobs
    blr_jobs = df[df['location'] == 'Bangalore']
    blr_jobs.to_json(
        "bangalore_jobs.json",
        orient='records', indent=2)
    print(f"Saved {len(blr_jobs)} Bangalore "
          f"jobs to bangalore_jobs.json")


def explore_job_market(df: pd.DataFrame) -> None:
    """Quick exploration of job market data."""
    print("\n=== Indian Job Market Quick Look ===\n")

    print(f"📊 Total job postings: {len(df)}")
    print(f"💰 Avg salary: "
          f"₹{df['salary_lpa'].mean():.1f} LPA")
    print(f"🏆 Max salary: "
          f"₹{df['salary_lpa'].max()} LPA")

    print(f"\n📍 Jobs by location:")
    print(df['location'].value_counts())

    print(f"\n🏢 Jobs by company:")
    print(df['company'].value_counts())

    print(f"\n💼 Salary statistics:")
    print(df['salary_lpa'].describe())


if __name__ == "__main__":
    print("=== Pandas File I/O Demo ===\n")

    # Create sample data
    df = create_sample_data()

    # Read back
    df_loaded = read_csv_demo()
    read_json_demo()

    # Save filtered
    save_filtered_data(df_loaded)

    # Explore
    explore_job_market(df_loaded)

    # Cleanup
    for f in ["indian_jobs.csv", "indian_jobs.json",
              "high_salary_jobs.csv",
              "bangalore_jobs.json"]:
        if os.path.exists(f):
            os.remove(f)
