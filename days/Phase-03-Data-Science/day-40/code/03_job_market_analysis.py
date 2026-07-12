"""
Day 40 — Pandas: GroupBy & Aggregations
Topic: Complete Indian Job Market Analysis
Date: 27 June 2026
Author: Bala Ravi

This is the foundation of the Indian Job Market
Analyzer project coming on Day 47!
"""
import pandas as pd
import numpy as np


def create_comprehensive_dataset() -> pd.DataFrame:
    """Create comprehensive Indian job market data."""
    np.random.seed(42)
    n = 200

    cities = ['Bangalore', 'Hyderabad', 'Mumbai',
              'Delhi', 'Pune', 'Chennai']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer', 'MLOps Engineer',
             'NLP Engineer', 'Computer Vision Engineer']
    companies = ['TCS', 'Infosys', 'Wipro', 'Amazon',
                 'Google', 'Microsoft', 'Flipkart',
                 'Swiggy', 'Zomato', 'Paytm', 'HDFC',
                 'PhonePe', 'Razorpay', 'Meesho']
    skills_pool = [
        'Python', 'ML', 'SQL', 'TensorFlow',
        'PyTorch', 'NLP', 'Computer Vision',
        'Spark', 'AWS', 'Docker', 'MLOps',
        'Statistics', 'Deep Learning', 'LLMs'
    ]

    data = []
    for _ in range(n):
        exp = np.random.randint(0, 12)
        base_salary = 8 + exp * 1.8 + np.random.normal(0, 3)
        city_bonus = {'Bangalore': 2, 'Mumbai': 1.5,
                      'Delhi': 1, 'Hyderabad': 0.5,
                      'Pune': 0, 'Chennai': 0}
        city = np.random.choice(cities)
        salary = max(5, base_salary +
                     city_bonus.get(city, 0))

        n_skills = np.random.randint(2, 7)
        skills = ','.join(np.random.choice(
            skills_pool, n_skills, replace=False))

        data.append({
            'job_title': np.random.choice(roles),
            'company': np.random.choice(companies),
            'city': city,
            'salary_lpa': round(salary, 1),
            'experience_years': exp,
            'skills': skills,
            'remote': np.random.choice(
                [True, False],
                p=[0.3, 0.7]),
            'rating': round(
                np.random.uniform(3.0, 5.0), 1),
            'posted_days_ago': np.random.randint(1, 30)
        })

    return pd.DataFrame(data)


def analyze_salary_trends(df: pd.DataFrame) -> None:
    """Analyze salary trends using GroupBy."""
    print("=" * 55)
    print("  INDIAN AI/ML JOB MARKET ANALYSIS")
    print("=" * 55)

    print(f"\n📊 Dataset: {len(df)} job postings\n")

    print("💰 Salary by City (₹ LPA):")
    city_stats = df.groupby('city').agg(
        avg=('salary_lpa', 'mean'),
        median=('salary_lpa', 'median'),
        max=('salary_lpa', 'max'),
        jobs=('salary_lpa', 'count')
    ).round(1).sort_values('avg', ascending=False)

    for city, row in city_stats.iterrows():
        bar = '█' * int(row['avg'] // 2)
        print(f"  {city:<12}: ₹{row['avg']:<5} avg "
              f"| {int(row['jobs'])} jobs {bar}")

    print("\n💼 Salary by Role (₹ LPA):")
    role_stats = (
        df.groupby('job_title')['salary_lpa']
        .agg(['mean', 'count'])
        .round(1)
        .sort_values('mean', ascending=False)
    )
    role_stats.columns = ['Avg Salary', 'Count']
    print(role_stats.to_string())


def analyze_experience_salary(
        df: pd.DataFrame) -> None:
    """Analyze how salary changes with experience."""
    print("\n📈 Salary vs Experience:")

    # Create experience buckets
    df = df.copy()
    df['exp_bucket'] = pd.cut(
        df['experience_years'],
        bins=[0, 2, 5, 8, 12],
        labels=['Fresher (0-2)',
                'Mid (2-5)',
                'Senior (5-8)',
                'Expert (8+)'],
        include_lowest=True
    )

    exp_salary = (
        df.groupby('exp_bucket', observed=True)
        ['salary_lpa']
        .agg(['mean', 'count'])
        .round(1)
    )
    exp_salary.columns = ['Avg Salary (LPA)', 'Count']
    print(exp_salary.to_string())


def analyze_skills_demand(df: pd.DataFrame) -> None:
    """Analyze most demanded skills."""
    print("\n🔧 Most In-Demand Skills:")

    all_skills = []
    for skills_str in df['skills']:
        all_skills.extend(
            [s.strip() for s in skills_str.split(',')])

    skills_series = pd.Series(all_skills)
    top_skills = skills_series.value_counts().head(10)

    for skill, count in top_skills.items():
        pct = count / len(df) * 100
        bar = '█' * int(pct // 3)
        print(f"  {skill:<20}: {count:>3} jobs "
              f"({pct:.0f}%) {bar}")


def analyze_remote_trends(df: pd.DataFrame) -> None:
    """Analyze remote work trends."""
    print("\n🏠 Remote Work Analysis:")

    remote_by_city = (
        df.groupby('city')['remote']
        .agg(['sum', 'count'])
    )
    remote_by_city['remote_pct'] = (
        remote_by_city['sum'] /
        remote_by_city['count'] * 100
    ).round(1)

    for city, row in remote_by_city.iterrows():
        print(f"  {city:<12}: {row['remote_pct']}% remote")

    print("\n💰 Remote vs Onsite Salary:")
    remote_salary = (
        df.groupby('remote')['salary_lpa']
        .agg(['mean', 'count'])
        .round(1)
    )
    remote_salary.index = ['Onsite', 'Remote']
    remote_salary.columns = ['Avg Salary', 'Count']
    print(remote_salary.to_string())


def generate_insights(df: pd.DataFrame) -> None:
    """Generate key insights for job seekers."""
    print("\n" + "=" * 55)
    print("  KEY INSIGHTS FOR JOB SEEKERS")
    print("=" * 55)

    best_city = (df.groupby('city')['salary_lpa']
                 .mean().idxmax())
    best_city_salary = (df.groupby('city')['salary_lpa']
                        .mean().max())

    best_role = (df.groupby('job_title')['salary_lpa']
                 .mean().idxmax())
    best_role_salary = (df.groupby('job_title')['salary_lpa']
                        .mean().max())

    remote_avg = df[df['remote']]['salary_lpa'].mean()
    onsite_avg = df[~df['remote']]['salary_lpa'].mean()

    senior_avg = df[df['experience_years'] >= 5]['salary_lpa'].mean()
    junior_avg = df[df['experience_years'] < 2]['salary_lpa'].mean()

    print(f"\n🏆 Best city for salary: "
          f"{best_city} (₹{best_city_salary:.1f} LPA)")
    print(f"🏆 Highest paying role: "
          f"{best_role} (₹{best_role_salary:.1f} LPA)")
    print(f"💻 Remote premium: "
          f"₹{remote_avg - onsite_avg:.1f} LPA extra")
    print(f"📈 Experience premium (5+ vs <2 yrs): "
          f"₹{senior_avg - junior_avg:.1f} LPA extra")
    print(f"\n💡 Key Takeaway:")
    print(f"   A {best_role} in {best_city}")
    print(f"   with 5+ years experience earns")
    print(f"   ₹{best_role_salary:.0f} LPA on average!")
    print(f"\n   This is the power of data analysis! 🔥")


if __name__ == "__main__":
    df = create_comprehensive_dataset()
    analyze_salary_trends(df)
    analyze_experience_salary(df)
    analyze_skills_demand(df)
    analyze_remote_trends(df)
    generate_insights(df)
