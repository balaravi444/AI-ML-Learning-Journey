"""
Day 45 — Statistics for ML
Topic: Descriptive Statistics
Date: 02 July 2026
Author: Bala Ravi

Descriptive statistics = summarizing data!
The foundation of every ML project!
"""
import numpy as np
import pandas as pd
from scipy import stats


def create_salary_dataset(n: int = 500) -> pd.DataFrame:
    """Create realistic salary dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']

    experience = np.random.randint(0, 12, n)
    city = np.random.choice(cities, n)
    role = np.random.choice(roles, n)

    city_bonus = {'Bangalore': 3, 'Mumbai': 2,
                  'Delhi': 1, 'Hyderabad': 0.5,
                  'Pune': 0}

    salary = np.clip(
        10 + experience * 1.8 +
        np.array([city_bonus[c] for c in city]) +
        np.random.normal(0, 3, n), 5, 55).round(1)

    return pd.DataFrame({
        'salary_lpa': salary,
        'experience_years': experience,
        'city': city,
        'job_title': role,
        'skills_count': np.random.randint(2, 10, n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1)
    })


def descriptive_stats_demo(
        df: pd.DataFrame) -> None:
    """Complete descriptive statistics analysis."""
    print("=== Descriptive Statistics ===\n")

    col = 'salary_lpa'
    data = df[col]

    # Central tendency
    mean = data.mean()
    median = data.median()
    mode = data.mode()[0]

    print(f"📊 Salary Distribution Analysis:")
    print(f"  Mean:   ₹{mean:.2f} LPA "
          f"(affected by outliers!)")
    print(f"  Median: ₹{median:.2f} LPA "
          f"(robust to outliers!)")
    print(f"  Mode:   ₹{mode:.2f} LPA "
          f"(most common salary)")

    if abs(mean - median) > 2:
        print(f"\n  ⚠️ Mean ≠ Median → "
              f"distribution is SKEWED!")
        print(f"  Use MEDIAN for missing value fill!")
    else:
        print(f"\n  ✅ Mean ≈ Median → "
              f"approximately symmetric!")

    # Spread
    std = data.std()
    var = data.var()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    data_range = data.max() - data.min()

    print(f"\n📐 Spread Measures:")
    print(f"  Std Dev:  ₹{std:.2f} LPA")
    print(f"  Variance: {var:.2f}")
    print(f"  IQR:      ₹{iqr:.2f} LPA "
          f"(Q3-Q1, robust spread)")
    print(f"  Range:    ₹{data_range:.2f} LPA")

    # Shape
    skew = data.skew()
    kurt = data.kurtosis()

    print(f"\n📏 Shape Measures:")
    print(f"  Skewness: {skew:.3f}")
    if skew > 1:
        print(f"    → Right-skewed (long right tail)")
        print(f"    → Log transform recommended!")
    elif skew < -1:
        print(f"    → Left-skewed (long left tail)")
    else:
        print(f"    → Approximately normal ✅")

    print(f"  Kurtosis: {kurt:.3f}")
    if kurt > 3:
        print(f"    → Heavy tails (more outliers)")
    else:
        print(f"    → Light tails (fewer outliers)")

    # Percentiles
    print(f"\n📈 Percentiles:")
    for p in [10, 25, 50, 75, 90, 95]:
        val = data.quantile(p/100)
        print(f"  P{p:>2}: ₹{val:.1f} LPA")

    # Outlier detection
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = data[(data < lower) | (data > upper)]
    print(f"\n🔍 Outlier Detection (IQR method):")
    print(f"  Lower bound: ₹{lower:.1f} LPA")
    print(f"  Upper bound: ₹{upper:.1f} LPA")
    print(f"  Outliers:    {len(outliers)} "
          f"({len(outliers)/len(data)*100:.1f}%)")


def distribution_check(df: pd.DataFrame) -> None:
    """Check if features follow normal distribution."""
    print("\n=== Normality Tests ===\n")
    print("Rule: p > 0.05 → approximately normal\n")

    for col in ['salary_lpa', 'experience_years',
                'skills_count', 'rating']:
        data = df[col]
        stat, p = stats.normaltest(data)
        is_normal = p > 0.05
        print(f"{col:<22}: "
              f"p={p:.4f} "
              f"{'✅ Normal' if is_normal else '⚠️ Not Normal'}")

    # Log transform check
    print("\n--- After Log Transform ---")
    salary_log = np.log1p(df['salary_lpa'])
    stat, p = stats.normaltest(salary_log)
    print(f"salary_lpa (log):      "
          f"p={p:.4f} "
          f"{'✅ Normal' if p > 0.05 else '⚠️ Not Normal'}")
    print(f"\n→ Log transform makes salary "
          f"more normal!")
    print(f"→ Better for linear models!")


if __name__ == "__main__":
    df = create_salary_dataset()
    descriptive_stats_demo(df)
    distribution_check(df)
