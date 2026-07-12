"""
Day 43 — EDA: Exploratory Data Analysis
Topic: Data Quality + Univariate Analysis
Date: 30 June 2026
Author: Bala Ravi

EDA Step 1 + 2: Load data, check quality,
analyze each feature individually.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "eda_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e293b",
    "figure.facecolor": "#0f172a",
    "text.color": "white",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
})


def create_raw_job_dataset(n: int = 300) -> pd.DataFrame:
    """
    Create realistic raw job dataset with
    real-world data quality issues.
    """
    np.random.seed(42)

    cities = ['Bangalore', 'bangalore', 'MUMBAI',
              'Mumbai', 'Delhi', ' Delhi ',
              'Hyderabad', 'Pune', None]
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer', None]
    companies = ['TCS', 'Infosys', 'Amazon',
                 'Google', 'Microsoft', 'Flipkart',
                 'Swiggy', 'Paytm', None]

    experience = np.random.randint(0, 12, n)
    base_salary = 10 + experience * 1.8
    salary_raw = base_salary + np.random.normal(0, 3, n)
    salary_raw = np.clip(salary_raw, 5, 55)

    # Add outliers
    salary_raw[np.random.choice(n, 5)] = [
        150, 200, -5, 0.5, 180]

    # Convert some to strings (real world problem!)
    salary_list = []
    for s in salary_raw:
        r = np.random.random()
        if r < 0.05:
            salary_list.append(None)
        elif r < 0.08:
            salary_list.append(f"₹{s:.0f} LPA")
        else:
            salary_list.append(s)

    return pd.DataFrame({
        'job_title': np.random.choice(roles, n),
        'company': np.random.choice(companies, n),
        'city': np.random.choice(cities, n),
        'salary_lpa': salary_list,
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1),
        'remote': np.random.choice(
            [True, False, None], n,
            p=[0.28, 0.68, 0.04]),
        'posted_days': np.random.randint(1, 30, n)
    })


def step1_first_look(df: pd.DataFrame) -> None:
    """EDA Step 1 — First look at data."""
    print("=" * 55)
    print("  EDA STEP 1 — FIRST LOOK")
    print("=" * 55)

    print(f"\n📐 Shape: {df.shape}")
    print(f"   {df.shape[0]} rows × "
          f"{df.shape[1]} columns\n")

    print("📋 First 5 rows:")
    print(df.head().to_string())

    print(f"\n📊 Column Info:")
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].isnull().sum()
        n_unique = df[col].nunique()
        print(f"  {col:<20} | {str(dtype):<10} | "
              f"missing: {n_missing:>3} | "
              f"unique: {n_unique:>4}")


def step2_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """EDA Step 2 — Data quality check and fix."""
    print("\n" + "=" * 55)
    print("  EDA STEP 2 — DATA QUALITY CHECK")
    print("=" * 55)

    # Missing values
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    for col, (cnt, pct) in zip(
            df.columns,
            zip(missing, missing_pct)):
        if cnt > 0:
            print(f"  {col:<20}: {cnt:>3} "
                  f"({pct}%)")

    # Duplicates
    dupes = df.duplicated().sum()
    print(f"\n🔄 Duplicate rows: {dupes}")

    # Data type issues
    print(f"\n⚠️ Data Type Issues:")
    print(f"  'salary_lpa' has mixed types: "
          f"{df['salary_lpa'].apply(type).unique()}")

    # Fix issues
    print(f"\n🔧 Fixing Issues...")
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Clean salary (extract numbers from strings)
    def clean_salary(val):
        if val is None:
            return np.nan
        if isinstance(val, str):
            import re
            nums = re.findall(r'\d+\.?\d*', val)
            return float(nums[0]) if nums else np.nan
        return float(val)

    df['salary_lpa'] = df['salary_lpa'].apply(
        clean_salary)

    # Fill missing values
    df['salary_lpa'].fillna(
        df['salary_lpa'].median(), inplace=True)
    df['job_title'].fillna(
        'Unknown', inplace=True)
    df['company'].fillna('Unknown', inplace=True)
    df['remote'].fillna(False, inplace=True)

    # Clean strings
    df['city'] = (df['city']
                 .fillna('Unknown')
                 .str.strip()
                 .str.title())

    # Remove salary outliers (IQR)
    Q1 = df['salary_lpa'].quantile(0.25)
    Q3 = df['salary_lpa'].quantile(0.75)
    IQR = Q3 - Q1
    before = len(df)
    df = df[(df['salary_lpa'] >= Q1 - 1.5*IQR) &
            (df['salary_lpa'] <= Q3 + 1.5*IQR)]
    df = df.reset_index(drop=True)

    print(f"  ✅ Removed {before - len(df)} outliers")
    print(f"  ✅ Fixed {missing.sum()} missing values")
    print(f"  ✅ Cleaned city names")
    print(f"  ✅ Fixed salary data types")
    print(f"\n  Clean shape: {df.shape}")

    return df


def step3_univariate_numerical(
        df: pd.DataFrame) -> None:
    """EDA Step 3a — Numerical feature analysis."""
    print("\n" + "=" * 55)
    print("  EDA STEP 3 — UNIVARIATE ANALYSIS")
    print("=" * 55)

    numerical_cols = ['salary_lpa', 'experience_years',
                      'skills_count', 'rating']

    print("\n📊 Numerical Features Summary:")
    print(df[numerical_cols].describe().round(2))

    print("\n📐 Skewness Check:")
    for col in numerical_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            verdict = "⚠️ HIGHLY SKEWED → log transform!"
        elif abs(skew) > 0.5:
            verdict = "⚡ Moderately skewed"
        else:
            verdict = "✅ Approximately normal"
        print(f"  {col:<22}: {skew:>6.2f} {verdict}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('Univariate Analysis — Numerical Features',
                 color='white', fontsize=15, y=1.01)

    for ax, col in zip(axes.flatten(),
                       numerical_cols):
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

        sns.histplot(data=df, x=col, kde=True,
                    color='#10b981', ax=ax)
        ax.axvline(df[col].mean(), color='#ef4444',
                   linestyle='--', linewidth=2,
                   label=f"Mean: {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color='#f59e0b',
                   linestyle='--', linewidth=2,
                   label=f"Median: {df[col].median():.1f}")
        ax.set_title(f'{col} Distribution',
                    color='white', fontsize=11)
        ax.legend(facecolor='#334155',
                 labelcolor='white', fontsize=8)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/univariate_numerical.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {path}")


def step3_univariate_categorical(
        df: pd.DataFrame) -> None:
    """EDA Step 3b — Categorical feature analysis."""
    print("\n📋 Categorical Features Summary:")

    cat_cols = ['job_title', 'city', 'company']
    for col in cat_cols:
        print(f"\n  {col}:")
        counts = df[col].value_counts().head(6)
        for val, cnt in counts.items():
            pct = cnt / len(df) * 100
            bar = '█' * int(pct // 2)
            print(f"    {val:<28}: "
                  f"{cnt:>3} ({pct:.0f}%) {bar}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('Univariate Analysis — '
                 'Categorical Features',
                 color='white', fontsize=14)

    colors = ['#10b981', '#3b82f6', '#f59e0b',
              '#ef4444', '#8b5cf6', '#ec4899']

    for ax, col in zip(axes, cat_cols):
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

        top6 = df[col].value_counts().head(6)
        ax.barh(top6.index[::-1], top6.values[::-1],
               color=colors[:len(top6)])

        for i, (val, cnt) in enumerate(
                zip(top6.index[::-1],
                    top6.values[::-1])):
            ax.text(cnt + 0.5, i,
                   f'{cnt}', va='center',
                   color='white', fontsize=9)

        ax.set_title(f'Top {col}s',
                    color='white', fontsize=11)
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, axis='x', alpha=0.2)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/univariate_categorical.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {path}")


if __name__ == "__main__":
    print("🔍 Starting EDA...\n")

    df_raw = create_raw_job_dataset()
    step1_first_look(df_raw)
    df_clean = step2_data_quality(df_raw)
    step3_univariate_numerical(df_clean)
    step3_univariate_categorical(df_clean)
