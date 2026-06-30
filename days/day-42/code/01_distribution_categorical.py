"""
Day 42 — Seaborn: Statistical Visualization
Topic: Distribution and Categorical Plots
Date: 29 June 2026
Author: Bala Ravi

Seaborn = Statistics + Matplotlib in one line!
Built directly on top of Matplotlib and Pandas.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e293b",
    "figure.facecolor": "#0f172a",
    "axes.edgecolor": "#334155",
    "grid.color": "#334155",
    "text.color": "white",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
})


def create_job_dataset() -> pd.DataFrame:
    """Create comprehensive job market dataset."""
    np.random.seed(42)
    n = 250

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer']

    experience = np.random.randint(0, 12, n)
    city = np.random.choice(cities, n)
    role = np.random.choice(roles, n)

    city_bonus = {'Bangalore': 3, 'Mumbai': 2,
                  'Delhi': 1, 'Hyderabad': 0.5,
                  'Pune': 0}
    role_bonus = {'Data Scientist': 3,
                  'ML Engineer': 4,
                  'Data Analyst': -2,
                  'AI Engineer': 5}

    salary = (10 + experience * 1.8 +
              np.array([city_bonus[c] for c in city]) +
              np.array([role_bonus[r] for r in role]) +
              np.random.normal(0, 2.5, n))
    salary = np.clip(salary, 5, 55).round(1)

    return pd.DataFrame({
        'job_title': role,
        'city': city,
        'salary_lpa': salary,
        'experience': experience,
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1),
        'remote': np.random.choice(
            [True, False], n, p=[0.3, 0.7])
    })


def distribution_plots_demo(df: pd.DataFrame) -> None:
    """Show distribution plots with KDE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')

    # Histogram with KDE overlay
    sns.histplot(data=df, x='salary_lpa', kde=True,
                 color='#10b981', ax=axes[0])
    axes[0].set_title('Salary Distribution (with KDE)',
                      color='white', fontsize=12)
    axes[0].set_xlabel('Salary (₹ LPA)')

    # Distribution by city (overlapping)
    for city in df['city'].unique()[:3]:
        subset = df[df['city'] == city]
        sns.kdeplot(data=subset, x='salary_lpa',
                   label=city, ax=axes[1], fill=True,
                   alpha=0.3)
    axes[1].set_title('Salary Distribution by City',
                      color='white', fontsize=12)
    axes[1].set_xlabel('Salary (₹ LPA)')
    axes[1].legend(facecolor='#334155',
                   labelcolor='white')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/seaborn_distributions.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def categorical_plots_demo(df: pd.DataFrame) -> None:
    """Show boxplot, violinplot, barplot comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f172a')

    palette = ['#10b981', '#3b82f6', '#f59e0b',
              '#ef4444', '#8b5cf6']

    # Box plot — quartiles + outliers
    sns.boxplot(data=df, x='city', y='salary_lpa',
               hue='city', palette=palette,
               ax=axes[0], legend=False)
    axes[0].set_title('Box Plot — Salary by City',
                      color='white', fontsize=11)
    axes[0].tick_params(axis='x', rotation=20)

    # Violin plot — shape + quartiles
    sns.violinplot(data=df, x='city', y='salary_lpa',
                  hue='city', palette=palette,
                  ax=axes[1], legend=False)
    axes[1].set_title('Violin Plot — Distribution Shape',
                      color='white', fontsize=11)
    axes[1].tick_params(axis='x', rotation=20)

    # Bar plot — mean with confidence interval
    sns.barplot(data=df, x='city', y='salary_lpa',
               hue='city', palette=palette,
               ax=axes[2], legend=False,
               errorbar='sd')
    axes[2].set_title('Bar Plot — Mean ± Std Dev',
                      color='white', fontsize=11)
    axes[2].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/seaborn_categorical.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def outlier_detection_boxplot(
        df: pd.DataFrame) -> None:
    """
    Use box plot for visual outlier detection.
    Same IQR math from Day 39 — now visual!
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f172a')

    sns.boxplot(data=df, y='salary_lpa',
               x='job_title', hue='job_title',
               palette='Greens', ax=ax,
               legend=False)

    ax.set_title(
        'Outlier Detection — Salary by Role',
        color='white', fontsize=13)
    ax.set_xlabel('Job Title', color='#94a3b8')
    ax.set_ylabel('Salary (₹ LPA)',
                  color='#94a3b8')
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/outlier_boxplot.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

    # Print actual outlier values for verification
    print("\nOutlier values detected (IQR method):")
    for role in df['job_title'].unique():
        subset = df[df['job_title'] == role]['salary_lpa']
        Q1, Q3 = subset.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = subset[
            (subset < Q1 - 1.5*IQR) |
            (subset > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            print(f"  {role}: {list(outliers.values)}")


if __name__ == "__main__":
    df = create_job_dataset()
    print(f"Dataset: {df.shape[0]} jobs\n")

    distribution_plots_demo(df)
    categorical_plots_demo(df)
    outlier_detection_boxplot(df)
