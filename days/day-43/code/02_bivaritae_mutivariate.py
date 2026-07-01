"""
Day 43 — EDA: Exploratory Data Analysis
Topic: Bivariate and Multivariate Analysis
Date: 30 June 2026
Author: Bala Ravi

EDA Steps 4 + 5:
Bivariate = 2 features together
Multivariate = 3+ features together
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


def create_clean_dataset(n: int = 300) -> pd.DataFrame:
    """Create clean dataset for analysis."""
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
    role_bonus = {'Data Scientist': 3,
                  'ML Engineer': 4,
                  'Data Analyst': -2,
                  'AI Engineer': 5,
                  'Data Engineer': 1}

    salary = np.clip(
        10 + experience * 1.8 +
        np.array([city_bonus[c] for c in city]) +
        np.array([role_bonus[r] for r in role]) +
        np.random.normal(0, 2, n), 5, 55).round(1)

    return pd.DataFrame({
        'job_title': role,
        'city': city,
        'salary_lpa': salary,
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1),
        'remote': np.random.choice(
            [True, False], n, p=[0.3, 0.7])
    })


def step4_bivariate_num_num(
        df: pd.DataFrame) -> None:
    """Bivariate analysis — Numerical vs Numerical."""
    print("=" * 55)
    print("  EDA STEP 4 — BIVARIATE ANALYSIS")
    print("=" * 55)
    print("\n📈 Numerical vs Numerical:")

    num_cols = ['salary_lpa', 'experience_years',
                'skills_count', 'rating']

    corr_matrix = df[num_cols].corr()

    print("\nCorrelation with salary_lpa:")
    salary_corr = corr_matrix['salary_lpa'].drop(
        'salary_lpa').sort_values(ascending=False)

    for feat, corr in salary_corr.items():
        strength = ("Strong 🔥" if abs(corr) > 0.5 else
                   "Moderate ⚡" if abs(corr) > 0.3 else
                   "Weak ❄️")
        direction = "↑" if corr > 0 else "↓"
        print(f"  {feat:<22}: {corr:>6.2f} "
              f"{direction} {strength}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('Bivariate Analysis — '
                 'Feature vs Salary',
                 color='white', fontsize=14)

    pairs = [('experience_years', '#10b981'),
             ('skills_count', '#3b82f6'),
             ('rating', '#f59e0b')]

    for ax, (col, color) in zip(axes, pairs):
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

        sns.regplot(data=df, x=col, y='salary_lpa',
                   ax=ax, color=color,
                   scatter_kws={'alpha': 0.4, 's': 25},
                   line_kws={'linewidth': 2,
                             'color': '#ef4444'})

        corr_val = df[col].corr(df['salary_lpa'])
        ax.set_title(f'{col} vs Salary\n'
                    f'(corr={corr_val:.2f})',
                    color='white', fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/bivariate_num_num.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {path}")


def step4_bivariate_num_cat(
        df: pd.DataFrame) -> None:
    """Bivariate analysis — Numerical vs Categorical."""
    print("\n📊 Numerical vs Categorical:")

    print("\n  Salary by City:")
    city_stats = df.groupby('city')['salary_lpa'].agg(
        ['mean', 'median', 'std', 'count']).round(1)
    print(city_stats.to_string())

    print("\n  Salary by Role:")
    role_stats = df.groupby('job_title')['salary_lpa'].agg(
        ['mean', 'median', 'count']).round(1)
    role_stats = role_stats.sort_values(
        'mean', ascending=False)
    print(role_stats.to_string())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('Bivariate Analysis — '
                 'Salary by Category',
                 color='white', fontsize=14, y=1.01)

    palette = 'Greens'

    for ax in axes.flatten():
        ax.set_facecolor('#1e293b')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

    # Box plot by city
    order_city = (df.groupby('city')['salary_lpa']
                 .mean()
                 .sort_values(ascending=False)
                 .index)
    sns.boxplot(data=df, x='city', y='salary_lpa',
               order=order_city, hue='city',
               palette=palette, legend=False,
               ax=axes[0, 0])
    axes[0, 0].set_title('Salary by City',
                         color='white', fontsize=11)
    axes[0, 0].tick_params(
        axis='x', rotation=20)

    # Violin by role
    order_role = (df.groupby('job_title')['salary_lpa']
                 .mean()
                 .sort_values(ascending=False)
                 .index)
    sns.violinplot(data=df, x='job_title',
                  y='salary_lpa',
                  order=order_role, hue='job_title',
                  palette='Blues', legend=False,
                  ax=axes[0, 1])
    axes[0, 1].set_title('Salary by Role (Violin)',
                         color='white', fontsize=11)
    axes[0, 1].tick_params(axis='x', rotation=20)

    # Remote vs onsite
    sns.boxplot(data=df, x='remote', y='salary_lpa',
               hue='remote',
               palette=['#334155', '#10b981'],
               legend=False, ax=axes[1, 0])
    axes[1, 0].set_xticklabels(
        ['Onsite', 'Remote'],
        color='#94a3b8')
    axes[1, 0].set_title(
        'Remote vs Onsite Salary',
        color='white', fontsize=11)

    # Experience bucket vs salary
    df_plot = df.copy()
    df_plot['exp_level'] = pd.cut(
        df_plot['experience_years'],
        bins=[0, 2, 5, 8, 12],
        labels=['Fresher', 'Mid', 'Senior', 'Expert'],
        include_lowest=True)
    sns.barplot(data=df_plot, x='exp_level',
               y='salary_lpa',
               hue='exp_level',
               palette='Oranges', legend=False,
               ax=axes[1, 1])
    axes[1, 1].set_title(
        'Salary by Experience Level',
        color='white', fontsize=11)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/bivariate_num_cat.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {path}")


def step5_multivariate(df: pd.DataFrame) -> None:
    """Multivariate analysis — correlation + pairplot."""
    print("\n" + "=" * 55)
    print("  EDA STEP 5 — MULTIVARIATE ANALYSIS")
    print("=" * 55)

    num_cols = ['salary_lpa', 'experience_years',
                'skills_count', 'rating']
    corr = df[num_cols].corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('Multivariate Analysis',
                 color='white', fontsize=14)

    for ax in axes:
        ax.set_facecolor('#1e293b')

    # Correlation heatmap
    sns.heatmap(corr, annot=True, fmt='.2f',
               cmap='Greens', ax=axes[0],
               linewidths=1, linecolor='#0f172a',
               square=True,
               cbar_kws={'shrink': 0.8})
    axes[0].set_title('Correlation Matrix',
                      color='white', fontsize=12)
    axes[0].tick_params(colors='#94a3b8',
                       rotation=30)

    # Salary by city and role (grouped bar)
    pivot = df.groupby(
        ['city', 'job_title'])['salary_lpa'].mean().unstack()
    pivot = pivot.fillna(0)

    x = np.arange(len(pivot.index))
    width = 0.15
    colors_list = ['#10b981', '#3b82f6', '#f59e0b',
                   '#ef4444', '#8b5cf6']

    for i, (role, color) in enumerate(
            zip(pivot.columns, colors_list)):
        axes[1].bar(x + i * width, pivot[role],
                   width, label=role[:8],
                   color=color, alpha=0.85)

    axes[1].set_title('Salary by City × Role',
                      color='white', fontsize=12)
    axes[1].set_xlabel('City', color='#94a3b8')
    axes[1].set_ylabel('Avg Salary (₹ LPA)',
                       color='#94a3b8')
    axes[1].set_xticks(x + width * 2)
    axes[1].set_xticklabels(pivot.index,
                            color='#94a3b8',
                            rotation=15)
    axes[1].legend(facecolor='#334155',
                  labelcolor='white',
                  fontsize=8)
    axes[1].grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/multivariate.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {path}")


if __name__ == "__main__":
    df = create_clean_dataset()
    step4_bivariate_num_num(df)
    step4_bivariate_num_cat(df)
    step5_multivariate(df)
