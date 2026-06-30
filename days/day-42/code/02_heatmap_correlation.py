"""
Day 42 — Seaborn: Statistical Visualization
Topic: Heatmaps and Correlation Analysis
Date: 29 June 2026
Author: Bala Ravi

THE most used Seaborn plot in ML!
Every ML project starts with a correlation heatmap.
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
    "text.color": "white",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
})


def create_ml_ready_dataset() -> pd.DataFrame:
    """Create dataset with multiple numeric features."""
    np.random.seed(42)
    n = 300

    experience = np.random.randint(0, 12, n)
    rating = np.clip(
        3 + experience * 0.1 +
        np.random.normal(0, 0.5, n), 1, 5)
    skills_count = np.random.randint(2, 10, n)
    age = experience + np.random.randint(22, 26, n)

    # Salary correlated with experience, rating, skills
    salary = (10 + experience * 1.8 +
              rating * 2 + skills_count * 0.8 +
              np.random.normal(0, 2, n))
    salary = np.clip(salary, 5, 55).round(1)

    # Posted days ago — uncorrelated (noise)
    posted_days = np.random.randint(1, 30, n)

    return pd.DataFrame({
        'salary_lpa': salary,
        'experience': experience,
        'rating': rating.round(1),
        'skills_count': skills_count,
        'age': age,
        'posted_days_ago': posted_days
    })


def correlation_heatmap_demo(
        df: pd.DataFrame) -> None:
    """
    Create correlation heatmap.
    ALWAYS the first chart in ML projects!
    """
    correlation = df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0f172a')

    sns.heatmap(correlation, annot=True,
               cmap='Greens', fmt='.2f',
               linewidths=1, linecolor='#0f172a',
               square=True, ax=ax,
               cbar_kws={'label': 'Correlation'},
               vmin=-1, vmax=1)

    ax.set_title(
        'Feature Correlation Matrix',
        color='white', fontsize=14, pad=15)
    ax.tick_params(axis='x', rotation=30,
                   colors='#94a3b8')
    ax.tick_params(axis='y', rotation=0,
                   colors='#94a3b8')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/correlation_heatmap.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")

    print("\n=== Correlation with Salary ===")
    salary_corr = correlation['salary_lpa'].drop(
        'salary_lpa').sort_values(ascending=False)
    for feature, corr in salary_corr.items():
        strength = ("Strong" if abs(corr) > 0.5 else
                    "Moderate" if abs(corr) > 0.3 else
                    "Weak")
        print(f"  {feature:<18}: {corr:>6.2f} "
              f"({strength})")


def regression_plots_demo(
        df: pd.DataFrame) -> None:
    """Show regplot with automatic trend line."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')

    sns.regplot(data=df, x='experience',
               y='salary_lpa', ax=axes[0],
               color='#10b981',
               scatter_kws={'alpha': 0.5, 's': 30},
               line_kws={'color': '#ef4444',
                        'linewidth': 2})
    axes[0].set_title(
        'Experience vs Salary (with trend)',
        color='white', fontsize=12)

    sns.regplot(data=df, x='skills_count',
               y='salary_lpa', ax=axes[1],
               color='#3b82f6',
               scatter_kws={'alpha': 0.5, 's': 30},
               line_kws={'color': '#ef4444',
                        'linewidth': 2})
    axes[1].set_title(
        'Skills Count vs Salary (with trend)',
        color='white', fontsize=12)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/regression_plots.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def feature_selection_insight(
        df: pd.DataFrame) -> None:
    """
    Use correlation to demonstrate feature selection.
    This is REAL ML preprocessing logic!
    """
    correlation = df.corr()['salary_lpa'].drop(
        'salary_lpa')

    print("\n=== ML Feature Selection Decision ===")
    print("Rule: |correlation| > 0.3 → KEEP feature")
    print("      |correlation| <= 0.3 → CONSIDER dropping\n")

    keep_features = []
    drop_features = []

    for feature, corr in correlation.items():
        if abs(corr) > 0.3:
            keep_features.append(feature)
            print(f"  ✅ KEEP '{feature}' "
                  f"(corr={corr:.2f})")
        else:
            drop_features.append(feature)
            print(f"  ⚠️ CONSIDER DROP '{feature}' "
                  f"(corr={corr:.2f})")

    print(f"\nFinal feature set for ML model:")
    print(f"  {keep_features}")
    print(f"\nThis is EXACTLY how feature selection")
    print(f"works in real Kaggle competitions! 🔥")


if __name__ == "__main__":
    df = create_ml_ready_dataset()
    print(f"Dataset shape: {df.shape}\n")

    correlation_heatmap_demo(df)
    regression_plots_demo(df)
    feature_selection_insight(df)
