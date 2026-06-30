"""
Day 42 — Seaborn: Statistical Visualization
Topic: Pairplot — Complete EDA in One Chart
Date: 29 June 2026
Author: Bala Ravi

Pairplot shows EVERY feature vs EVERY other feature!
The fastest way to do complete EDA!
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


def create_classification_dataset() -> pd.DataFrame:
    """
    Create dataset for classification-style EDA.
    Simulates job seniority classification!
    """
    np.random.seed(42)
    n = 200

    # Junior cluster
    junior_exp = np.random.normal(1.5, 0.8, n // 3)
    junior_salary = np.random.normal(12, 2, n // 3)
    junior_rating = np.random.normal(3.5, 0.4, n // 3)

    # Mid cluster
    mid_exp = np.random.normal(4, 1, n // 3)
    mid_salary = np.random.normal(20, 3, n // 3)
    mid_rating = np.random.normal(4.0, 0.3, n // 3)

    # Senior cluster
    senior_exp = np.random.normal(8, 1.5,
                                   n - 2*(n // 3))
    senior_salary = np.random.normal(
        32, 4, n - 2*(n // 3))
    senior_rating = np.random.normal(
        4.5, 0.3, n - 2*(n // 3))

    df = pd.DataFrame({
        'experience': np.concatenate(
            [junior_exp, mid_exp, senior_exp]),
        'salary_lpa': np.concatenate(
            [junior_salary, mid_salary, senior_salary]),
        'rating': np.concatenate(
            [junior_rating, mid_rating, senior_rating]),
        'level': (['Junior'] * (n // 3) +
                 ['Mid'] * (n // 3) +
                 ['Senior'] * (n - 2*(n // 3)))
    })

    df['experience'] = df['experience'].clip(
        0, 15).round(1)
    df['salary_lpa'] = df['salary_lpa'].clip(
        5, 55).round(1)
    df['rating'] = df['rating'].clip(1, 5).round(1)

    return df


def pairplot_demo(df: pd.DataFrame) -> None:
    """
    Create pairplot — sees ALL relationships at once!
    """
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#1e293b",
        "figure.facecolor": "#0f172a",
        "text.color": "white",
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
    })

    palette = {'Junior': '#3b82f6',
               'Mid': '#f59e0b',
               'Senior': '#10b981'}

    g = sns.pairplot(
        df, hue='level', palette=palette,
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 40},
        height=2.8)

    g.fig.patch.set_facecolor('#0f172a')
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_facecolor('#1e293b')

    g.fig.suptitle(
        'Complete EDA — Career Level Patterns',
        color='white', fontsize=14, y=1.02)

    path = f"{OUTPUT_DIR}/pairplot_eda.png"
    g.savefig(path, dpi=150, facecolor='#0f172a',
             bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")
    print(f"   → Shows ALL feature relationships")
    print(f"   → Career levels CLEARLY separable!")
    print(f"   → This means ML classification")
    print(f"     will work well on this data! 🎯")


def class_separability_insight(
        df: pd.DataFrame) -> None:
    """
    Explain what pairplot tells us about ML readiness.
    """
    print("\n=== ML Readiness Analysis ===\n")

    for level in df['level'].unique():
        subset = df[df['level'] == level]
        print(f"{level}:")
        print(f"  Experience: "
              f"{subset['experience'].mean():.1f} "
              f"± {subset['experience'].std():.1f} yrs")
        print(f"  Salary: "
              f"₹{subset['salary_lpa'].mean():.1f} "
              f"± ₹{subset['salary_lpa'].std():.1f} LPA")
        print(f"  Rating: "
              f"{subset['rating'].mean():.1f} "
              f"± {subset['rating'].std():.1f}")
        print()

    print("💡 Insight: If clusters in pairplot")
    print("   are well-separated (like above) —")
    print("   a simple classifier (Decision Tree,")
    print("   KNN) will achieve high accuracy!")
    print()
    print("   If clusters OVERLAP heavily —")
    print("   you need more features or a more")
    print("   complex model (Random Forest, NN)!")


if __name__ == "__main__":
    df = create_classification_dataset()
    print(f"Dataset: {df.shape[0]} samples\n")

    pairplot_demo(df)
    class_separability_insight(df)
