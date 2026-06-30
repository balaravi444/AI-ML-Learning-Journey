"""
Day 42 — Seaborn: Statistical Visualization
Topic: ArthAI Financial Analysis with Seaborn
Date: 29 June 2026
Author: Bala Ravi

Apply Seaborn to real financial data analysis!
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


def create_transaction_dataset() -> pd.DataFrame:
    """Create realistic ArthAI transaction data."""
    np.random.seed(42)
    n = 300

    categories = ['Rent', 'Groceries', 'Transport',
                 'Dining', 'Entertainment', 'Shopping',
                 'Medical', 'Investment', 'EMI',
                 'Utilities']

    cat_ranges = {
        'Rent': (10000, 16000),
        'Groceries': (2000, 6000),
        'Transport': (500, 2500),
        'Dining': (300, 3500),
        'Entertainment': (200, 2500),
        'Shopping': (500, 12000),
        'Medical': (200, 6000),
        'Investment': (1000, 12000),
        'EMI': (5000, 22000),
        'Utilities': (500, 2500)
    }

    cat_list = np.random.choice(categories, n)
    amounts = [np.random.randint(*cat_ranges[c])
               for c in cat_list]
    months = np.random.choice(
        ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], n)

    return pd.DataFrame({
        'category': cat_list,
        'amount': amounts,
        'month': months,
        'type': ['investment' if c == 'Investment'
                else 'expense' for c in cat_list]
    })


def spending_boxplot_outliers(
        df: pd.DataFrame) -> None:
    """
    Box plot to find unusual transactions.
    Real ArthAI fraud/anomaly detection feature!
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('#0f172a')

    expense_df = df[df['type'] == 'expense']
    order = (expense_df.groupby('category')['amount']
            .median()
            .sort_values(ascending=False)
            .index)

    sns.boxplot(data=expense_df, x='category',
               y='amount', order=order,
               hue='category', palette='Greens_r',
               legend=False, ax=ax)

    ax.set_title(
        'Spending by Category (Outlier Detection)',
        color='white', fontsize=14)
    ax.set_xlabel('Category', color='#94a3b8')
    ax.set_ylabel('Amount (₹)', color='#94a3b8')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/arthai_spending_outliers.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def monthly_spending_trend(df: pd.DataFrame) -> None:
    """Line plot showing monthly spending trends."""
    monthly = (df[df['type'] == 'expense']
              .groupby(['month', 'category'])['amount']
              .sum()
              .reset_index())

    month_order = ['Jan', 'Feb', 'Mar',
                   'Apr', 'May', 'Jun']
    monthly['month'] = pd.Categorical(
        monthly['month'], categories=month_order,
        ordered=True)
    monthly = monthly.sort_values('month')

    top_categories = (df[df['type'] == 'expense']
                      .groupby('category')['amount']
                      .sum()
                      .nlargest(5)
                      .index)
    monthly_top = monthly[
        monthly['category'].isin(top_categories)]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0f172a')

    sns.lineplot(data=monthly_top, x='month',
                y='amount', hue='category',
                marker='o', linewidth=2.5,
                palette='Set2', ax=ax)

    ax.set_title('Monthly Spending Trend (Top 5 Categories)',
                color='white', fontsize=14)
    ax.set_xlabel('Month', color='#94a3b8')
    ax.set_ylabel('Amount (₹)', color='#94a3b8')
    ax.legend(facecolor='#334155',
              labelcolor='white',
              bbox_to_anchor=(1.02, 1),
              loc='upper left')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/arthai_monthly_trend.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


def budget_vs_actual_heatmap(
        df: pd.DataFrame) -> None:
    """Heatmap showing spending intensity by month/category."""
    pivot = df[df['type'] == 'expense'].pivot_table(
        values='amount', index='category',
        columns='month', aggfunc='sum', fill_value=0)

    month_order = ['Jan', 'Feb', 'Mar',
                   'Apr', 'May', 'Jun']
    pivot = pivot.reindex(
        columns=[m for m in month_order
                if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0f172a')

    sns.heatmap(pivot, annot=True, fmt='.0f',
               cmap='Greens', linewidths=1,
               linecolor='#0f172a', ax=ax,
               cbar_kws={'label': 'Amount (₹)'})

    ax.set_title('Spending Intensity Heatmap',
                color='white', fontsize=14, pad=15)
    ax.set_xlabel('Month', color='#94a3b8')
    ax.set_ylabel('Category', color='#94a3b8')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/arthai_spending_heatmap.png"
    plt.savefig(path, dpi=150, facecolor='#0f172a',
                bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    df = create_transaction_dataset()
    print(f"Transactions: {len(df)}\n")

    spending_boxplot_outliers(df)
    monthly_spending_trend(df)
    budget_vs_actual_heatmap(df)

    print(f"\n💰 ArthAI Seaborn dashboard complete!")
    print(f"   These charts could be added to the")
    print(f"   ArthAI Portfolio Tracker module! 🔥")
