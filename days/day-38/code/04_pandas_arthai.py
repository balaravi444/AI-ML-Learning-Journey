"""
Day 38 — Pandas: DataFrames
Topic: ArthAI Data Analysis with Pandas
Date: 25 June 2026
Author: Bala Ravi

Apply Pandas to real financial data analysis!
This is the foundation of Indian Job Market
Analyzer project coming on Day 47!
"""
import pandas as pd
import numpy as np


def create_financial_dataset() -> pd.DataFrame:
    """
    Create a realistic personal finance dataset.
    Simulates a user's transaction history!
    """
    np.random.seed(42)
    n = 100

    categories = [
        'Rent', 'Groceries', 'Transport',
        'Food/Dining', 'Entertainment',
        'Shopping', 'Medical', 'Investment',
        'EMI', 'Utilities'
    ]

    amounts = {
        'Rent': (8000, 15000),
        'Groceries': (2000, 5000),
        'Transport': (500, 2000),
        'Food/Dining': (300, 3000),
        'Entertainment': (200, 2000),
        'Shopping': (500, 10000),
        'Medical': (200, 5000),
        'Investment': (1000, 10000),
        'EMI': (5000, 20000),
        'Utilities': (500, 2000)
    }

    dates = pd.date_range(
        start='2026-01-01',
        end='2026-06-30',
        periods=n)

    cat_list = np.random.choice(categories, n)
    amount_list = [
        np.random.randint(*amounts[cat])
        for cat in cat_list
    ]

    return pd.DataFrame({
        'date': dates,
        'category': cat_list,
        'amount': amount_list,
        'month': [d.strftime('%B') for d in dates],
        'type': ['expense' if cat != 'Investment'
                 else 'investment'
                 for cat in cat_list]
    })


def analyze_spending(df: pd.DataFrame) -> None:
    """
    Analyze spending patterns using Pandas.
    This is what ArthAI's analytics dashboard does!
    """
    print("=== ArthAI Spending Analysis ===\n")
    print(f"Transaction history: {len(df)} records")
    print(f"Date range: {df['date'].min().date()} "
          f"to {df['date'].max().date()}")

    print(f"\n💰 Total spent: "
          f"₹{df[df['type']=='expense']['amount'].sum():,}")
    print(f"💹 Total invested: "
          f"₹{df[df['type']=='investment']['amount'].sum():,}")

    print(f"\n📊 Spending by category:")
    category_spend = (
        df[df['type'] == 'expense']
        .groupby('category')['amount']
        .agg(['sum', 'mean', 'count'])
        .sort_values('sum', ascending=False)
    )
    category_spend.columns = [
        'Total (₹)', 'Average (₹)', 'Count']
    category_spend['Total (₹)'] = (
        category_spend['Total (₹)'].apply(
            lambda x: f"₹{x:,}"))
    category_spend['Average (₹)'] = (
        category_spend['Average (₹)'].apply(
            lambda x: f"₹{x:,.0f}"))
    print(category_spend)

    print(f"\n📅 Monthly spending trend:")
    monthly = (df[df['type'] == 'expense']
               .groupby('month')['amount']
               .sum()
               .reset_index())
    monthly.columns = ['Month', 'Total Spent (₹)']
    for _, row in monthly.iterrows():
        bar = '█' * int(row['Total Spent (₹)'] // 2000)
        print(f"  {row['Month']:>10}: "
              f"₹{row['Total Spent (₹)']:>8,} {bar}")


def budget_vs_actual(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare actual spending vs budget.
    Core feature of ArthAI Budget Planner!

    Args:
        df: Transaction DataFrame

    Returns:
        Budget analysis DataFrame
    """
    print("\n=== Budget vs Actual ===\n")

    monthly_budget = {
        'Rent': 12000,
        'Groceries': 3000,
        'Transport': 1500,
        'Food/Dining': 2000,
        'Entertainment': 1000,
        'Shopping': 3000,
        'Medical': 1000,
        'Investment': 5000,
        'EMI': 10000,
        'Utilities': 1500
    }

    actual = (df[df['type'] == 'expense']
              .groupby('category')['amount']
              .mean()
              .round(0))

    budget_df = pd.DataFrame({
        'Budget (₹)': monthly_budget,
        'Actual (₹)': actual
    }).fillna(0)

    budget_df['Difference'] = (
        budget_df['Budget (₹)'] -
        budget_df['Actual (₹)'])
    budget_df['Status'] = budget_df['Difference'].apply(
        lambda x: '✅ Under' if x >= 0 else '❌ Over')

    print(budget_df.to_string())

    over_budget = budget_df[
        budget_df['Status'] == '❌ Over']
    if len(over_budget) > 0:
        print(f"\n⚠️ Over budget in "
              f"{len(over_budget)} categories!")

    return budget_df


if __name__ == "__main__":
    print("=== ArthAI Financial Analysis ===\n")

    df = create_financial_dataset()
    print("Transaction data created!")
    print(df.head())

    analyze_spending(df)
    budget_df = budget_vs_actual(df)

    print(f"\n✅ Pandas makes financial analysis simple!")
    print(f"   This powers ArthAI's budget tracker!")
