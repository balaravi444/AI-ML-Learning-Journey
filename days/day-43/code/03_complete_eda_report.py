"""
Day 43 — EDA: Exploratory Data Analysis
Topic: Complete EDA Report Generator
Date: 30 June 2026
Author: Bala Ravi

Full EDA workflow — from raw data to insights!
This is EXACTLY what the Indian Job Market
Analyzer project will do on Day 47!
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


def create_job_dataset(n: int = 400) -> pd.DataFrame:
    """Create comprehensive clean job dataset."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune', 'Chennai']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer', 'MLOps Engineer']
    companies = ['TCS', 'Infosys', 'Amazon',
                 'Google', 'Microsoft', 'Flipkart',
                 'Swiggy', 'Paytm', 'Razorpay',
                 'Meesho']

    experience = np.random.randint(0, 12, n)
    city = np.random.choice(cities, n)
    role = np.random.choice(roles, n)

    city_bonus = {'Bangalore': 3, 'Mumbai': 2,
                  'Delhi': 1, 'Hyderabad': 0.5,
                  'Pune': 0, 'Chennai': 0.3}
    role_bonus = {'Data Scientist': 3,
                  'ML Engineer': 4,
                  'Data Analyst': -2,
                  'AI Engineer': 5,
                  'Data Engineer': 1,
                  'MLOps Engineer': 3}

    salary = np.clip(
        10 + experience * 1.8 +
        np.array([city_bonus[c] for c in city]) +
        np.array([role_bonus[r] for r in role]) +
        np.random.normal(0, 2, n), 5, 55).round(1)

    return pd.DataFrame({
        'job_title': role,
        'company': np.random.choice(companies, n),
        'city': city,
        'salary_lpa': salary,
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1),
        'remote': np.random.choice(
            [True, False], n, p=[0.3, 0.7])
    })


def run_complete_eda(df: pd.DataFrame) -> dict:
    """
    Run complete EDA and return key insights.
    This is the core of Indian Job Market Analyzer!

    Returns:
        Dictionary of EDA findings
    """
    insights = {}

    # Basic stats
    insights['total_jobs'] = len(df)
    insights['avg_salary'] = df['salary_lpa'].mean()
    insights['median_salary'] = df['salary_lpa'].median()
    insights['salary_std'] = df['salary_lpa'].std()
    insights['salary_skew'] = df['salary_lpa'].skew()

    # Best city
    city_salary = df.groupby(
        'city')['salary_lpa'].mean()
    insights['best_city'] = city_salary.idxmax()
    insights['best_city_salary'] = city_salary.max()

    # Best role
    role_salary = df.groupby(
        'job_title')['salary_lpa'].mean()
    insights['best_role'] = role_salary.idxmax()
    insights['best_role_salary'] = role_salary.max()

    # Remote premium
    remote_avg = df[df['remote']]['salary_lpa'].mean()
    onsite_avg = df[~df['remote']]['salary_lpa'].mean()
    insights['remote_premium'] = remote_avg - onsite_avg

    # Experience impact
    senior = df[df['experience_years'] >= 5]
    junior = df[df['experience_years'] < 2]
    insights['exp_premium'] = (
        senior['salary_lpa'].mean() -
        junior['salary_lpa'].mean())

    # Correlation with salary
    num_cols = ['experience_years',
                'skills_count', 'rating']
    correlations = {col: df[col].corr(
        df['salary_lpa']) for col in num_cols}
    insights['top_predictor'] = max(
        correlations, key=correlations.get)
    insights['correlations'] = correlations

    return insights


def generate_eda_dashboard(df: pd.DataFrame) -> None:
    """Generate complete EDA visual dashboard."""
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle(
        '🔍 Complete EDA — Indian AI/ML Job Market',
        color='white', fontsize=18,
        fontweight='bold', y=0.99)

    def style_ax(ax):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8',
                      labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.grid(True, alpha=0.15, color='#64748b')

    # 1. Salary Distribution (top-left)
    ax1 = fig.add_subplot(3, 3, 1)
    style_ax(ax1)
    sns.histplot(data=df, x='salary_lpa', kde=True,
                color='#10b981', ax=ax1)
    ax1.axvline(df['salary_lpa'].mean(),
               color='#ef4444', linestyle='--',
               linewidth=1.5)
    ax1.set_title('Salary Distribution',
                 color='white', fontsize=10)

    # 2. City salary box plot (top-center)
    ax2 = fig.add_subplot(3, 3, 2)
    style_ax(ax2)
    order_city = (df.groupby('city')['salary_lpa']
                 .mean()
                 .sort_values(ascending=False)
                 .index)
    sns.boxplot(data=df, x='city', y='salary_lpa',
               order=order_city, hue='city',
               palette='Greens', legend=False,
               ax=ax2)
    ax2.set_title('Salary by City',
                 color='white', fontsize=10)
    ax2.tick_params(axis='x', rotation=25)
    ax2.set_xlabel('')

    # 3. Role salary bar (top-right)
    ax3 = fig.add_subplot(3, 3, 3)
    style_ax(ax3)
    role_avg = (df.groupby('job_title')['salary_lpa']
               .mean()
               .sort_values(ascending=True))
    ax3.barh(role_avg.index, role_avg.values,
            color='#3b82f6', edgecolor='#0f172a')
    ax3.set_title('Avg Salary by Role',
                 color='white', fontsize=10)
    for i, v in enumerate(role_avg.values):
        ax3.text(v + 0.1, i, f'₹{v:.0f}L',
                va='center', color='white',
                fontsize=7)

    # 4. Experience vs Salary scatter (middle-left)
    ax4 = fig.add_subplot(3, 3, 4)
    style_ax(ax4)
    sns.regplot(data=df, x='experience_years',
               y='salary_lpa', ax=ax4,
               color='#10b981',
               scatter_kws={'alpha': 0.4, 's': 20},
               line_kws={'color': '#ef4444',
                        'linewidth': 2})
    corr = df['experience_years'].corr(
        df['salary_lpa'])
    ax4.set_title(
        f'Experience vs Salary (r={corr:.2f})',
        color='white', fontsize=10)

    # 5. Correlation heatmap (middle-center)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.set_facecolor('#1e293b')
    num_cols = ['salary_lpa', 'experience_years',
                'skills_count', 'rating']
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f',
               cmap='Greens', ax=ax5,
               linewidths=1,
               linecolor='#0f172a',
               cbar=False)
    ax5.set_title('Correlation Matrix',
                 color='white', fontsize=10)
    ax5.tick_params(colors='#94a3b8', rotation=30,
                   labelsize=8)

    # 6. Remote vs Onsite (middle-right)
    ax6 = fig.add_subplot(3, 3, 6)
    style_ax(ax6)
    remote_data = df.groupby('remote')['salary_lpa']
    means = [remote_data.get_group(False).mean(),
             remote_data.get_group(True).mean()]
    bars = ax6.bar(['Onsite', 'Remote'], means,
                  color=['#334155', '#10b981'],
                  edgecolor='#0f172a', width=0.5)
    for bar, mean in zip(bars, means):
        ax6.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'₹{mean:.1f}L', ha='center',
                color='white', fontweight='bold',
                fontsize=9)
    ax6.set_title('Remote vs Onsite Salary',
                 color='white', fontsize=10)
    ax6.set_ylim(0, max(means) * 1.25)

    # 7. Experience distribution (bottom-left)
    ax7 = fig.add_subplot(3, 3, 7)
    style_ax(ax7)
    sns.histplot(data=df, x='experience_years',
                bins=12, color='#f59e0b', ax=ax7)
    ax7.set_title('Experience Distribution',
                 color='white', fontsize=10)

    # 8. Job count by city (bottom-center)
    ax8 = fig.add_subplot(3, 3, 8)
    style_ax(ax8)
    city_counts = df['city'].value_counts()
    palette_city = ['#10b981'] + ['#334155'] * 5
    ax8.bar(city_counts.index, city_counts.values,
           color=palette_city,
           edgecolor='#0f172a')
    ax8.set_title('Job Count by City',
                 color='white', fontsize=10)
    ax8.tick_params(axis='x', rotation=20)
    for i, v in enumerate(city_counts.values):
        ax8.text(i, v + 0.5, str(v),
                ha='center', color='white',
                fontsize=8)

    # 9. Skills count distribution (bottom-right)
    ax9 = fig.add_subplot(3, 3, 9)
    style_ax(ax9)
    skills_salary = df.groupby(
        'skills_count')['salary_lpa'].mean()
    ax9.plot(skills_salary.index,
            skills_salary.values,
            color='#10b981', linewidth=2.5,
            marker='o', markersize=6)
    ax9.set_title('Skills Count vs Avg Salary',
                 color='white', fontsize=10)
    ax9.set_xlabel('Number of Skills',
                  color='#94a3b8', fontsize=9)
    ax9.set_ylabel('Avg Salary (₹ LPA)',
                  color='#94a3b8', fontsize=9)

    plt.subplots_adjust(
        hspace=0.45, wspace=0.35,
        top=0.94, bottom=0.05)

    path = f"{OUTPUT_DIR}/complete_eda_dashboard.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


def print_eda_summary(df: pd.DataFrame) -> None:
    """Print complete EDA insights report."""
    insights = run_complete_eda(df)

    print("\n" + "=" * 55)
    print("  📋 COMPLETE EDA REPORT")
    print("  Indian AI/ML Job Market 2026")
    print("=" * 55)

    print(f"\n📊 Dataset Overview:")
    print(f"  Total job postings: "
          f"{insights['total_jobs']}")
    print(f"  Avg salary: "
          f"₹{insights['avg_salary']:.1f} LPA")
    print(f"  Median salary: "
          f"₹{insights['median_salary']:.1f} LPA")
    print(f"  Salary skewness: "
          f"{insights['salary_skew']:.2f}")

    if abs(insights['salary_skew']) > 1:
        print(f"  ⚠️ Highly skewed → "
              f"log transform recommended for ML!")
    else:
        print(f"  ✅ Approximately normal → "
              f"ready for linear models!")

    print(f"\n🏆 Key Findings:")
    print(f"  Best city: {insights['best_city']} "
          f"(₹{insights['best_city_salary']:.1f} LPA)")
    print(f"  Best role: {insights['best_role']} "
          f"(₹{insights['best_role_salary']:.1f} LPA)")
    print(f"  Remote premium: "
          f"₹{insights['remote_premium']:.1f} LPA extra")
    print(f"  Experience premium (5+ vs <2 yrs): "
          f"₹{insights['exp_premium']:.1f} LPA")

    print(f"\n🔗 Feature Correlations with Salary:")
    for feat, corr in sorted(
            insights['correlations'].items(),
            key=lambda x: abs(x[1]),
            reverse=True):
        strength = ("Strong" if abs(corr) > 0.5 else
                   "Moderate" if abs(corr) > 0.3 else
                   "Weak")
        print(f"  {feat:<22}: {corr:>6.2f} "
              f"({strength})")

    print(f"\n💡 ML Model Recommendations:")
    print(f"  Top predictor: "
          f"'{insights['top_predictor']}'")
    print(f"  Suggested features: "
          f"experience_years, city, job_title, skills_count")
    print(f"  Algorithm: Random Forest or XGBoost "
          f"(handles categorical features well)")
    print(f"  Target: salary_lpa prediction")

    print(f"\n🚀 This EDA powers the")
    print(f"   Indian Job Market Analyzer (Day 47)!")


if __name__ == "__main__":
    print("🔍 Running Complete EDA Workflow...\n")

    df = create_job_dataset()
    generate_eda_dashboard(df)
    print_eda_summary(df)

    print(f"\n📁 All EDA charts saved to "
          f"'{OUTPUT_DIR}/' folder!")
