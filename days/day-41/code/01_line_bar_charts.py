"""
Day 41 — Matplotlib: Data Visualization
Topic: Line Charts and Bar Charts
Date: 28 June 2026
Author: Bala Ravi

Matplotlib = Python's core visualization library!
Foundation of ALL data science visualization!
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import numpy as np
import os

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def line_chart_sip_growth() -> None:
    """
    Line chart showing SIP corpus growth over time.
    ArthAI financial visualization!
    """
    def future_value_sip(sip: float,
                          rate: float,
                          years: int) -> float:
        r = rate / 12 / 100
        n = years * 12
        return sip * (((1 + r)**n - 1) / r) * (1 + r)

    years = np.arange(1, 31)

    # Multiple SIP amounts
    sip_amounts = [2000, 5000, 10000, 20000]
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
    labels = ['₹2,000/mo', '₹5,000/mo',
               '₹10,000/mo', '₹20,000/mo']

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    for sip, color, label in zip(
            sip_amounts, colors, labels):
        corpus = [future_value_sip(sip, 12, y)
                  for y in years]
        ax.plot(years, corpus, color=color,
                linewidth=2.5, marker='o',
                markersize=4, label=label)

    # Format y-axis in lakhs/crores
    def format_currency(x, pos):
        if x >= 10000000:
            return f'₹{x/10000000:.1f}Cr'
        elif x >= 100000:
            return f'₹{x/100000:.0f}L'
        return f'₹{x:,.0f}'

    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(
        FuncFormatter(format_currency))

    ax.set_title('SIP Growth at 12% Annual Returns',
                 color='white', fontsize=16, pad=15)
    ax.set_xlabel('Years', color='#94a3b8')
    ax.set_ylabel('Corpus Value', color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#334155', labelcolor='white',
              fontsize=10)
    ax.grid(True, alpha=0.2, color='#64748b')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/sip_growth.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


def bar_chart_salary_by_city() -> None:
    """Bar chart of average salary by city."""
    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune', 'Chennai']
    salaries = [24.2, 22.8, 20.1, 21.5, 18.9, 19.2]
    colors = ['#10b981' if s == max(salaries)
              else '#334155' for s in salaries]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.bar(cities, salaries, color=colors,
                  edgecolor='#0f172a', linewidth=0.5,
                  width=0.6)

    for bar, salary in zip(bars, salaries):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'₹{salary}L',
                ha='center', va='bottom',
                color='white', fontweight='bold',
                fontsize=11)

    ax.set_title('Average AI/ML Salary by City (₹ LPA)',
                 color='white', fontsize=15, pad=15)
    ax.set_xlabel('City', color='#94a3b8')
    ax.set_ylabel('Average Salary (₹ LPA)',
                  color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.set_ylim(0, max(salaries) * 1.2)
    ax.grid(True, axis='y', alpha=0.2, color='#64748b')

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/salary_by_city.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


def horizontal_bar_skills() -> None:
    """Horizontal bar chart of top skills."""
    skills = ['Python', 'Machine Learning', 'SQL',
              'TensorFlow', 'Deep Learning', 'NLP',
              'PyTorch', 'AWS', 'Docker', 'Statistics']
    demand_pct = [78, 65, 52, 43, 38, 35, 32, 28, 25, 22]

    colors = ['#10b981' if p >= 50 else
              '#3b82f6' if p >= 35 else
              '#64748b' for p in demand_pct]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    bars = ax.barh(skills, demand_pct, color=colors,
                   edgecolor='#0f172a', height=0.6)

    for bar, pct in zip(bars, demand_pct):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{pct}%', va='center',
                color='white', fontweight='bold')

    ax.set_title('Most In-Demand Skills in AI/ML Jobs',
                 color='white', fontsize=15, pad=15)
    ax.set_xlabel('% of Job Postings',
                  color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.set_xlim(0, 95)
    ax.grid(True, axis='x', alpha=0.2, color='#64748b')
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/skills_demand.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    print("=== Line & Bar Charts ===\n")
    line_chart_sip_growth()
    bar_chart_salary_by_city()
    horizontal_bar_skills()
    print("\nAll charts saved to 'charts/' folder!")
