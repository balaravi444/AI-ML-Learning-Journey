"""
Day 41 — Matplotlib: Data Visualization
Topic: Subplots — Multi-chart Dashboard
Date: 28 June 2026
Author: Bala Ravi

Subplots = Multiple charts in one figure!
Used to create professional dashboards!
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def job_market_dashboard() -> None:
    """
    Complete Indian Job Market Dashboard.
    Combines all chart types in one figure!
    This is the foundation of the Job Market Analyzer!
    """
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer',
             'Data Engineer']
    city_salaries = [24.2, 22.8, 20.1, 21.5, 18.9]
    role_salaries = [25.1, 23.4, 15.2, 26.8, 19.5]
    role_counts = [45, 38, 62, 28, 35]

    experience = np.random.randint(0, 12, 200)
    salary = np.clip(
        8 + experience * 1.8 +
        np.random.normal(0, 3, 200), 5, 55)

    skills = ['Python', 'ML', 'SQL', 'TF',
              'DL', 'NLP', 'PyTorch']
    skill_pct = [78, 65, 52, 43, 38, 35, 32]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f172a')

    fig.suptitle(
        '🇮🇳 Indian AI/ML Job Market Dashboard 2026',
        color='white', fontsize=18,
        fontweight='bold', y=0.98)

    def style_ax(ax):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.grid(True, alpha=0.2, color='#64748b')

    # Chart 1 — Salary by City (top-left)
    ax1 = fig.add_subplot(2, 3, 1)
    style_ax(ax1)
    colors1 = ['#10b981' if s == max(city_salaries)
               else '#334155' for s in city_salaries]
    bars = ax1.bar(cities, city_salaries,
                   color=colors1, edgecolor='#0f172a')
    for bar, s in zip(bars, city_salaries):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f'₹{s}L', ha='center',
                 color='white', fontsize=8,
                 fontweight='bold')
    ax1.set_title('Avg Salary by City (₹ LPA)',
                  color='white', fontsize=11, pad=8)
    ax1.set_ylim(0, 30)
    ax1.tick_params(axis='x', rotation=15)

    # Chart 2 — Jobs by Role (top-center)
    ax2 = fig.add_subplot(2, 3, 2)
    style_ax(ax2)
    wedge_colors = ['#10b981', '#3b82f6', '#f59e0b',
                    '#ef4444', '#8b5cf6']
    wedges, texts, autotexts = ax2.pie(
        role_counts, labels=roles,
        colors=wedge_colors,
        autopct='%1.0f%%',
        pctdistance=0.75,
        startangle=90)
    for text in texts:
        text.set_color('white')
        text.set_fontsize(8)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)
    ax2.set_title('Job Distribution by Role',
                  color='white', fontsize=11, pad=8)

    # Chart 3 — Skills Demand (top-right)
    ax3 = fig.add_subplot(2, 3, 3)
    style_ax(ax3)
    skill_colors = ['#10b981' if p >= 50 else
                    '#3b82f6' if p >= 35 else
                    '#64748b' for p in skill_pct]
    ax3.barh(skills, skill_pct,
             color=skill_colors,
             edgecolor='#0f172a')
    for i, pct in enumerate(skill_pct):
        ax3.text(pct + 0.5, i, f'{pct}%',
                 va='center', color='white',
                 fontsize=9, fontweight='bold')
    ax3.set_title('Top Skills Demand (%)',
                  color='white', fontsize=11, pad=8)
    ax3.set_xlim(0, 95)
    ax3.invert_yaxis()

    # Chart 4 — Experience vs Salary (bottom-left)
    ax4 = fig.add_subplot(2, 3, 4)
    style_ax(ax4)
    ax4.scatter(experience, salary, alpha=0.5,
                color='#10b981', s=25,
                edgecolors='none')
    z = np.polyfit(experience, salary, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 12, 100)
    ax4.plot(x_line, p(x_line), color='#ef4444',
             linewidth=2, linestyle='--',
             label='Trend')
    ax4.set_title('Experience vs Salary',
                  color='white', fontsize=11, pad=8)
    ax4.set_xlabel('Experience (yrs)',
                   color='#94a3b8', fontsize=9)
    ax4.set_ylabel('Salary (₹ LPA)',
                   color='#94a3b8', fontsize=9)
    ax4.legend(facecolor='#334155',
               labelcolor='white', fontsize=8)

    # Chart 5 — Salary Distribution (bottom-center)
    ax5 = fig.add_subplot(2, 3, 5)
    style_ax(ax5)
    ax5.hist(salary, bins=20, color='#3b82f6',
             edgecolor='#0f172a', alpha=0.8)
    ax5.axvline(np.mean(salary), color='#ef4444',
                linestyle='--', linewidth=2,
                label=f'Mean: ₹{np.mean(salary):.1f}L')
    ax5.axvline(np.median(salary),
                color='#f59e0b',
                linestyle='--', linewidth=2,
                label=f'Median: ₹{np.median(salary):.1f}L')
    ax5.set_title('Salary Distribution',
                  color='white', fontsize=11, pad=8)
    ax5.set_xlabel('Salary (₹ LPA)',
                   color='#94a3b8', fontsize=9)
    ax5.legend(facecolor='#334155',
               labelcolor='white', fontsize=8)

    # Chart 6 — Role Salary Comparison (bottom-right)
    ax6 = fig.add_subplot(2, 3, 6)
    style_ax(ax6)
    short_roles = ['DS', 'MLE', 'DA', 'AIE', 'DE']
    role_colors = ['#10b981' if s == max(role_salaries)
                   else '#334155'
                   for s in role_salaries]
    bars6 = ax6.bar(short_roles, role_salaries,
                    color=role_colors,
                    edgecolor='#0f172a')
    for bar, s in zip(bars6, role_salaries):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f'₹{s}L', ha='center',
                 color='white', fontsize=8,
                 fontweight='bold')
    ax6.set_title('Avg Salary by Role (₹ LPA)',
                  color='white', fontsize=11, pad=8)
    ax6.set_ylim(0, 32)

    # Legend for role abbreviations
    legend_text = ("DS=Data Scientist, "
                   "MLE=ML Engineer,\n"
                   "DA=Data Analyst, "
                   "AIE=AI Engineer, "
                   "DE=Data Engineer")
    fig.text(0.5, 0.01, legend_text,
             ha='center', color='#64748b',
             fontsize=8)

    plt.subplots_adjust(
        hspace=0.4, wspace=0.35,
        top=0.93, bottom=0.06)

    path = f"{OUTPUT_DIR}/job_market_dashboard.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")
    print(f"   → This is the Job Market Analyzer "
          f"dashboard! 🚀")


def arthai_financial_dashboard() -> None:
    """
    ArthAI Financial Dashboard using Matplotlib.
    Combines EMI, SIP, budget visualizations!
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('🏦 ArthAI — Financial Dashboard',
                 color='white', fontsize=16,
                 fontweight='bold', y=0.98)

    def style_ax(ax):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
        ax.grid(True, alpha=0.2, color='#64748b')

    # Chart 1 — SIP Growth
    ax1 = fig.add_subplot(2, 2, 1)
    style_ax(ax1)
    years = np.arange(1, 26)

    def fv_sip(sip, rate, yr):
        r = rate / 12 / 100
        n = yr * 12
        return sip * (((1+r)**n - 1)/r) * (1+r)

    for sip, color, label in [
        (3000, '#64748b', '₹3K/mo'),
        (5000, '#3b82f6', '₹5K/mo'),
        (10000, '#10b981', '₹10K/mo')
    ]:
        corpus = [fv_sip(sip, 12, y) / 100000
                  for y in years]
        ax1.plot(years, corpus, color=color,
                 linewidth=2, label=label)
    ax1.set_title('SIP Growth (₹ Lakhs at 12%)',
                  color='white', fontsize=11, pad=8)
    ax1.set_xlabel('Years', color='#94a3b8',
                   fontsize=9)
    ax1.set_ylabel('Corpus (₹ Lakhs)',
                   color='#94a3b8', fontsize=9)
    ax1.legend(facecolor='#334155',
               labelcolor='white', fontsize=9)

    # Chart 2 — EMI Breakdown
    ax2 = fig.add_subplot(2, 2, 2)
    style_ax(ax2)
    months = np.arange(1, 13)
    principal_paid = [12500, 12800, 13100,
                      13400, 13700, 14000,
                      14300, 14600, 14900,
                      15200, 15500, 15800]
    interest_paid = [17700, 17400, 17100,
                     16800, 16500, 16200,
                     15900, 15600, 15300,
                     15000, 14700, 14400]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr',
                    'May', 'Jun', 'Jul', 'Aug',
                    'Sep', 'Oct', 'Nov', 'Dec']
    ax2.bar(month_labels, principal_paid,
            label='Principal', color='#10b981',
            edgecolor='#0f172a')
    ax2.bar(month_labels, interest_paid,
            bottom=principal_paid,
            label='Interest', color='#ef4444',
            alpha=0.8, edgecolor='#0f172a')
    ax2.set_title('EMI Breakdown (Principal vs Interest)',
                  color='white', fontsize=11, pad=8)
    ax2.set_xlabel('Month', color='#94a3b8',
                   fontsize=9)
    ax2.set_ylabel('Amount (₹)', color='#94a3b8',
                   fontsize=9)
    ax2.legend(facecolor='#334155',
               labelcolor='white', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)

    # Chart 3 — Budget Allocation
    ax3 = fig.add_subplot(2, 2, 3)
    style_ax(ax3)
    categories = ['Rent', 'Food', 'Transport',
                  'Utilities', 'Insurance']
    needs_amounts = [14000, 8750, 5250, 3500, 3500]
    wants_cats = ['Dining', 'Entertainment',
                  'Shopping', 'Subscriptions']
    wants_amounts = [3150, 2625, 2625, 2100]
    savings_cats = ['Emergency', 'Invest', 'Retire']
    savings_amounts = [2100, 3500, 1400]

    all_cats = (categories + wants_cats +
                savings_cats)
    all_amounts = (needs_amounts + wants_amounts +
                   savings_amounts)
    all_colors = (['#10b981'] * 5 +
                  ['#3b82f6'] * 4 +
                  ['#f59e0b'] * 3)

    wedges, texts = ax3.pie(
        all_amounts, labels=all_cats,
        colors=all_colors,
        startangle=90,
        wedgeprops={'edgecolor': '#0f172a',
                    'linewidth': 0.5})
    for text in texts:
        text.set_color('white')
        text.set_fontsize(7)
    ax3.set_title('Budget Allocation (₹35K Income)',
                  color='white', fontsize=11, pad=8)

    # Chart 4 — Tax Savings
    ax4 = fig.add_subplot(2, 2, 4)
    style_ax(ax4)
    deduction_types = ['Standard\nDeduction',
                       '80C\nELSS/PPF',
                       '80D\nHealth Ins',
                       'NPS\n80CCD']
    deduction_amounts = [50000, 150000, 25000, 50000]
    deduction_colors = ['#64748b', '#10b981',
                        '#3b82f6', '#f59e0b']
    bars4 = ax4.bar(deduction_types, deduction_amounts,
                    color=deduction_colors,
                    edgecolor='#0f172a')
    for bar, amt in zip(bars4, deduction_amounts):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1000,
                 f'₹{amt//1000}K',
                 ha='center', color='white',
                 fontsize=9, fontweight='bold')
    ax4.set_title('Tax Deductions Available',
                  color='white', fontsize=11, pad=8)
    ax4.set_ylabel('Amount (₹)', color='#94a3b8',
                   fontsize=9)
    total = sum(deduction_amounts)
    ax4.text(0.5, 0.92,
             f'Total: ₹{total//1000}K saved!',
             transform=ax4.transAxes,
             ha='center', color='#10b981',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round',
                       facecolor='#064e3b',
                       alpha=0.8))

    plt.subplots_adjust(
        hspace=0.45, wspace=0.35,
        top=0.92, bottom=0.05)

    path = f"{OUTPUT_DIR}/arthai_dashboard.png"
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"✅ Saved: {path}")
    print(f"   → ArthAI financial dashboard! 💰")


if __name__ == "__main__":
    print("=== Dashboard Charts ===\n")
    job_market_dashboard()
    arthai_financial_dashboard()
    print(f"\n🎨 All dashboards saved to 'charts/'!")
    print(f"   Open them to see your visualizations!")
