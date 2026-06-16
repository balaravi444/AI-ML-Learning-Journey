"""
ArthAI — Main Application
Smart Financial Advisor for Every Indian 🇮🇳

Author: Bala Ravi
Date: 16 June 2026
"""
from modules.financial_utils import (
    calculate_emi,
    total_interest_paid,
    future_value_sip,
    find_minimum_sip,
    smart_budget_plan,
    calculate_tax_savings,
    retirement_corpus_needed,
    loan_amortization,
    moving_average
)


def display_budget_plan(monthly_income: float) -> None:
    """Display smart budget plan."""
    plan = smart_budget_plan(monthly_income)

    print("\n" + "="*50)
    print("💰 ArthAI — SMART BUDGET PLAN")
    print("="*50)
    print(f"Monthly Income: ₹{plan['monthly_income']:,}")
    print()

    for category in ['needs', 'wants', 'savings']:
        data = plan[category]
        emoji = {'needs': '🏠', 'wants': '🎉',
                 'savings': '💹'}[category]
        print(f"{emoji} {category.upper()} "
              f"({data['percentage']}%) — "
              f"₹{data['amount']:,}")
        for item, amount in data['breakdown'].items():
            print(f"   • {item}: ₹{amount:,}")
        print()


def display_emi_analysis(
        principal: float,
        rate: float,
        years: int) -> None:
    """Display complete EMI analysis."""
    analysis = total_interest_paid(
        principal, rate, years)

    print("\n" + "="*50)
    print("🏦 ArthAI — LOAN EMI ANALYSIS")
    print("="*50)
    print(f"Loan Amount:    ₹{analysis['principal']:,}")
    print(f"Monthly EMI:    ₹{analysis['monthly_emi']:,}")
    print(f"Total Payment:  ₹{analysis['total_payment']:,}")
    print(f"Total Interest: ₹{analysis['total_interest']:,}")
    print(f"Interest %:     {analysis['interest_percentage']}%")
    print()

    # Show first 3 months
    schedule = loan_amortization(principal, rate, years)
    print("📊 Amortization (First 3 Months):")
    print(f"{'Month':>6} {'EMI':>10} "
          f"{'Principal':>10} {'Interest':>10} "
          f"{'Balance':>12}")
    print("-" * 52)
    for row in schedule[:3]:
        print(f"{row['month']:>6} "
              f"₹{row['emi']:>9,} "
              f"₹{row['principal_paid']:>9,} "
              f"₹{row['interest_paid']:>9,} "
              f"₹{row['balance']:>11,}")


def display_sip_analysis(
        target: float,
        rate: float,
        years: int) -> None:
    """Display SIP investment analysis."""
    min_sip = find_minimum_sip(target, rate, years)
    corpus = future_value_sip(min_sip, rate, years)

    print("\n" + "="*50)
    print("📈 ArthAI — SIP INVESTMENT PLANNER")
    print("="*50)
    print(f"Target Corpus:  ₹{target:,}")
    print(f"Expected Return: {rate}% annually")
    print(f"Duration:        {years} years")
    print(f"Min Monthly SIP: ₹{min_sip:,}")
    print(f"Expected Corpus: ₹{corpus:,.0f}")
    print()
    print("💡 SIP Comparison:")
    for sip in [min_sip, min_sip*2, min_sip*3]:
        c = future_value_sip(sip, rate, years)
        print(f"  ₹{sip:,}/month → ₹{c:,.0f}")


def display_retirement_plan(
        age: int,
        retire_age: int,
        expenses: float) -> None:
    """Display retirement planning analysis."""
    plan = retirement_corpus_needed(
        age, retire_age, expenses)

    print("\n" + "="*50)
    print("👴 ArthAI — RETIREMENT PLANNER")
    print("="*50)
    print(f"Current Age:        {plan['current_age']}")
    print(f"Retirement Age:     {plan['retirement_age']}")
    print(f"Years to Retire:    {plan['years_to_retire']}")
    print(f"Current Expenses:   ₹{plan['current_monthly_expenses']:,}/month")
    print(f"Future Expenses:    ₹{plan['future_monthly_expenses']:,}/month")
    print(f"Corpus Needed:      ₹{plan['corpus_needed']:,}")
    print(f"Monthly SIP Needed: ₹{plan['monthly_sip_required']:,}")
    print()
    print(f"💡 {plan['message']}")


def main() -> None:
    """ArthAI Main Demo."""
    print("\n" + "🏦"*25)
    print("       ARTHAI — SMART FINANCIAL ADVISOR")
    print("       For Every Indian 🇮🇳")
    print("🏦"*25)

    print("\n📌 Demo — Bala's Financial Profile")
    print("   Age: 22 | Income: ₹35,000/month")

    # 1. Budget Plan
    display_budget_plan(35000)

    # 2. Home Loan EMI
    display_emi_analysis(
        principal=2500000,  # ₹25L home loan
        rate=8.5,           # 8.5% interest
        years=20            # 20 years
    )

    # 3. SIP for retirement
    display_sip_analysis(
        target=10000000,  # ₹1 Crore
        rate=12,          # 12% returns
        years=30          # 30 years
    )

    # 4. Retirement Plan
    display_retirement_plan(
        age=22,
        retire_age=55,
        expenses=35000
    )

    print("\n" + "="*50)
    print("🚀 ArthAI — More features coming soon!")
    print("   Tax Saver | Goal Planner | AI Chatbot")
    print("="*50)


if __name__ == "__main__":
    main()
