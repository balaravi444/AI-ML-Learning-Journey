"""
ArthAI — Financial Utility Functions
Core mathematical calculations for all modules.

Uses DSA concepts learned:
- Binary Search (find optimal SIP)
- Dynamic Programming (loan optimization)
- Sliding Window (moving averages)
- Prefix Sum (cumulative returns)

Author: Bala Ravi
Date: 16 June 2026
"""
import math
from typing import Union


def future_value_sip(
        monthly_sip: float,
        annual_rate: float,
        years: int) -> float:
    """
    Calculate future value of monthly SIP.

    Formula: FV = P * [((1+r)^n - 1) / r] * (1+r)

    Args:
        monthly_sip: Monthly investment amount
        annual_rate: Expected annual return %
        years: Investment duration in years

    Returns:
        Future corpus value

    Example:
        ₹5000/month for 20 years at 12% = ₹49,95,740
    """
    monthly_rate = annual_rate / 12 / 100
    months = years * 12

    if monthly_rate == 0:
        return monthly_sip * months

    fv = (monthly_sip *
          (((1 + monthly_rate) ** months - 1) /
           monthly_rate) *
          (1 + monthly_rate))

    return round(fv, 2)


def find_minimum_sip(
        target_corpus: float,
        annual_rate: float,
        years: int) -> int:
    """
    Find minimum monthly SIP to reach target.
    Uses BINARY SEARCH — DSA concept! 🔥

    Args:
        target_corpus: Target amount in rupees
        annual_rate: Expected annual return %
        years: Investment duration

    Returns:
        Minimum monthly SIP amount
    """
    left, right = 100, 1000000

    while left < right:
        mid = (left + right) // 2
        corpus = future_value_sip(mid,
                                   annual_rate,
                                   years)
        if corpus >= target_corpus:
            right = mid
        else:
            left = mid + 1

    return left


def calculate_emi(
        principal: float,
        annual_rate: float,
        years: int) -> float:
    """
    Calculate monthly EMI for a loan.

    Formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate %
        years: Loan duration in years

    Returns:
        Monthly EMI amount
    """
    monthly_rate = annual_rate / 12 / 100
    months = years * 12

    if monthly_rate == 0:
        return round(principal / months, 2)

    emi = (principal *
           monthly_rate *
           (1 + monthly_rate) ** months /
           ((1 + monthly_rate) ** months - 1))

    return round(emi, 2)


def loan_amortization(
        principal: float,
        annual_rate: float,
        years: int) -> list[dict]:
    """
    Generate complete loan amortization schedule.
    Shows principal vs interest split every month!

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate %
        years: Loan duration

    Returns:
        Monthly payment breakdown
    """
    monthly_rate = annual_rate / 12 / 100
    emi = calculate_emi(principal, annual_rate, years)
    balance = principal
    schedule = []

    for month in range(1, years * 12 + 1):
        interest = round(balance * monthly_rate, 2)
        principal_paid = round(emi - interest, 2)
        balance = round(balance - principal_paid, 2)

        if balance < 0:
            balance = 0

        schedule.append({
            "month": month,
            "emi": emi,
            "principal_paid": principal_paid,
            "interest_paid": interest,
            "balance": max(0, balance)
        })

    return schedule


def total_interest_paid(
        principal: float,
        annual_rate: float,
        years: int) -> dict:
    """
    Calculate total interest paid on loan.

    Args:
        principal: Loan amount
        annual_rate: Annual interest rate %
        years: Loan duration

    Returns:
        Loan cost breakdown
    """
    emi = calculate_emi(principal, annual_rate, years)
    total_payment = emi * years * 12
    total_interest = total_payment - principal

    return {
        "principal": round(principal, 2),
        "monthly_emi": emi,
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "interest_percentage": round(
            total_interest / principal * 100, 1)
    }


def calculate_tax_savings(
        annual_income: float,
        investments: dict) -> dict:
    """
    Calculate tax savings under Indian tax laws.
    80C, 80D, HRA, NPS deductions!

    Args:
        annual_income: Annual income in rupees
        investments: Dict of investment amounts

    Returns:
        Tax saving analysis
    """
    # Standard deduction
    standard_deduction = 50000

    # 80C limit (ELSS, PPF, LIC, EPF, etc)
    section_80c = min(
        investments.get('80c_investments', 0),
        150000)

    # 80D health insurance
    section_80d = min(
        investments.get('health_insurance', 0),
        25000)

    # NPS additional deduction
    nps = min(
        investments.get('nps', 0),
        50000)

    total_deductions = (standard_deduction +
                        section_80c +
                        section_80d +
                        nps)

    taxable_income = max(0,
                         annual_income - total_deductions)

    # New tax regime slabs (FY 2024-25)
    tax = calculate_income_tax(taxable_income)
    tax_without_deductions = calculate_income_tax(
        annual_income - standard_deduction)

    return {
        "annual_income": annual_income,
        "total_deductions": total_deductions,
        "taxable_income": taxable_income,
        "tax_payable": tax,
        "tax_saved": tax_without_deductions - tax,
        "effective_tax_rate": round(
            tax / annual_income * 100, 1)
    }


def calculate_income_tax(income: float) -> float:
    """
    Calculate income tax as per Indian tax slabs.
    New regime FY 2024-25.

    Args:
        income: Taxable income

    Returns:
        Tax amount
    """
    if income <= 300000:
        return 0
    elif income <= 600000:
        return (income - 300000) * 0.05
    elif income <= 900000:
        return 15000 + (income - 600000) * 0.10
    elif income <= 1200000:
        return 45000 + (income - 900000) * 0.15
    elif income <= 1500000:
        return 90000 + (income - 1200000) * 0.20
    else:
        return 150000 + (income - 1500000) * 0.30


def smart_budget_plan(
        monthly_income: float) -> dict:
    """
    Create smart budget plan using 50/30/20 rule.
    Customized for Indian income levels!

    Args:
        monthly_income: Monthly take-home income

    Returns:
        Detailed budget breakdown
    """
    # Indian-specific budget allocation
    if monthly_income <= 25000:
        # Lower income — survival focused
        needs_pct = 60
        wants_pct = 20
        savings_pct = 20
    elif monthly_income <= 75000:
        # Middle income — balanced
        needs_pct = 50
        wants_pct = 30
        savings_pct = 20
    else:
        # Higher income — wealth building
        needs_pct = 40
        wants_pct = 30
        savings_pct = 30

    needs = monthly_income * needs_pct / 100
    wants = monthly_income * wants_pct / 100
    savings = monthly_income * savings_pct / 100

    return {
        "monthly_income": monthly_income,
        "needs": {
            "amount": round(needs),
            "percentage": needs_pct,
            "breakdown": {
                "rent": round(needs * 0.40),
                "groceries": round(needs * 0.25),
                "utilities": round(needs * 0.10),
                "transport": round(needs * 0.15),
                "insurance": round(needs * 0.10)
            }
        },
        "wants": {
            "amount": round(wants),
            "percentage": wants_pct,
            "breakdown": {
                "dining_out": round(wants * 0.30),
                "entertainment": round(wants * 0.25),
                "shopping": round(wants * 0.25),
                "subscriptions": round(wants * 0.20)
            }
        },
        "savings": {
            "amount": round(savings),
            "percentage": savings_pct,
            "breakdown": {
                "emergency_fund": round(savings * 0.30),
                "investments": round(savings * 0.50),
                "retirement": round(savings * 0.20)
            }
        }
    }


def retirement_corpus_needed(
        current_age: int,
        retirement_age: int,
        monthly_expenses: float,
        inflation_rate: float = 6.0,
        post_retirement_years: int = 25) -> dict:
    """
    Calculate retirement corpus needed.
    Inflation-adjusted calculation!

    Args:
        current_age: Current age
        retirement_age: Target retirement age
        monthly_expenses: Current monthly expenses
        inflation_rate: Expected annual inflation %
        post_retirement_years: Years after retirement

    Returns:
        Retirement planning details
    """
    years_to_retire = retirement_age - current_age

    # Future monthly expenses (inflation adjusted)
    future_monthly = (monthly_expenses *
                      (1 + inflation_rate/100) **
                      years_to_retire)

    # Total corpus needed (25x annual expenses rule)
    annual_expenses = future_monthly * 12
    corpus_needed = annual_expenses * post_retirement_years

    # Monthly SIP needed (assuming 12% returns)
    monthly_sip = find_minimum_sip(
        corpus_needed, 12, years_to_retire)

    return {
        "current_age": current_age,
        "retirement_age": retirement_age,
        "years_to_retire": years_to_retire,
        "current_monthly_expenses": monthly_expenses,
        "future_monthly_expenses": round(future_monthly),
        "corpus_needed": round(corpus_needed),
        "monthly_sip_required": monthly_sip,
        "message": (
            f"Start SIP of ₹{monthly_sip:,}/month "
            f"today to retire at {retirement_age}!"
        )
    }


def moving_average(values: list[float],
                   window: int) -> list[float]:
    """
    Calculate moving average using sliding window.
    Used for stock price analysis in ArthAI!

    Args:
        values: List of values
        window: Window size

    Returns:
        Moving averages
    """
    if len(values) < window:
        return []

    window_sum = sum(values[:window])
    averages = [round(window_sum / window, 2)]

    for i in range(window, len(values)):
        window_sum += values[i] - values[i - window]
        averages.append(round(window_sum / window, 2))

    return averages
