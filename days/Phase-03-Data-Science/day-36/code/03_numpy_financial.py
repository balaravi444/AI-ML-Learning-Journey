"""
Day 36 — NumPy: Arrays & Operations
Topic: Financial Analysis with NumPy
Date: 23 June 2026
Author: Bala Ravi

ArthAI calculations — now vectorized with NumPy!
Process thousands of loan/SIP calculations at once!

Real World Connection:
    Banks process millions of loan applications!
    NumPy vectorization makes this fast enough!
"""
import numpy as np


def calculate_emi_batch(
        principals: np.ndarray,
        annual_rates: np.ndarray,
        years: np.ndarray) -> np.ndarray:
    """
    Calculate EMI for multiple loans simultaneously!
    Vectorized — no loops needed!

    Args:
        principals: Loan amounts array
        annual_rates: Interest rates array
        years: Loan durations array

    Returns:
        Monthly EMI amounts array

    Time Complexity: O(n) vectorized — very fast!
    """
    monthly_rates = annual_rates / 12 / 100
    months = years * 12
    emi = (principals * monthly_rates *
           (1 + monthly_rates) ** months /
           ((1 + monthly_rates) ** months - 1))
    return np.round(emi, 2)


def calculate_sip_corpus_batch(
        monthly_sips: np.ndarray,
        annual_rates: np.ndarray,
        years: np.ndarray) -> np.ndarray:
    """
    Calculate SIP corpus for multiple scenarios.
    Vectorized batch calculation!

    Args:
        monthly_sips: Monthly SIP amounts
        annual_rates: Expected return rates
        years: Investment durations

    Returns:
        Expected corpus for each scenario
    """
    monthly_rates = annual_rates / 12 / 100
    months = years * 12
    corpus = (monthly_sips *
              (((1 + monthly_rates) ** months - 1) /
               monthly_rates) *
              (1 + monthly_rates))
    return np.round(corpus, 2)


def portfolio_statistics(
        returns: np.ndarray) -> dict:
    """
    Calculate portfolio statistics using NumPy.

    Args:
        returns: Array of monthly returns (%)

    Returns:
        Statistical summary of portfolio
    """
    return {
        "mean_return": round(float(np.mean(returns)), 3),
        "std_deviation": round(float(np.std(returns)), 3),
        "sharpe_ratio": round(
            float(np.mean(returns) / np.std(returns)), 3),
        "max_return": round(float(np.max(returns)), 3),
        "min_return": round(float(np.min(returns)), 3),
        "positive_months": int(np.sum(returns > 0)),
        "negative_months": int(np.sum(returns < 0)),
        "best_streak": _max_consecutive(returns > 0),
        "worst_streak": _max_consecutive(returns < 0)
    }


def _max_consecutive(mask: np.ndarray) -> int:
    """Find maximum consecutive True values."""
    max_streak = current = 0
    for val in mask:
        if val:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def loan_comparison_analysis(
        loan_amount: float,
        rates: np.ndarray,
        tenure_years: int) -> None:
    """
    Compare EMI and total interest across banks.
    Shows power of vectorized batch processing!

    Args:
        loan_amount: Loan amount in rupees
        rates: Array of interest rates to compare
        tenure_years: Loan duration
    """
    principals = np.full_like(rates, loan_amount)
    years = np.full_like(rates, tenure_years)

    emis = calculate_emi_batch(principals, rates, years)
    total_payments = emis * tenure_years * 12
    total_interests = total_payments - loan_amount

    print(f"\n=== Loan Comparison Analysis ===")
    print(f"Loan: ₹{loan_amount:,.0f} for {tenure_years} years\n")
    print(f"{'Bank Rate':>10} | {'Monthly EMI':>12} | "
          f"{'Total Interest':>15} | {'Total Payment':>13}")
    print("-" * 57)

    for rate, emi, interest, total in zip(
            rates, emis, total_interests, total_payments):
        print(f"{rate:>9.1f}% | ₹{emi:>11,.0f} | "
              f"₹{interest:>14,.0f} | ₹{total:>12,.0f}")


def sip_goal_analysis(
        target: float,
        rate: float,
        year_range: np.ndarray) -> None:
    """
    Show how SIP amount changes with time horizon.

    Args:
        target: Target corpus
        rate: Expected annual return %
        year_range: Array of years to analyze
    """
    print(f"\n=== SIP Analysis for ₹{target:,.0f} ===")
    print(f"Expected return: {rate}% annually\n")

    monthly_rate = rate / 12 / 100

    def min_sip_vectorized(years_arr):
        months = years_arr * 12
        factor = (((1 + monthly_rate) ** months - 1) /
                  monthly_rate * (1 + monthly_rate))
        return np.ceil(target / factor).astype(int)

    sips = min_sip_vectorized(year_range)

    print(f"{'Years':>8} | {'Monthly SIP':>12} | "
          f"{'Total Invested':>15}")
    print("-" * 40)

    for years, sip in zip(year_range, sips):
        total = sip * years * 12
        print(f"{years:>8} | ₹{sip:>11,} | "
              f"₹{total:>14,}")


if __name__ == "__main__":
    print("=== ArthAI Financial Analysis with NumPy ===\n")

    # Loan comparison across banks
    rates = np.array([7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
    loan_comparison_analysis(2500000, rates, 20)

    # SIP analysis
    year_range = np.arange(5, 31, 5)
    sip_goal_analysis(10000000, 12.0, year_range)

    # Portfolio statistics
    print("\n=== Portfolio Statistics ===")
    np.random.seed(42)
    monthly_returns = np.random.normal(1.2, 3.5, 60)
    stats = portfolio_statistics(monthly_returns)

    for key, value in stats.items():
        print(f"  {key}: {value}")
