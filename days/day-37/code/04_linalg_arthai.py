"""
Day 37 — NumPy: Linear Algebra
Topic: Linear Algebra Applied to ArthAI
Date: 24 June 2026
Author: Bala Ravi

Upgrade ArthAI calculations with linear algebra!
Portfolio optimization using matrix operations!
"""
import numpy as np


def portfolio_covariance_matrix(
        returns: np.ndarray) -> np.ndarray:
    """
    Calculate covariance matrix for portfolio assets.
    Foundation of Modern Portfolio Theory!

    Args:
        returns: Matrix of historical returns
                 (n_months, n_assets)

    Returns:
        Covariance matrix (n_assets, n_assets)
    """
    return np.cov(returns.T)


def portfolio_variance(
        weights: np.ndarray,
        cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio variance using matrix math.
    Markowitz Portfolio Theory!

    Formula: σ² = w.T @ Σ @ w

    Args:
        weights: Asset weights (n_assets,)
        cov_matrix: Covariance matrix (n_assets, n_assets)

    Returns:
        Portfolio variance
    """
    return float(weights.T @ cov_matrix @ weights)


def portfolio_return(
        weights: np.ndarray,
        expected_returns: np.ndarray) -> float:
    """
    Calculate expected portfolio return.

    Formula: E[R] = w.T @ r

    Args:
        weights: Asset weights
        expected_returns: Expected return per asset

    Returns:
        Expected portfolio return
    """
    return float(weights.T @ expected_returns)


def find_optimal_portfolio(
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_tolerance: float = 0.5) -> dict:
    """
    Find optimal portfolio weights using linear algebra!
    Uses matrix operations from Markowitz theory.

    Args:
        expected_returns: Expected returns per asset
        cov_matrix: Covariance matrix
        risk_tolerance: 0=min risk, 1=max return

    Returns:
        Optimal portfolio allocation
    """
    n_assets = len(expected_returns)

    # Monte Carlo simulation for portfolio optimization
    best_portfolio = None
    best_score = -np.inf

    np.random.seed(42)

    for _ in range(10000):
        # Random weights summing to 1
        weights = np.random.dirichlet(
            np.ones(n_assets))

        ret = portfolio_return(
            weights, expected_returns)
        var = portfolio_variance(weights, cov_matrix)
        std = np.sqrt(var)

        # Risk-adjusted score
        score = (risk_tolerance * ret -
                 (1 - risk_tolerance) * std)

        if score > best_score:
            best_score = score
            best_portfolio = {
                "weights": weights,
                "expected_return": ret,
                "volatility": std,
                "sharpe_ratio": ret / std if std > 0 else 0
            }

    return best_portfolio


def demonstrate_portfolio_optimization() -> None:
    """Full portfolio optimization demo for ArthAI."""
    print("=== ArthAI Portfolio Optimization ===\n")

    np.random.seed(42)

    assets = [
        "Equity Fund",
        "Debt Fund",
        "Gold",
        "Fixed Deposit"
    ]

    # Historical monthly returns (%)
    n_months = 60
    returns = np.array([
        np.random.normal(1.2, 4.5, n_months),  # Equity
        np.random.normal(0.6, 1.2, n_months),  # Debt
        np.random.normal(0.8, 2.5, n_months),  # Gold
        np.random.normal(0.5, 0.1, n_months),  # FD
    ]).T

    print(f"Historical data: {n_months} months, "
          f"{len(assets)} assets\n")

    expected_returns = returns.mean(axis=0) * 12
    cov_matrix = portfolio_covariance_matrix(returns) * 12

    print("Expected Annual Returns:")
    for asset, ret in zip(assets, expected_returns):
        print(f"  {asset}: {ret:.1f}%")

    print("\nCorrelation Matrix:")
    corr = np.corrcoef(returns.T)
    print(f"{'':>15}", end="")
    for a in assets:
        print(f"{a[:8]:>10}", end="")
    print()
    for i, asset in enumerate(assets):
        print(f"{asset[:15]:>15}", end="")
        for j in range(len(assets)):
            print(f"{corr[i,j]:>10.3f}", end="")
        print()

    print("\n=== Optimal Portfolios ===")
    for risk_name, risk_val in [
        ("Conservative (min risk)", 0.2),
        ("Balanced", 0.5),
        ("Aggressive (max return)", 0.8)
    ]:
        portfolio = find_optimal_portfolio(
            expected_returns, cov_matrix, risk_val)

        print(f"\n{risk_name}:")
        for asset, weight in zip(
                assets, portfolio['weights']):
            print(f"  {asset}: {weight*100:.1f}%")
        print(f"  Expected Return: "
              f"{portfolio['expected_return']:.1f}%")
        print(f"  Volatility: "
              f"{portfolio['volatility']:.1f}%")
        print(f"  Sharpe Ratio: "
              f"{portfolio['sharpe_ratio']:.3f}")


if __name__ == "__main__":
    demonstrate_portfolio_optimization()
