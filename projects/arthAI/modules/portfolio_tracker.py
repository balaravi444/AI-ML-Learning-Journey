"""
ArthAI — Portfolio Tracker Module
Aggregate and analyze investments across asset classes.

Author: Bala Ravi
Date: 20 June 2026
"""
import math
from typing import TypedDict


class Holding(TypedDict):
    name: str
    asset_type: str
    invested_amount: float
    current_value: float
    risk_score: int


ASSET_RISK_SCORES = {
    "Fixed Deposit": 1,
    "PPF": 1,
    "EPF": 1,
    "Debt Mutual Fund": 3,
    "Balanced Fund": 5,
    "Equity Mutual Fund": 7,
    "Direct Stocks": 8,
    "Crypto": 10
}


def aggregate_portfolio(holdings: list[Holding]) -> dict:
    """
    Aggregate portfolio by asset type.

    Args:
        holdings: List of investment holdings

    Returns:
        Aggregated portfolio summary
    """
    asset_totals: dict[str, float] = {}
    total_invested = 0
    total_current = 0

    for holding in holdings:
        asset_type = holding['asset_type']
        asset_totals[asset_type] = (
            asset_totals.get(asset_type, 0) +
            holding['current_value'])
        total_invested += holding['invested_amount']
        total_current += holding['current_value']

    overall_return = (
        ((total_current - total_invested) /
         total_invested * 100)
        if total_invested > 0 else 0)

    return {
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current, 2),
        "overall_gain_loss": round(
            total_current - total_invested, 2),
        "overall_return_pct": round(overall_return, 2),
        "asset_breakdown": {
            k: round(v, 2)
            for k, v in asset_totals.items()
        }
    }


def diversification_score(
        asset_breakdown: dict[str, float]) -> dict:
    """
    Calculate portfolio diversification using Shannon Entropy.
    Same formula used in Decision Tree information gain!

    Args:
        asset_breakdown: Dict of asset_type -> value

    Returns:
        Diversification analysis
    """
    total = sum(asset_breakdown.values())

    if total == 0:
        return {"score": 0,
                "rating": "No investments",
                "max_score": 0}

    entropy = 0.0
    for amount in asset_breakdown.values():
        weight = amount / total
        if weight > 0:
            entropy -= weight * math.log2(weight)

    n = len(asset_breakdown)
    max_entropy = math.log2(n) if n > 1 else 1
    normalized_score = (entropy / max_entropy * 100
                        if max_entropy > 0 else 0)

    if normalized_score >= 80:
        rating = "Excellent diversification 🟢"
    elif normalized_score >= 50:
        rating = "Good diversification 🟡"
    elif normalized_score >= 25:
        rating = "Needs more diversification 🟠"
    else:
        rating = "High concentration risk! 🔴"

    return {
        "score": round(normalized_score, 1),
        "rating": rating,
        "entropy": round(entropy, 3)
    }


def calculate_portfolio_risk(
        holdings: list[Holding]) -> dict:
    """
    Calculate weighted average risk score.

    Args:
        holdings: List of investment holdings

    Returns:
        Risk analysis
    """
    total_value = sum(h['current_value'] for h in holdings)

    if total_value == 0:
        return {"risk_score": 0, "risk_level": "No data"}

    weighted_risk = sum(
        h['current_value'] * h['risk_score']
        for h in holdings) / total_value

    if weighted_risk <= 2:
        risk_level = "Conservative 🛡️"
    elif weighted_risk <= 4:
        risk_level = "Moderate-Conservative 🟢"
    elif weighted_risk <= 6:
        risk_level = "Moderate 🟡"
    elif weighted_risk <= 8:
        risk_level = "Aggressive 🟠"
    else:
        risk_level = "Very Aggressive 🔴"

    return {
        "risk_score": round(weighted_risk, 1),
        "risk_level": risk_level,
        "max_possible_risk": 10
    }


def analyze_trend(prices: list[float],
                  window: int = 7) -> dict:
    """
    Analyze price trend using sliding window.
    Callback to Day 29 sliding window pattern!

    Args:
        prices: Historical price data
        window: Window size for comparison

    Returns:
        Trend analysis
    """
    if len(prices) < window * 2:
        return {"trend": "Insufficient data",
                "change_pct": 0}

    recent_avg = sum(prices[-window:]) / window
    previous_avg = sum(
        prices[-window * 2:-window]) / window

    change_pct = (
        (recent_avg - previous_avg) / previous_avg * 100
        if previous_avg != 0 else 0)

    if change_pct > 2:
        trend = "Trending UP 📈"
    elif change_pct < -2:
        trend = "Trending DOWN 📉"
    else:
        trend = "STABLE ➡️"

    return {
        "trend": trend,
        "change_pct": round(change_pct, 2),
        "recent_average": round(recent_avg, 2),
        "previous_average": round(previous_avg, 2)
    }


def generate_portfolio_report(
        holdings: list[Holding]) -> dict:
    """
    Generate complete portfolio analysis report.

    Args:
        holdings: List of all investment holdings

    Returns:
        Comprehensive portfolio report
    """
    aggregated = aggregate_portfolio(holdings)
    diversification = diversification_score(
        aggregated['asset_breakdown'])
    risk = calculate_portfolio_risk(holdings)

    recommendations = []

    if diversification['score'] < 50:
        recommendations.append(
            "Consider diversifying across more asset types "
            "to reduce concentration risk")

    if risk['risk_score'] > 7:
        recommendations.append(
            "Your portfolio is high-risk. Consider adding "
            "some debt funds or FDs for stability")
    elif risk['risk_score'] < 3:
        recommendations.append(
            "Your portfolio is very conservative. Consider "
            "some equity exposure for better long-term growth")

    if aggregated['overall_return_pct'] < 0:
        recommendations.append(
            "Your portfolio is currently in loss. Avoid "
            "panic selling — review your investment horizon")

    if not recommendations:
        recommendations.append(
            "Your portfolio looks well-balanced! Keep "
            "monitoring regularly")

    return {
        "summary": aggregated,
        "diversification": diversification,
        "risk_analysis": risk,
        "recommendations": recommendations
    }
