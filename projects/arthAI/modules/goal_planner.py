"""
ArthAI — Goal Planner Module
Multi-goal financial optimization using Knapsack DP!

Author: Bala Ravi
Date: 18 June 2026
"""
from typing import TypedDict
from modules.financial_utils import find_minimum_sip


class FinancialGoal(TypedDict):
    name: str
    target_amount: float
    years: int
    priority: int
    monthly_required: float


def prioritize_goals(
        goals: list[FinancialGoal]) -> list[FinancialGoal]:
    """
    Sort goals by priority and urgency.

    Args:
        goals: List of financial goals

    Returns:
        Goals sorted by importance
    """
    def score(goal: FinancialGoal) -> float:
        urgency = 1 / max(goal['years'], 0.5)
        return goal['priority'] * urgency

    return sorted(goals, key=score, reverse=True)


def optimize_goal_allocation(
        goals: list[FinancialGoal],
        available_monthly_savings: float) -> dict:
    """
    Optimize which goals to fund with limited budget.
    Uses KNAPSACK DP pattern from Day 28! 🔥

    Args:
        goals: List of financial goals
        available_monthly_savings: Total monthly budget

    Returns:
        Optimal goal allocation plan
    """
    unit = 100
    capacity = int(available_monthly_savings / unit)
    n = len(goals)

    weights = [max(1, int(g['monthly_required'] / unit))
               for g in goals]
    values = [g['priority'] * 10 for g in goals]

    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    keep = [[False] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                with_item = (dp[i - 1][w - weights[i - 1]] +
                             values[i - 1])
                if with_item > dp[i][w]:
                    dp[i][w] = with_item
                    keep[i][w] = True

    funded = []
    deferred = []
    w = capacity

    for i in range(n, 0, -1):
        if keep[i][w]:
            funded.append(goals[i - 1])
            w -= weights[i - 1]
        else:
            deferred.append(goals[i - 1])

    total_allocated = sum(g['monthly_required']
                          for g in funded)

    return {
        "available_savings": available_monthly_savings,
        "total_allocated": total_allocated,
        "remaining": available_monthly_savings - total_allocated,
        "funded_goals": funded,
        "deferred_goals": deferred,
        "max_priority_score": dp[n][capacity]
    }


def merge_goal_timelines(
        goals: list[FinancialGoal]) -> list[dict]:
    """
    Merge overlapping goal timelines.
    Uses MERGE INTERVALS pattern from Day 29!

    Args:
        goals: List of financial goals

    Returns:
        Timeline periods with combined monthly amounts
    """
    if not goals:
        return []

    periods = [
        {
            "start": 0,
            "end": g['years'],
            "amount": g['monthly_required'],
            "goals": [g['name']]
        }
        for g in goals
    ]

    periods.sort(key=lambda x: x['start'])
    merged = [periods[0]]

    for period in periods[1:]:
        last = merged[-1]
        if period['start'] <= last['end']:
            last['end'] = max(last['end'], period['end'])
            last['amount'] += period['amount']
            last['goals'].extend(period['goals'])
        else:
            merged.append(period)

    return merged


def generate_goal_report(
        goals: list[FinancialGoal],
        monthly_savings: float) -> dict:
    """
    Generate complete goal planning report.

    Args:
        goals: List of financial goals
        monthly_savings: Available monthly savings

    Returns:
        Complete analysis report
    """
    prioritized = prioritize_goals(goals)
    optimization = optimize_goal_allocation(
        goals, monthly_savings)
    timeline = merge_goal_timelines(goals)

    funded = len(optimization['funded_goals'])
    deferred = len(optimization['deferred_goals'])

    if deferred == 0:
        recommendation = (
            f"Great news! You can fund all "
            f"{funded} goals with your current savings!")
    else:
        deferred_names = ", ".join(
            g['name'] for g in optimization['deferred_goals'])
        recommendation = (
            f"You can fully fund {funded} priority goals now. "
            f"Consider increasing savings or delaying: "
            f"{deferred_names}")

    return {
        "total_goals": len(goals),
        "prioritized_order": [g['name'] for g in prioritized],
        "optimization": optimization,
        "timeline": timeline,
        "recommendation": recommendation
    }
