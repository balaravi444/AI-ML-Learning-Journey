"""
ArthAI — Goal Planner Module
Multi-goal financial optimization using Knapsack DP!

Author: Bala Ravi
Date: 18 June 2026
"""
from typing import TypedDict


class FinancialGoal(TypedDict):
    """Type definition for a financial goal."""
    name: str
    target_amount: float
    years: int
    priority: int  # 1-10, 10 = highest
    monthly_required: float


def calculate_monthly_required(
        target: float,
        years: int,
        rate: float = 10.0) -> float:
    """
    Calculate monthly savings needed for a goal.

    Args:
        target: Target amount
        years: Time horizon
        rate: Expected annual return %

    Returns:
        Monthly amount needed
    """
    from financial_utils import find_minimum_sip
    return find_minimum_sip(target, rate, years)


def prioritize_goals(
        goals: list[FinancialGoal]) -> list[FinancialGoal]:
    """
    Sort goals by priority and urgency.

    Key insight:
        Priority score = priority_weight / years_remaining
        Urgent + Important goals score highest!

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

    Key insight:
        goals = items
        priority = value
        monthly_required = weight (in units of 100)
        available_savings = capacity

    Args:
        goals: List of financial goals
        available_monthly_savings: Total monthly budget

    Returns:
        Optimal goal allocation plan

    Time Complexity: O(n * capacity)
    """
    # Convert to integer units (per ₹100) for DP
    unit = 100
    capacity = int(available_monthly_savings / unit)
    n = len(goals)

    weights = [int(g['monthly_required'] / unit)
               for g in goals]
    values = [g['priority'] * 10 for g in goals]

    # Standard 0/1 Knapsack DP
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

    # Backtrack to find which goals are funded
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
    Merge overlapping goal timelines for visualization.
    Uses MERGE INTERVALS pattern from Day 29!

    Args:
        goals: List of financial goals

    Returns:
        Timeline periods with combined monthly amounts
    """
    if not goals:
        return []

    # Convert to [start_year, end_year, amount, names]
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

    return {
        "total_goals": len(goals),
        "prioritized_order": [g['name'] for g in prioritized],
        "optimization": optimization,
        "timeline": timeline,
        "recommendation": _generate_recommendation(
            optimization)
    }


def _generate_recommendation(optimization: dict) -> str:
    """Generate human-readable recommendation."""
    funded = len(optimization['funded_goals'])
    deferred = len(optimization['deferred_goals'])

    if deferred == 0:
        return (f"🎉 Great news! You can fund all "
                f"{funded} goals with your current savings!")

    deferred_names = ", ".join(
        g['name'] for g in optimization['deferred_goals'])

    return (f"You can fully fund {funded} priority goals now. "
            f"Consider increasing savings or delaying: "
            f"{deferred_names}")


if __name__ == "__main__":
    print("=== ArthAI Goal Planner Demo ===\n")

    my_goals: list[FinancialGoal] = [
        {
            "name": "Emergency Fund",
            "target_amount": 200000,
            "years": 1,
            "priority": 10,
            "monthly_required": calculate_monthly_required(
                200000, 1)
        },
        {
            "name": "House Down Payment",
            "target_amount": 1500000,
            "years": 5,
            "priority": 8,
            "monthly_required": calculate_monthly_required(
                1500000, 5)
        },
        {
            "name": "Child Education",
            "target_amount": 2500000,
            "years": 15,
            "priority": 9,
            "monthly_required": calculate_monthly_required(
                2500000, 15)
        },
        {
            "name": "Dream Vacation",
            "target_amount": 300000,
            "years": 2,
            "priority": 4,
            "monthly_required": calculate_monthly_required(
                300000, 2)
        },
        {
            "name": "New Car",
            "target_amount": 800000,
            "years": 3,
            "priority": 6,
            "monthly_required": calculate_monthly_required(
                800000, 3)
        }
    ]

    print("📋 Your Financial Goals:")
    for g in my_goals:
        print(f"  • {g['name']}: ₹{g['monthly_required']:,}/month "
              f"for {g['years']} years (priority: {g['priority']})")

    available = 15000
    print(f"\n💰 Available Monthly Savings: ₹{available:,}")

    report = generate_goal_report(my_goals, available)

    print(f"\n🎯 Optimized Goal Plan:")
    print(f"  Total Allocated: ₹{report['optimization']['total_allocated']:,.0f}")
    print(f"  Remaining: ₹{report['optimization']['remaining']:,.0f}")

    print(f"\n✅ Funded Goals:")
    for g in report['optimization']['funded_goals']:
        print(f"  • {g['name']}")

    if report['optimization']['deferred_goals']:
        print(f"\n⏳ Deferred Goals:")
        for g in report['optimization']['deferred_goals']:
            print(f"  • {g['name']}")

    print(f"\n💡 {report['recommendation']}")
