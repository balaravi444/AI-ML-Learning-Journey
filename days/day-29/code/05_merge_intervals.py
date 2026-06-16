"""
Day 29 — Advanced DSA Patterns
Topic: Merge Intervals
Date: 16 June 2026
Author: Bala Ravi

Merge overlapping intervals!

Time Complexity:  O(n log n)
Space Complexity: O(n)

Real World Connection:
    Merging time periods in financial planning!
    Calendar scheduling in ML training jobs!
    ArthAI — merging investment periods! 🔥
"""


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Merge all overlapping intervals.
    LeetCode #56

    Args:
        intervals: List of [start, end] intervals

    Returns:
        Merged non-overlapping intervals

    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def insert_interval(intervals: list[list[int]],
                    new_interval: list[int]) -> list[list[int]]:
    """
    Insert new interval and merge.
    LeetCode #57

    Time Complexity: O(n)
    """
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals before new_interval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0],
                              intervals[i][0])
        new_interval[1] = max(new_interval[1],
                              intervals[i][1])
        i += 1

    result.append(new_interval)

    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1

    return result


def merge_financial_periods(
        goals: list[dict]) -> list[dict]:
    """
    Merge overlapping financial goal periods.
    Used in ArthAI goal planner! 🔥

    Args:
        goals: List of financial goals with periods

    Returns:
        Merged goal periods
    """
    if not goals:
        return []

    # Sort by start year
    goals.sort(key=lambda x: x['start_year'])
    merged = [goals[0].copy()]

    for goal in goals[1:]:
        last = merged[-1]
        if goal['start_year'] <= last['end_year']:
            last['end_year'] = max(last['end_year'],
                                   goal['end_year'])
            last['monthly_amount'] += goal['monthly_amount']
            last['goals'].extend(goal['goals'])
        else:
            merged.append(goal.copy())

    return merged


if __name__ == "__main__":
    print("=== Merge Intervals — LeetCode #56 ===")

    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    print(f"Input:  {intervals}")
    print(f"Merged: {merge_intervals(intervals)}")

    print("\n=== ArthAI — Financial Goal Merging ===")
    goals = [
        {
            "goals": ["Emergency Fund"],
            "start_year": 2026,
            "end_year": 2027,
            "monthly_amount": 5000
        },
        {
            "goals": ["Vacation"],
            "start_year": 2026,
            "end_year": 2028,
            "monthly_amount": 3000
        },
        {
            "goals": ["Car Purchase"],
            "start_year": 2029,
            "end_year": 2031,
            "monthly_amount": 10000
        }
    ]

    merged = merge_financial_periods(goals)
    print("Merged Financial Plan:")
    for period in merged:
        print(f"  {period['start_year']}-{period['end_year']}: "
              f"₹{period['monthly_amount']:,}/month "
              f"for {period['goals']}")
