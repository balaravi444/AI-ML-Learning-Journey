"""
Day 25 — DSA: Recursion & Backtracking
Topic: Combination Sum — LeetCode #39
Date: 12 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Backtracking with pruning

Find all combinations that sum to target!
Can reuse same element multiple times.

Time Complexity:  O(2ⁿ)
Space Complexity: O(target/min_candidate)

Real World Connection:
    Combination sum pattern used in:
    - Budget allocation in ML experiments
    - Finding optimal hyperparameter combos
    that sum to a constraint!
"""


def combination_sum(candidates: list[int],
                    target: int) -> list[list[int]]:
    """
    Find all combinations summing to target.

    Args:
        candidates: Available numbers
        target: Target sum

    Returns:
        All valid combinations

    Time Complexity: O(2ⁿ)
    Space Complexity: O(target/min_candidate)
    """
    result = []
    candidates.sort()  # sort for pruning!

    def backtrack(start: int,
                  current: list,
                  remaining: int) -> None:
        # Base case — found combination!
        if remaining == 0:
            result.append(current.copy())
            return

        for i in range(start, len(candidates)):
            # Pruning — if too large, stop!
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            # Can reuse same element (i not i+1)
            backtrack(i, current, remaining - candidates[i])
            current.pop()  # backtrack!

    backtrack(0, [], target)
    return result


if __name__ == "__main__":
    print("=== Combination Sum — LeetCode #39 ===")

    candidates = [2, 3, 6, 7]
    target = 7
    result = combination_sum(candidates, target)
    print(f"candidates={candidates}, target={target}")
    print(f"Combinations: {result}")

    candidates2 = [2, 3, 5]
    target2 = 8
    result2 = combination_sum(candidates2, target2)
    print(f"\ncandidates={candidates2}, target={target2}")
    print(f"Combinations: {result2}")
