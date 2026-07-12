"""
Day 29 — Advanced DSA Patterns
Topic: Binary Search on Answer
Date: 16 June 2026
Author: Bala Ravi

Binary search not just on arrays —
but on the ANSWER SPACE!

Real World Connection:
    Finding optimal SIP amount for ArthAI!
    Finding optimal learning rate in ML!
    Finding optimal batch size for GPU!
"""


def koko_eating_bananas(piles: list[int],
                         h: int) -> int:
    """
    Minimum eating speed to finish in h hours.
    LeetCode #875

    Key insight:
        Answer is between 1 and max(piles).
        Binary search on this range!

    Time Complexity: O(n log m)
    Space Complexity: O(1)
    """
    import math

    def can_finish(speed: int) -> bool:
        hours = sum(math.ceil(pile / speed)
                    for pile in piles)
        return hours <= h

    left, right = 1, max(piles)

    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1

    return left


def find_min_sip_amount(target: float,
                         years: int,
                         annual_rate: float) -> int:
    """
    Find minimum monthly SIP to reach target corpus.
    DIRECTLY used in ArthAI retirement planner! 🔥

    Binary search on SIP amount!

    Args:
        target: Target corpus in rupees
        years: Investment duration
        annual_rate: Expected annual return %

    Returns:
        Minimum monthly SIP amount
    """
    monthly_rate = annual_rate / 12 / 100
    months = years * 12

    def calculate_corpus(sip: int) -> float:
        """Future value of SIP."""
        return sip * (((1 + monthly_rate) ** months - 1) /
                      monthly_rate) * (1 + monthly_rate)

    left, right = 100, 1000000

    while left < right:
        mid = (left + right) // 2
        if calculate_corpus(mid) >= target:
            right = mid
        else:
            left = mid + 1

    return left


def search_rotated_array(nums: list[int],
                          target: int) -> int:
    """
    Search in rotated sorted array.
    LeetCode #33

    Time Complexity: O(log n)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1


if __name__ == "__main__":
    print("=== Binary Search on Answer ===")

    piles = [3, 6, 7, 11]
    h = 8
    print(f"Koko bananas {piles}, h={h}: {koko_eating_bananas(piles, h)}")

    print("\n=== ArthAI — Minimum SIP Calculator ===")
    scenarios = [
        (5000000, 20, 12),   # ₹50L in 20 years at 12%
        (10000000, 25, 12),  # ₹1Cr in 25 years at 12%
        (2500000, 10, 10),   # ₹25L in 10 years at 10%
    ]

    for target, years, rate in scenarios:
        sip = find_min_sip_amount(target, years, rate)
        print(f"Target: ₹{target:,} in {years}yrs at {rate}%")
        print(f"Min SIP: ₹{sip:,}/month\n")

    print("=== Search Rotated Array ===")
    nums = [4, 5, 6, 7, 0, 1, 2]
    print(f"Search 0 in {nums}: index {search_rotated_array(nums, 0)}")
