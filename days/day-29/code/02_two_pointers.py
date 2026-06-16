"""
Day 29 — Advanced DSA Patterns
Topic: Two Pointers Pattern
Date: 16 June 2026
Author: Bala Ravi

Two pointers = two indices moving toward each other
or in same direction!

Real World Connection:
    Used in data preprocessing for ML!
    Merging sorted datasets in ML pipelines!
"""


def three_sum(nums: list[int]) -> list[list[int]]:
    """
    Find all triplets summing to 0.
    LeetCode #15

    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # skip duplicates

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i],
                                nums[left],
                                nums[right]])
                while left < right and \
                        nums[left] == nums[left + 1]:
                    left += 1
                while left < right and \
                        nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result


def container_with_most_water(height: list[int]) -> int:
    """
    Container with most water.
    LeetCode #11

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        water = min(height[left],
                    height[right]) * (right - left)
        max_water = max(max_water, water)

        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water


def remove_duplicates(nums: list[int]) -> int:
    """
    Remove duplicates from sorted array in-place.
    LeetCode #26

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return 0

    slow = 0

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1


if __name__ == "__main__":
    print("=== Two Pointers Patterns ===")

    nums = [-1, 0, 1, 2, -1, -4]
    print(f"3Sum {nums}: {three_sum(nums)}")

    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    print(f"Max water: {container_with_most_water(heights)}")

    arr = [1, 1, 2, 3, 3, 4]
    k = remove_duplicates(arr)
    print(f"After dedup: {arr[:k]}")
