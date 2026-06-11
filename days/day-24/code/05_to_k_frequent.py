"""
Day 24 — DSA: Hash Maps & Sets
Topic: Top K Frequent Elements — LeetCode #347
Date: 11 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Hash Map + Bucket Sort

Find k most frequent elements!

Optimal solution uses bucket sort — O(n)!
Bucket sort: index = frequency!

Time Complexity:  O(n) bucket sort solution
Space Complexity: O(n)

Real World Connection:
    Top K frequent is used in:
    - Finding top keywords in NLP
    - Recommendation systems (top K items)
    - Log analysis in ML systems!
"""
from collections import Counter
import heapq


def top_k_frequent_sort(nums: list[int], k: int) -> list[int]:
    """
    Find top k frequent elements using sort.

    Time Complexity: O(n log n)
    """
    freq = Counter(nums)
    return sorted(freq.keys(),
                  key=lambda x: freq[x],
                  reverse=True)[:k]


def top_k_frequent_heap(nums: list[int], k: int) -> list[int]:
    """
    Find top k frequent elements using min heap.

    Time Complexity: O(n log k)
    Space Complexity: O(n)
    """
    freq = Counter(nums)
    return heapq.nlargest(k, freq.keys(), key=freq.get)


def top_k_frequent_bucket(nums: list[int], k: int) -> list[int]:
    """
    Find top k frequent elements using bucket sort.
    Most optimal solution — O(n)!

    Key insight:
        Maximum frequency is n (all same elements).
        Create n+1 buckets where index = frequency!
        Elements with frequency f go in bucket[f].

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    freq = Counter(nums)

    # Buckets where index = frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    # Collect top k from highest frequency buckets
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result

    return result


if __name__ == "__main__":
    print("=== Top K Frequent — LeetCode #347 ===")

    test_cases = [
        ([1, 1, 1, 2, 2, 3], 2),
        ([1], 1),
        ([1, 2, 2, 3, 3, 3], 2)
    ]

    for nums, k in test_cases:
        result = top_k_frequent_bucket(nums, k)
        print(f"nums={nums}, k={k} → {result}")

    print("\n=== NLP Application — Top Keywords ===")
    text = "AI ML AI Python AI ML Python Data Science AI"
    words = text.split()
    top_3 = top_k_frequent_bucket(words, 3)
    print(f"Text: '{text}'")
    print(f"Top 3 keywords: {top_3}")
