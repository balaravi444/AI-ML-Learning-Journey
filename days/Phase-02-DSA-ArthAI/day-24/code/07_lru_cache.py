"""
Day 24 — DSA: Hash Maps & Sets
Topic: LRU Cache — LeetCode #146
Date: 11 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Hash Map + Doubly Linked List

LRU = Least Recently Used Cache
Evicts the least recently used item when full!

Time Complexity: O(1) for get and put!
Space Complexity: O(capacity)

Real World Connection:
    LRU Cache is used EVERYWHERE in ML:
    - Caching model predictions
    - GPU memory management
    - Database query caching
    - Feature store caching in ML pipelines!
"""
from collections import OrderedDict


class LRUCache:
    """
    LRU Cache using OrderedDict.
    OrderedDict maintains insertion order
    and supports move_to_end()!

    Time Complexity: O(1) for all operations
    Space Complexity: O(capacity)
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        """
        Get value by key.
        Moves accessed item to end (most recent).

        Args:
            key: Key to look up

        Returns:
            Value if exists, -1 otherwise

        Time Complexity: O(1)
        """
        if key not in self.cache:
            return -1

        # Move to end — most recently used!
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        """
        Insert or update key-value pair.
        Evicts LRU item if at capacity.

        Args:
            key: Key to insert
            value: Value to store

        Time Complexity: O(1)
        """
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict LRU if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # remove first (LRU)

    def __str__(self) -> str:
        return f"LRU Cache: {dict(self.cache)}"


if __name__ == "__main__":
    print("=== LRU Cache — LeetCode #146 ===")

    cache = LRUCache(3)

    operations = [
        ("put", 1, 1),
        ("put", 2, 2),
        ("put", 3, 3),
        ("get", 1, None),
        ("put", 4, 4),   # evicts key 2 (LRU)!
        ("get", 2, None), # returns -1 (evicted)
        ("get", 3, None),
        ("get", 4, None)
    ]

    for op in operations:
        if op[0] == "put":
            cache.put(op[1], op[2])
            print(f"put({op[1]}, {op[2]}) → {cache}")
        else:
            result = cache.get(op[1])
            print(f"get({op[1]}) → {result}")

    print("\n=== ML Application — Prediction Cache ===")
    prediction_cache = LRUCache(3)

    inputs = ["cat image", "dog image", "bird image",
              "cat image", "fish image", "cat image"]

    for inp in inputs:
        result = prediction_cache.get(hash(inp) % 100)
        if result == -1:
            # Cache miss — compute prediction
            prediction = len(inp) * 0.1  # fake prediction
            prediction_cache.put(hash(inp) % 100, prediction)
            print(f"Cache MISS for '{inp}' → computed {prediction:.1f}")
        else:
            print(f"Cache HIT  for '{inp}' → returned {result:.1f}")
