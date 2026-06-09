"""
Day 22 — DSA: Stack
Topic: Min Stack — LeetCode #155
Date: 09 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Two Stacks

Design a stack that supports:
push, pop, top, and getMin in O(1)!

Key insight: Keep a second stack tracking minimums!

Time Complexity: O(1) for all operations!
Space Complexity: O(n)
"""


class MinStack:
    """
    Stack that retrieves minimum in O(1).
    Uses two stacks — one for values, one for minimums.
    """

    def __init__(self) -> None:
        """Initialize with two stacks."""
        self._stack: list = []
        self._min_stack: list = []

    def push(self, val: int) -> None:
        """
        Push value to stack.
        Also updates minimum stack.

        Args:
            val: Value to push

        Time Complexity: O(1)
        """
        self._stack.append(val)

        # Update min stack
        if not self._min_stack or val <= self._min_stack[-1]:
            self._min_stack.append(val)

    def pop(self) -> None:
        """
        Remove top element.
        Also updates minimum stack if needed.

        Time Complexity: O(1)
        """
        if self._stack:
            top = self._stack.pop()
            if top == self._min_stack[-1]:
                self._min_stack.pop()

    def top(self) -> int:
        """Return top element. Time: O(1)"""
        return self._stack[-1]

    def get_min(self) -> int:
        """Return minimum element. Time: O(1)"""
        return self._min_stack[-1]


if __name__ == "__main__":
    print("=== Min Stack — LeetCode #155 ===")

    min_stack = MinStack()
    operations = [
        ("push", -2),
        ("push", 0),
        ("push", -3),
        ("getMin", None),
        ("pop", None),
        ("top", None),
        ("getMin", None)
    ]

    for op, val in operations:
        if op == "push":
            min_stack.push(val)
            print(f"push({val})")
        elif op == "pop":
            min_stack.pop()
            print("pop()")
        elif op == "top":
            print(f"top() → {min_stack.top()}")
        elif op == "getMin":
            print(f"getMin() → {min_stack.get_min()}")
