"""
Day 23 — DSA: Linked Lists
Topic: Linked List Cycle — LeetCode #141
Date: 10 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Floyd's Cycle Detection (Fast & Slow Pointer)

If there's a cycle — fast will eventually
catch up to slow (like a circular race track!)

Time Complexity:  O(n)
Space Complexity: O(1)

Real World Connection:
    Cycle detection is used in:
    - Detecting infinite loops in ML training
    - Detecting circular dependencies in ML pipelines
    - Floyd's algorithm used in optimization theory!
"""


class Node:
    def __init__(self, data: int) -> None:
        self.data = data
        self.next = None


def has_cycle(head: Node) -> bool:
    """
    Detect if linked list has a cycle.
    Uses Floyd's cycle detection algorithm.

    Key insight:
        If cycle exists — fast pointer will
        eventually meet slow pointer!
        Like two runners on a circular track —
        faster one will lap the slower one!

    Args:
        head: Head of linked list

    Returns:
        True if cycle exists, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next        # 1 step
        fast = fast.next.next   # 2 steps

        if slow == fast:        # they met — cycle!
            return True

    return False  # fast reached end — no cycle


if __name__ == "__main__":
    print("=== Linked List Cycle — LeetCode #141 ===")

    # No cycle
    head1 = Node(1)
    head1.next = Node(2)
    head1.next.next = Node(3)
    print(f"No cycle: {has_cycle(head1)}")

    # With cycle
    head2 = Node(1)
    head2.next = Node(2)
    head2.next.next = Node(3)
    head2.next.next.next = head2.next  # cycle back to node 2!
    print(f"Has cycle: {has_cycle(head2)}")
