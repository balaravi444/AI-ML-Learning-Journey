"""
Day 23 — DSA: Linked Lists
Topic: Middle of Linked List — LeetCode #876
Date: 10 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Fast & Slow Pointer

Genius technique — two pointers, one pass!
Fast moves 2x speed of slow.
When fast reaches end — slow is at middle!

Time Complexity:  O(n)
Space Complexity: O(1)

Real World Connection:
    Fast/slow pointer is used in:
    - Merge sort (find midpoint to split)
    - Cycle detection in computation graphs
    - Finding median in data streams for ML!
"""


class Node:
    def __init__(self, data: int) -> None:
        self.data = data
        self.next = None


def build_list(values: list[int]) -> Node:
    if not values:
        return None
    head = Node(values[0])
    current = head
    for val in values[1:]:
        current.next = Node(val)
        current = current.next
    return head


def find_middle(head: Node) -> Node:
    """
    Find middle node using fast & slow pointer.

    Key insight:
        slow moves 1 step, fast moves 2 steps
        When fast reaches end → slow is at middle!

    Args:
        head: Head of linked list

    Returns:
        Middle node

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next        # 1 step
        fast = fast.next.next   # 2 steps

    return slow  # slow is at middle!


if __name__ == "__main__":
    print("=== Middle of Linked List — LeetCode #876 ===")

    # Odd length
    head = build_list([1, 2, 3, 4, 5])
    middle = find_middle(head)
    print(f"List [1,2,3,4,5] → Middle: {middle.data}")

    # Even length
    head2 = build_list([1, 2, 3, 4, 5, 6])
    middle2 = find_middle(head2)
    print(f"List [1,2,3,4,5,6] → Middle: {middle2.data}")
