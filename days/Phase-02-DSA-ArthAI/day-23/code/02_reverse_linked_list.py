"""
Day 23 — DSA: Linked Lists
Topic: Reverse Linked List — LeetCode #206
Date: 10 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Iterative pointer manipulation

Most asked linked list interview question!
Asked at Google, Amazon, Microsoft!

Time Complexity:  O(n)
Space Complexity: O(1) iterative, O(n) recursive

Real World Connection:
    Reversing sequences is used in:
    - NLP — processing text backwards
    - Time series — reversing temporal order
    - Backpropagation — gradient flows backward!
"""


class Node:
    def __init__(self, data: int) -> None:
        self.data = data
        self.next = None


def build_list(values: list[int]) -> Node:
    """Build linked list from list of values."""
    if not values:
        return None
    head = Node(values[0])
    current = head
    for val in values[1:]:
        current.next = Node(val)
        current = current.next
    return head


def list_to_array(head: Node) -> list[int]:
    """Convert linked list to array for display."""
    result = []
    current = head
    while current:
        result.append(current.data)
        current = current.next
    return result


def reverse_iterative(head: Node) -> Node:
    """
    Reverse linked list iteratively.

    Key insight:
        Keep track of prev, current, next
        Reverse the pointer direction!

    Args:
        head: Head of linked list

    Returns:
        New head (old tail)

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    prev = None
    current = head

    while current:
        next_node = current.next  # save next
        current.next = prev       # reverse pointer!
        prev = current            # move prev forward
        current = next_node       # move current forward

    return prev  # prev is new head!


def reverse_recursive(head: Node) -> Node:
    """
    Reverse linked list recursively.

    Args:
        head: Head of linked list

    Returns:
        New head

    Time Complexity: O(n)
    Space Complexity: O(n) — call stack
    """
    # Base case
    if not head or not head.next:
        return head

    # Reverse rest of list
    new_head = reverse_recursive(head.next)

    # Reverse current pointer
    head.next.next = head
    head.next = None

    return new_head


if __name__ == "__main__":
    print("=== Reverse Linked List — LeetCode #206 ===")

    # Test iterative
    head = build_list([1, 2, 3, 4, 5])
    print(f"Original:            {list_to_array(head)}")

    reversed_head = reverse_iterative(head)
    print(f"Reversed (iterative): {list_to_array(reversed_head)}")

    # Test recursive
    head2 = build_list([1, 2, 3, 4, 5])
    reversed_head2 = reverse_recursive(head2)
    print(f"Reversed (recursive): {list_to_array(reversed_head2)}")
