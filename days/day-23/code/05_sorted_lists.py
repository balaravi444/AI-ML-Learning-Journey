"""
Day 23 — DSA: Linked Lists
Topic: Merge Two Sorted Lists — LeetCode #21
Date: 10 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Two Pointers + Dummy Node

Classic merge operation — foundation of Merge Sort!
Python's built-in sort uses this internally!

Time Complexity:  O(n + m)
Space Complexity: O(1)

Real World Connection:
    Merging sorted lists is used in:
    - Merge sort (Python's Timsort uses this!)
    - Merging sorted datasets in ML pipelines
    - Combining sorted predictions from ensemble models!
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


def list_to_array(head: Node) -> list[int]:
    result = []
    current = head
    while current:
        result.append(current.data)
        current = current.next
    return result


def merge_sorted_lists(l1: Node, l2: Node) -> Node:
    """
    Merge two sorted linked lists.

    Key insight:
        Use dummy node to simplify edge cases!
        Compare heads of both lists → take smaller one.

    Args:
        l1: Head of first sorted list
        l2: Head of second sorted list

    Returns:
        Head of merged sorted list

    Time Complexity: O(n + m)
    Space Complexity: O(1)
    """
    # Dummy node simplifies edge cases!
    dummy = Node(0)
    current = dummy

    while l1 and l2:
        if l1.data <= l2.data:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    # Attach remaining nodes
    current.next = l1 if l1 else l2

    return dummy.next  # skip dummy!


if __name__ == "__main__":
    print("=== Merge Two Sorted Lists — LeetCode #21 ===")

    l1 = build_list([1, 2, 4])
    l2 = build_list([1, 3, 4])
    merged = merge_sorted_lists(l1, l2)
    print(f"List 1: [1, 2, 4]")
    print(f"List 2: [1, 3, 4]")
    print(f"Merged: {list_to_array(merged)}")

    l3 = build_list([1, 3, 5, 7])
    l4 = build_list([2, 4, 6, 8])
    merged2 = merge_sorted_lists(l3, l4)
    print(f"\nList 1: [1, 3, 5, 7]")
    print(f"List 2: [2, 4, 6, 8]")
    print(f"Merged: {list_to_array(merged2)}")
