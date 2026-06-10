"""
Day 23 — DSA: Linked Lists
Topic: Remove Nth Node From End — LeetCode #19
Date: 10 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Fast & Slow Pointer + Dummy Node

One pass solution using two pointers!
Fast pointer goes n steps ahead first,
then both move together until fast reaches end.
Slow is now at the node before the one to delete!

Time Complexity:  O(n)
Space Complexity: O(1)
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


def remove_nth_from_end(head: Node, n: int) -> Node:
    """
    Remove nth node from end of list in one pass.

    Key insight:
        Move fast pointer n steps ahead.
        Then move both until fast reaches end.
        Slow is now at node BEFORE the one to delete!

    Args:
        head: Head of linked list
        n: Position from end to remove

    Returns:
        Head of modified list

    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    dummy = Node(0)
    dummy.next = head

    fast = dummy
    slow = dummy

    # Move fast n+1 steps ahead
    for _ in range(n + 1):
        fast = fast.next

    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next

    # Remove nth node from end
    slow.next = slow.next.next

    return dummy.next


if __name__ == "__main__":
    print("=== Remove Nth Node From End — LeetCode #19 ===")

    head = build_list([1, 2, 3, 4, 5])
    print(f"Original: {list_to_array(head)}")

    result = remove_nth_from_end(head, 2)
    print(f"Remove 2nd from end: {list_to_array(result)}")

    head2 = build_list([1, 2, 3, 4, 5])
    result2 = remove_nth_from_end(head2, 1)
    print(f"Remove 1st from end: {list_to_array(result2)}")
