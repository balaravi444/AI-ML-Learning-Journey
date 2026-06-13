"""
Day 26 — DSA: Trees & BST
Topic: Sorted Array to BST — LeetCode #108
Date: 13 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Recursion — divide and conquer

Convert sorted array to height-balanced BST!

Key insight:
    Middle element becomes root!
    Left half becomes left subtree
    Right half becomes right subtree
    Recursively build each half!

Time Complexity:  O(n)
Space Complexity: O(log n)

Real World Connection:
    Building balanced BST from sorted data
    is used in ML for:
    - Building decision boundaries efficiently
    - Indexing sorted features for fast lookup
    - Creating balanced feature hierarchies!
"""
from collections import deque


class TreeNode:
    def __init__(self, val: int) -> None:
        self.val = val
        self.left = None
        self.right = None


def sorted_array_to_bst(nums: list[int]) -> TreeNode:
    """
    Convert sorted array to height-balanced BST.

    Args:
        nums: Sorted array of integers

    Returns:
        Root of height-balanced BST

    Time Complexity: O(n)
    Space Complexity: O(log n)
    """
    if not nums:
        return None

    # Middle element is root!
    mid = len(nums) // 2
    root = TreeNode(nums[mid])

    # Recursively build left and right
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid + 1:])

    return root


def level_order(root: TreeNode) -> list[list[int]]:
    """Level order traversal."""
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


def inorder(root: TreeNode) -> list[int]:
    """Inorder traversal."""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


if __name__ == "__main__":
    print("=== Sorted Array to BST — LeetCode #108 ===")

    nums = [-10, -3, 0, 5, 9]
    root = sorted_array_to_bst(nums)
    print(f"Input array: {nums}")
    print(f"BST level order: {level_order(root)}")
    print(f"Inorder (should match input): {inorder(root)}")

    nums2 = [1, 2, 3, 4, 5, 6, 7]
    root2 = sorted_array_to_bst(nums2)
    print(f"\nInput: {nums2}")
    print(f"BST level order: {level_order(root2)}")
