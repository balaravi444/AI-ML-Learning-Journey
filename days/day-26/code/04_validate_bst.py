"""
Day 26 — DSA: Trees & BST
Topic: Validate Binary Search Tree — LeetCode #98
Date: 13 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: DFS with min/max bounds

Check if a binary tree is a valid BST!

Key insight:
    Each node must be within a valid range!
    Left subtree: all values < current node
    Right subtree: all values > current node
    Track min and max bounds while traversing!

Time Complexity:  O(n)
Space Complexity: O(h)

Real World Connection:
    Validating tree structure is used in:
    - Validating Decision Tree structure in ML
    - Checking neural network architecture validity
    - Validating hierarchical data in ML pipelines!
"""


class TreeNode:
    def __init__(self, val: int) -> None:
        self.val = val
        self.left = None
        self.right = None


def is_valid_bst(root: TreeNode) -> bool:
    """
    Validate if binary tree is a valid BST.

    Args:
        root: Root of binary tree

    Returns:
        True if valid BST, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(h)
    """
    def validate(node: TreeNode,
                 min_val: float,
                 max_val: float) -> bool:
        # Empty tree is valid BST
        if not node:
            return True

        # Current node must be within bounds
        if node.val <= min_val or node.val >= max_val:
            return False

        # Left must be less than current
        # Right must be greater than current
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))


def build_tree(vals: list) -> TreeNode:
    """Build tree from level order list."""
    from collections import deque
    if not vals:
        return None
    root = TreeNode(vals[0])
    queue = deque([root])
    i = 1
    while queue and i < len(vals):
        node = queue.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            queue.append(node.right)
        i += 1
    return root


if __name__ == "__main__":
    print("=== Validate BST — LeetCode #98 ===")

    # Valid BST
    valid = build_tree([5, 3, 7, 1, 4, 6, 8])
    print(f"[5,3,7,1,4,6,8] valid BST: {is_valid_bst(valid)}")

    # Invalid BST
    invalid = build_tree([5, 1, 4, None, None, 3, 6])
    print(f"[5,1,4,_,_,3,6] valid BST: {is_valid_bst(invalid)}")
