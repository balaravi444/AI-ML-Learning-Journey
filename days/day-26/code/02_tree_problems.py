"""
Day 26 — DSA: Trees & BST
Topic: Common Tree Problems
Date: 13 June 2026
Author: Bala Ravi

Covers:
- Invert Binary Tree LeetCode #226
- Same Tree LeetCode #100
- Maximum Depth LeetCode #104
- Symmetric Tree LeetCode #101
"""
from collections import deque


class TreeNode:
    def __init__(self, val: int) -> None:
        self.val = val
        self.left = None
        self.right = None


def invert_tree(root: TreeNode) -> TreeNode:
    """
    Invert binary tree — LeetCode #226.
    Swap left and right children recursively!

    Args:
        root: Root of binary tree

    Returns:
        Root of inverted tree

    Time Complexity: O(n)
    Space Complexity: O(h)
    """
    if not root:
        return None

    # Swap children
    root.left, root.right = root.right, root.left

    # Recurse
    invert_tree(root.left)
    invert_tree(root.right)

    return root


def is_same_tree(p: TreeNode, q: TreeNode) -> bool:
    """
    Check if two trees are identical — LeetCode #100.

    Args:
        p: Root of first tree
        q: Root of second tree

    Returns:
        True if identical, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(h)
    """
    # Both None — same!
    if not p and not q:
        return True

    # One None, one not — different!
    if not p or not q:
        return False

    # Values must match + children must match
    return (p.val == q.val and
            is_same_tree(p.left, q.left) and
            is_same_tree(p.right, q.right))


def is_symmetric(root: TreeNode) -> bool:
    """
    Check if tree is symmetric — LeetCode #101.

    Args:
        root: Root of binary tree

    Returns:
        True if symmetric (mirror image)

    Time Complexity: O(n)
    """
    def is_mirror(left: TreeNode,
                  right: TreeNode) -> bool:
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))

    return is_mirror(root.left, root.right)


def level_order(root: TreeNode) -> list[list[int]]:
    """Level order traversal helper."""
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


def build_tree(vals: list) -> TreeNode:
    """Build tree from level order list."""
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
    print("=== Invert Binary Tree — LeetCode #226 ===")
    root = build_tree([4, 2, 7, 1, 3, 6, 9])
    print(f"Before: {level_order(root)}")
    invert_tree(root)
    print(f"After:  {level_order(root)}")

    print("\n=== Same Tree — LeetCode #100 ===")
    p = build_tree([1, 2, 3])
    q = build_tree([1, 2, 3])
    r = build_tree([1, 2, 4])
    print(f"[1,2,3] == [1,2,3]: {is_same_tree(p, q)}")
    print(f"[1,2,3] == [1,2,4]: {is_same_tree(p, r)}")

    print("\n=== Symmetric Tree — LeetCode #101 ===")
    sym = build_tree([1, 2, 2, 3, 4, 4, 3])
    not_sym = build_tree([1, 2, 2, None, 3, None, 3])
    print(f"[1,2,2,3,4,4,3] symmetric: {is_symmetric(sym)}")
    print(f"[1,2,2,_,3,_,3] symmetric: {is_symmetric(not_sym)}")
