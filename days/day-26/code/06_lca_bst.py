"""
Day 26 — DSA: Trees & BST
Topic: Lowest Common Ancestor of BST — LeetCode #235
Date: 13 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: BST property traversal

Find lowest common ancestor of two nodes in BST!

Key insight using BST property:
    If both p and q are less than root → LCA is in left
    If both p and q are greater than root → LCA is in right
    Otherwise → root IS the LCA!

Time Complexity:  O(h) — h is tree height
Space Complexity: O(1) iterative

Real World Connection:
    LCA is used in:
    - Finding common ancestor in decision trees
    - Identifying common features in ML hierarchies
    - Knowledge graph reasoning in AI!
"""


class TreeNode:
    def __init__(self, val: int) -> None:
        self.val = val
        self.left = None
        self.right = None


def lca_bst(root: TreeNode,
            p: TreeNode,
            q: TreeNode) -> TreeNode:
    """
    Find lowest common ancestor in BST.

    Args:
        root: Root of BST
        p: First node
        q: Second node

    Returns:
        Lowest common ancestor node

    Time Complexity: O(h)
    Space Complexity: O(1)
    """
    current = root

    while current:
        # Both in left subtree
        if p.val < current.val and q.val < current.val:
            current = current.left

        # Both in right subtree
        elif p.val > current.val and q.val > current.val:
            current = current.right

        # Split point — current is LCA!
        else:
            return current

    return None


def build_bst(values: list[int]) -> TreeNode:
    """Build BST from list of values."""
    root = None

    def insert(node, val):
        if not node:
            return TreeNode(val)
        if val < node.val:
            node.left = insert(node.left, val)
        else:
            node.right = insert(node.right, val)
        return node

    for val in values:
        root = insert(root, val)
    return root


def find_node(root: TreeNode, val: int) -> TreeNode:
    """Find node with given value."""
    if not root:
        return None
    if root.val == val:
        return root
    if val < root.val:
        return find_node(root.left, val)
    return find_node(root.right, val)


if __name__ == "__main__":
    print("=== LCA of BST — LeetCode #235 ===")

    bst = build_bst([6, 2, 8, 0, 4, 7, 9, 3, 5])

    test_cases = [(2, 8), (2, 4), (0, 5)]
    for p_val, q_val in test_cases:
        p = find_node(bst, p_val)
        q = find_node(bst, q_val)
        lca = lca_bst(bst, p, q)
        print(f"LCA({p_val}, {q_val}) = {lca.val}")
