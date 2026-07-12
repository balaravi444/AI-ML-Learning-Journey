"""
Day 26 — DSA: Trees & BST
Topic: Binary Tree & All Traversals
Date: 13 June 2026
Author: Bala Ravi

Time Complexity: O(n) for all traversals
Space Complexity: O(h) where h = tree height

Real World Connection:
    Decision Trees in Scikit-learn use
    EXACTLY this TreeNode structure internally!
    tree.fit() builds a binary tree like this!
"""
from collections import deque


class TreeNode:
    """Single node in a binary tree."""

    def __init__(self, val: int) -> None:
        """
        Initialize tree node.

        Args:
            val: Value stored in node
        """
        self.val = val
        self.left = None
        self.right = None

    def __str__(self) -> str:
        return str(self.val)


def build_tree(values: list) -> TreeNode:
    """
    Build binary tree from level-order list.
    None represents missing nodes.

    Args:
        values: Level-order list of values

    Returns:
        Root of binary tree
    """
    if not values or values[0] is None:
        return None

    root = TreeNode(values[0])
    queue = deque([root])
    i = 1

    while queue and i < len(values):
        node = queue.popleft()

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1

    return root


def inorder(root: TreeNode) -> list[int]:
    """
    Inorder traversal — Left, Root, Right.
    Gives SORTED order for BST!

    Time Complexity: O(n)
    Space Complexity: O(h)
    """
    if not root:
        return []
    return (inorder(root.left) +
            [root.val] +
            inorder(root.right))


def preorder(root: TreeNode) -> list[int]:
    """
    Preorder traversal — Root, Left, Right.
    Used to copy/serialize a tree!

    Time Complexity: O(n)
    """
    if not root:
        return []
    return ([root.val] +
            preorder(root.left) +
            preorder(root.right))


def postorder(root: TreeNode) -> list[int]:
    """
    Postorder traversal — Left, Right, Root.
    Used to delete a tree!

    Time Complexity: O(n)
    """
    if not root:
        return []
    return (postorder(root.left) +
            postorder(root.right) +
            [root.val])


def level_order(root: TreeNode) -> list[list[int]]:
    """
    Level order traversal using BFS.
    Returns nodes level by level!

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result


def max_depth(root: TreeNode) -> int:
    """
    Find maximum depth of binary tree.
    Same as max_depth parameter in ML trees!

    Time Complexity: O(n)
    Space Complexity: O(h)
    """
    if not root:
        return 0
    return 1 + max(max_depth(root.left),
                   max_depth(root.right))


if __name__ == "__main__":
    # Build tree:
    #        1
    #       / \
    #      2   3
    #     / \ / \
    #    4  5 6  7

    values = [1, 2, 3, 4, 5, 6, 7]
    root = build_tree(values)

    print("=== Binary Tree Traversals ===")
    print(f"Tree values: {values}")
    print(f"Inorder    (L,R,Root): {inorder(root)}")
    print(f"Preorder   (Root,L,R): {preorder(root)}")
    print(f"Postorder  (L,R,Root): {postorder(root)}")
    print(f"Level Order:           {level_order(root)}")
    print(f"Max Depth: {max_depth(root)}")
