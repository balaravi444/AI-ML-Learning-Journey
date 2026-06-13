"""
Day 26 — DSA: Trees & BST
Topic: Binary Search Tree Implementation
Date: 13 June 2026
Author: Bala Ravi

BST Property: Left < Root < Right

Time Complexity:
    Search: O(log n) average, O(n) worst
    Insert: O(log n) average, O(n) worst
    Delete: O(log n) average, O(n) worst

Real World Connection:
    Database indexes use BST-like structures!
    When ML training data is indexed for
    fast lookup — BST variants are used!
"""


class TreeNode:
    def __init__(self, val: int) -> None:
        self.val = val
        self.left = None
        self.right = None


class BST:
    """
    Binary Search Tree implementation.
    Left subtree < root < right subtree.
    """

    def __init__(self) -> None:
        """Initialize empty BST."""
        self.root = None

    def insert(self, val: int) -> None:
        """
        Insert value into BST.

        Args:
            val: Value to insert

        Time Complexity: O(log n) average
        """
        self.root = self._insert(self.root, val)

    def _insert(self,
                node: TreeNode,
                val: int) -> TreeNode:
        """Recursive insert helper."""
        if not node:
            return TreeNode(val)

        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)

        return node

    def search(self, val: int) -> bool:
        """
        Search for value in BST.

        Args:
            val: Value to find

        Returns:
            True if found, False otherwise

        Time Complexity: O(log n) average
        """
        return self._search(self.root, val)

    def _search(self,
                node: TreeNode,
                val: int) -> bool:
        """Recursive search helper."""
        if not node:
            return False

        if val == node.val:
            return True
        elif val < node.val:
            return self._search(node.left, val)
        else:
            return self._search(node.right, val)

    def inorder(self) -> list[int]:
        """
        Inorder traversal — returns sorted values!

        Returns:
            Sorted list of all values
        """
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self,
                 node: TreeNode,
                 result: list) -> None:
        """Recursive inorder helper."""
        if not node:
            return
        self._inorder(node.left, result)
        result.append(node.val)
        self._inorder(node.right, result)


if __name__ == "__main__":
    print("=== Binary Search Tree ===")
    bst = BST()

    values = [5, 3, 7, 1, 4, 6, 8, 2]
    for val in values:
        bst.insert(val)

    print(f"Inserted: {values}")
    print(f"Inorder (sorted): {bst.inorder()}")

    print("\n=== Search ===")
    for val in [4, 9, 1, 10]:
        found = bst.search(val)
        print(f"Search {val}: {found}")
