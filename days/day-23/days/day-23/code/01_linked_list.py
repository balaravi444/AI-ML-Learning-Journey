"""
Day 23 — DSA: Linked Lists
Topic: Singly Linked List Implementation
Date: 10 June 2026
Author: Bala Ravi

Time Complexity:
    append()   → O(n)
    prepend()  → O(1)
    delete()   → O(n)
    search()   → O(n)
    display()  → O(n)

Space Complexity: O(n)

Real World Connection:
    PyTorch computation graph is essentially a linked list!
    Each layer points to the next — gradient flows backward
    through this chain during backpropagation!
"""


class Node:
    """Single node in a linked list."""

    def __init__(self, data: int) -> None:
        """
        Initialize node with data.

        Args:
            data: Value to store in node
        """
        self.data = data
        self.next = None


class LinkedList:
    """
    Singly Linked List implementation.
    Each node points to the next node.
    """

    def __init__(self) -> None:
        """Initialize empty linked list."""
        self.head = None
        self._size = 0

    def append(self, data: int) -> None:
        """
        Add node at end of list.

        Args:
            data: Value to add

        Time Complexity: O(n) — must traverse to end
        """
        new_node = Node(data)

        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

        self._size += 1

    def prepend(self, data: int) -> None:
        """
        Add node at beginning of list.

        Args:
            data: Value to add

        Time Complexity: O(1) — just update head!
        """
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1

    def delete(self, data: int) -> bool:
        """
        Delete first node with given data.

        Args:
            data: Value to delete

        Returns:
            True if deleted, False if not found

        Time Complexity: O(n)
        """
        if not self.head:
            return False

        # Delete head
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True

        # Find and delete
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next

        return False

    def search(self, data: int) -> int:
        """
        Search for data in list.

        Args:
            data: Value to find

        Returns:
            Index if found, -1 otherwise

        Time Complexity: O(n)
        """
        current = self.head
        index = 0

        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1

        return -1

    def get_length(self) -> int:
        """Return length of linked list."""
        return self._size

    def to_list(self) -> list:
        """Convert linked list to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result

    def __str__(self) -> str:
        if not self.head:
            return "Empty List"
        nodes = self.to_list()
        return " → ".join(map(str, nodes)) + " → None"


if __name__ == "__main__":
    print("=== Singly Linked List ===")
    ll = LinkedList()

    # Append elements
    for val in [1, 2, 3, 4, 5]:
        ll.append(val)
    print(f"After append: {ll}")

    # Prepend
    ll.prepend(0)
    print(f"After prepend 0: {ll}")

    # Search
    idx = ll.search(3)
    print(f"\nSearch for 3: found at index {idx}")

    # Delete
    ll.delete(3)
    print(f"After delete 3: {ll}")

    # Length
    print(f"Length: {ll.get_length()}")
