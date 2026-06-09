"""
Day 22 — DSA: Stacks & Queues
Topic: Stack Implementation
Date: 09 June 2026
Author: Bala Ravi

Stack = LIFO (Last In First Out)
Like a stack of plates — add/remove from top!

Time Complexity:
    push()  → O(1)
    pop()   → O(1)
    peek()  → O(1)
    is_empty() → O(1)

Real World Connection:
    Python's function call stack uses this!
    Every function call is pushed → when done popped.
    RecursionError = stack overflow!
"""


class Stack:
    """
    Stack data structure implementation using Python list.
    LIFO — Last In First Out.
    """

    def __init__(self) -> None:
        """Initialize empty stack."""
        self._items: list = []

    def push(self, item) -> None:
        """
        Add item to top of stack.

        Args:
            item: Item to add

        Time Complexity: O(1)
        """
        self._items.append(item)

    def pop(self):
        """
        Remove and return top item.

        Returns:
            Top item

        Raises:
            IndexError: If stack is empty

        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("Pop from empty stack!")
        return self._items.pop()

    def peek(self):
        """
        Return top item without removing.

        Returns:
            Top item

        Raises:
            IndexError: If stack is empty

        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("Peek at empty stack!")
        return self._items[-1]

    def is_empty(self) -> bool:
        """Check if stack is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Return number of items in stack."""
        return len(self._items)

    def __str__(self) -> str:
        return f"Stack{self._items} ← top"


if __name__ == "__main__":
    print("=== Stack Demo ===")
    stack = Stack()

    # Push items
    for item in [1, 2, 3, 4, 5]:
        stack.push(item)
        print(f"Pushed {item}: {stack}")

    print(f"\nPeek: {stack.peek()}")
    print(f"Size: {stack.size()}")

    # Pop items
    print("\nPopping:")
    while not stack.is_empty():
        print(f"Popped: {stack.pop()} | {stack}")
