"""
Day 22 — DSA: Stacks & Queues
Topic: Queue Implementation
Date: 09 June 2026
Author: Bala Ravi

Queue = FIFO (First In First Out)
Like a bank queue — first come first served!

Time Complexity:
    enqueue() → O(1)
    dequeue() → O(1)
    peek()    → O(1)

Real World Connection:
    ML training pipeline uses queue!
    Tasks are queued and processed in order:
    load_data → preprocess → train → evaluate
"""
from collections import deque


class Queue:
    """
    Queue data structure using deque.
    FIFO — First In First Out.
    """

    def __init__(self) -> None:
        """Initialize empty queue."""
        self._items: deque = deque()

    def enqueue(self, item) -> None:
        """
        Add item to back of queue.

        Args:
            item: Item to add

        Time Complexity: O(1)
        """
        self._items.append(item)

    def dequeue(self):
        """
        Remove and return front item.

        Returns:
            Front item

        Raises:
            IndexError: If queue is empty

        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("Dequeue from empty queue!")
        return self._items.popleft()

    def peek(self):
        """Return front item without removing."""
        if self.is_empty():
            raise IndexError("Peek at empty queue!")
        return self._items[0]

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._items) == 0

    def size(self) -> int:
        """Return number of items."""
        return len(self._items)

    def __str__(self) -> str:
        return f"front → {list(self._items)} ← back"


if __name__ == "__main__":
    print("=== Queue Demo ===")
    queue = Queue()

    # Enqueue items
    tasks = ["load_data", "preprocess", "train", "evaluate"]
    for task in tasks:
        queue.enqueue(task)
        print(f"Enqueued '{task}': {queue}")

    print(f"\nPeek: {queue.peek()}")
    print(f"Size: {queue.size()}")

    # Process queue — ML pipeline!
    print("\n=== Running ML Pipeline ===")
    while not queue.is_empty():
        task = queue.dequeue()
        print(f"✅ Running: {task}")
