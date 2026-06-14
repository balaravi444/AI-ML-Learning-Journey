"""
Day 27 — DSA: Graphs
Topic: Clone Graph — LeetCode #133
Date: 14 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: BFS with hash map

Deep copy a graph!
Each node has value and list of neighbors.

Time Complexity:  O(V + E)
Space Complexity: O(V)

Real World Connection:
    Cloning graphs used in:
    - Copying neural network architectures
    - Creating model copies for ensemble learning
    - Forking computation graphs in PyTorch!
"""
from collections import deque


class Node:
    """Graph node with value and neighbors."""

    def __init__(self, val: int = 0,
                 neighbors: list = None) -> None:
        self.val = val
        self.neighbors = neighbors if neighbors else []


def clone_graph(node: Node) -> Node:
    """
    Deep clone a graph using BFS.

    Key insight:
        Use hash map to track original → clone mapping!
        Process each node once with BFS.

    Args:
        node: Any node in the original graph

    Returns:
        Corresponding node in cloned graph

    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    if not node:
        return None

    # Map original node → cloned node
    cloned = {node: Node(node.val)}
    queue = deque([node])

    while queue:
        current = queue.popleft()

        for neighbor in current.neighbors:
            if neighbor not in cloned:
                # Create clone for new neighbor
                cloned[neighbor] = Node(neighbor.val)
                queue.append(neighbor)

            # Connect cloned nodes
            cloned[current].neighbors.append(
                cloned[neighbor])

    return cloned[node]


if __name__ == "__main__":
    print("=== Clone Graph — LeetCode #133 ===")

    # Build graph: 1--2
    #              |  |
    #              4--3
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n4 = Node(4)

    n1.neighbors = [n2, n4]
    n2.neighbors = [n1, n3]
    n3.neighbors = [n2, n4]
    n4.neighbors = [n1, n3]

    # Clone
    cloned_n1 = clone_graph(n1)

    print(f"Original node 1 val: {n1.val}")
    print(f"Cloned node 1 val: {cloned_n1.val}")
    print(f"Same object? {n1 is cloned_n1}")
    print(f"Neighbors: {[n.val for n in cloned_n1.neighbors]}")
