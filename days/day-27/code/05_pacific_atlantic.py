"""
Day 27 — DSA: Graphs
Topic: Pacific Atlantic Water Flow — LeetCode #417
Date: 14 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Multi-source BFS/DFS

Find cells where water can flow to both oceans!

Key insight:
    Instead of checking each cell (too slow),
    work BACKWARDS from ocean edges!
    BFS from Pacific edges, BFS from Atlantic edges.
    Intersection = cells that reach both!

Time Complexity:  O(m * n)
Space Complexity: O(m * n)

Real World Connection:
    Multi-source BFS used in:
    - Finding regions in satellite imagery for ML
    - Watershed analysis in environmental ML
    - Multi-target pathfinding in robotics AI!
"""
from collections import deque


def pacific_atlantic(heights: list[list[int]]) -> list[list[int]]:
    """
    Find cells that can flow to both oceans.

    Args:
        heights: 2D grid of heights

    Returns:
        List of [row, col] that reach both oceans

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    if not heights:
        return []

    rows, cols = len(heights), len(heights[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def bfs(starts: list) -> set:
        """BFS from multiple starting points."""
        visited = set(starts)
        queue = deque(starts)

        while queue:
            row, col = queue.popleft()

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                if (0 <= new_row < rows and
                        0 <= new_col < cols and
                        (new_row, new_col) not in visited and
                        heights[new_row][new_col] >=
                        heights[row][col]):
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col))

        return visited

    # Pacific touches top and left edges
    pacific_starts = ([(0, c) for c in range(cols)] +
                      [(r, 0) for r in range(rows)])

    # Atlantic touches bottom and right edges
    atlantic_starts = ([(rows - 1, c) for c in range(cols)] +
                       [(r, cols - 1) for r in range(rows)])

    pacific = bfs(pacific_starts)
    atlantic = bfs(atlantic_starts)

    # Intersection = cells reaching both!
    return [[r, c] for r, c in pacific & atlantic]


if __name__ == "__main__":
    print("=== Pacific Atlantic — LeetCode #417 ===")

    heights = [
        [1, 2, 2, 3, 5],
        [3, 2, 3, 4, 4],
        [2, 4, 5, 3, 1],
        [6, 7, 1, 4, 5],
        [5, 1, 1, 2, 4]
    ]

    result = pacific_atlantic(heights)
    print(f"Cells reaching both oceans: {result}")
