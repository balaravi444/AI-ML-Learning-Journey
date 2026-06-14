"""
Day 27 — DSA: Graphs
Topic: Number of Islands — LeetCode #200
Date: 14 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: DFS on 2D grid

Count number of islands in a 2D grid!
'1' = land, '0' = water

Time Complexity:  O(m * n)
Space Complexity: O(m * n) — recursion stack

Real World Connection:
    Grid DFS is used in:
    - Image segmentation in Computer Vision ML!
    - Finding connected regions in satellite images
    - Semantic segmentation for self-driving cars!
    - Medical image analysis (finding tumors)!
"""


def num_islands(grid: list[list[str]]) -> int:
    """
    Count number of islands using DFS.

    Key insight:
        For each unvisited land cell → start DFS
        Mark all connected land as visited
        Each DFS call = one complete island!

    Args:
        grid: 2D grid of '1' (land) and '0' (water)

    Returns:
        Number of islands

    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    count = 0

    def dfs(row: int, col: int) -> None:
        """Mark all connected land as visited."""
        # Boundary check
        if (row < 0 or row >= rows or
                col < 0 or col >= cols or
                grid[row][col] != '1'):
            return

        # Mark as visited
        grid[row][col] = '#'

        # Explore all 4 directions
        dfs(row + 1, col)
        dfs(row - 1, col)
        dfs(row, col + 1)
        dfs(row, col - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count


def max_area_island(grid: list[list[int]]) -> int:
    """
    Find maximum area island — LeetCode #695.

    Args:
        grid: 2D grid of 1 (land) and 0 (water)

    Returns:
        Maximum island area

    Time Complexity: O(m * n)
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    max_area = 0

    def dfs(row: int, col: int) -> int:
        if (row < 0 or row >= rows or
                col < 0 or col >= cols or
                grid[row][col] != 1):
            return 0

        grid[row][col] = 0  # mark visited

        return (1 +
                dfs(row + 1, col) +
                dfs(row - 1, col) +
                dfs(row, col + 1) +
                dfs(row, col - 1))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))

    return max_area


if __name__ == "__main__":
    print("=== Number of Islands — LeetCode #200 ===")

    grids = [
        [["1", "1", "1", "1", "0"],
         ["1", "1", "0", "1", "0"],
         ["1", "1", "0", "0", "0"],
         ["0", "0", "0", "0", "0"]],

        [["1", "1", "0", "0", "0"],
         ["1", "1", "0", "0", "0"],
         ["0", "0", "1", "0", "0"],
         ["0", "0", "0", "1", "1"]]
    ]

    for grid in grids:
        result = num_islands([row[:] for row in grid])
        print(f"Islands: {result}")

    print("\n=== Max Area Island — LeetCode #695 ===")
    grid = [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]]
    print(f"Max island area: {max_area_island(grid)}")
