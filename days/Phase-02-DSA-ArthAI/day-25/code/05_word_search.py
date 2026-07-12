"""
Day 25 — DSA: Recursion & Backtracking
Topic: Word Search — LeetCode #79
Date: 12 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Backtracking + DFS on grid

Find if word exists in 2D grid!
Can move up, down, left, right.

Time Complexity:  O(m * n * 4^L) where L = word length
Space Complexity: O(L) — recursion depth

Real World Connection:
    Grid DFS used in:
    - Image segmentation in Computer Vision ML!
    - Finding connected regions in brain MRI scans
    - Pathfinding in robotics AI!
"""


def word_search(board: list[list[str]],
                word: str) -> bool:
    """
    Search for word in 2D grid.

    Args:
        board: 2D grid of characters
        word: Word to find

    Returns:
        True if word exists, False otherwise

    Time Complexity: O(m * n * 4^L)
    Space Complexity: O(L)
    """
    rows, cols = len(board), len(board[0])

    def dfs(row: int, col: int, index: int) -> bool:
        # Base case — found complete word!
        if index == len(word):
            return True

        # Boundary and character check
        if (row < 0 or row >= rows or
                col < 0 or col >= cols or
                board[row][col] != word[index]):
            return False

        # Mark as visited
        temp = board[row][col]
        board[row][col] = '#'

        # Explore all 4 directions
        found = (dfs(row + 1, col, index + 1) or
                 dfs(row - 1, col, index + 1) or
                 dfs(row, col + 1, index + 1) or
                 dfs(row, col - 1, index + 1))

        # Backtrack — restore cell!
        board[row][col] = temp

        return found

    # Try starting from every cell
    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True

    return False


if __name__ == "__main__":
    print("=== Word Search — LeetCode #79 ===")

    board = [
        ['A', 'B', 'C', 'E'],
        ['S', 'F', 'C', 'S'],
        ['A', 'D', 'E', 'E']
    ]

    words = ["ABCCED", "SEE", "ABCB"]
    for word in words:
        result = word_search([row[:] for row in board], word)
        print(f"'{word}': {result}")
