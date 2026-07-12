"""
Day 25 — DSA: Recursion & Backtracking
Topic: N-Queens — LeetCode #51
Date: 12 June 2026
Author: Bala Ravi

Difficulty: Hard
Pattern: Backtracking with constraint checking

Place N queens on NxN board so no two attack each other!

Time Complexity:  O(n!)
Space Complexity: O(n)

Real World Connection:
    N-Queens type constraint satisfaction used in:
    - Scheduling ML training jobs on GPUs
    - Resource allocation in ML pipelines
    - Conflict resolution in AI planning systems!
"""


def solve_n_queens(n: int) -> list[list[str]]:
    """
    Solve N-Queens problem.

    Key insight:
        Place one queen per row.
        Track columns, diagonals under attack.
        Backtrack when constraint violated!

    Args:
        n: Board size and number of queens

    Returns:
        All valid board configurations

    Time Complexity: O(n!)
    Space Complexity: O(n)
    """
    result = []
    board = [['.'] * n for _ in range(n)]

    # Track attacked positions
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row: int) -> None:
        # Base case — placed all queens!
        if row == n:
            result.append([''.join(r) for r in board])
            return

        for col in range(n):
            # Skip if under attack
            if (col in cols or
                    row - col in diag1 or
                    row + col in diag2):
                continue

            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            # Recurse to next row
            backtrack(row + 1)

            # Backtrack — remove queen
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result


if __name__ == "__main__":
    print("=== N-Queens — LeetCode #51 ===")

    for n in [4, 5]:
        solutions = solve_n_queens(n)
        print(f"\n{n}-Queens: {len(solutions)} solutions")
        if solutions:
            print("First solution:")
            for row in solutions[0]:
                print(f"  {row}")
