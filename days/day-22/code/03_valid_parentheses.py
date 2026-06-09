"""
Day 22 — DSA: Stack
Topic: Valid Parentheses — LeetCode #20
Date: 09 June 2026
Author: Bala Ravi

Difficulty: Easy
Pattern: Stack

Classic stack problem!
Asked in every company interview!

Time Complexity:  O(n)
Space Complexity: O(n)
"""


def is_valid(s: str) -> bool:
    """
    Check if string has valid parentheses.

    Key insight:
        Opening bracket → push to stack
        Closing bracket → must match top of stack!

    Args:
        s: String containing brackets

    Returns:
        True if valid, False otherwise

    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # Closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # Opening bracket
            stack.append(char)

    return len(stack) == 0


if __name__ == "__main__":
    print("=== Valid Parentheses — LeetCode #20 ===")

    test_cases = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("{[]}", True),
        ("", True)
    ]

    for s, expected in test_cases:
        result = is_valid(s)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{s}' → {result}")
