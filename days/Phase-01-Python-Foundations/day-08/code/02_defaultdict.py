"""
Day 08 — Collections Module
Topic: defaultdict for safe dictionary operations
Date: 26 May 2026
Author: Bala Ravi

Real World Connection:
    defaultdict prevents KeyError in data processing!
    Used when building word indexes for NLP and
    grouping data before Pandas is available.
"""
from collections import defaultdict


def group_students_by_grade(students: list[dict]) -> dict:
    """
    Group students by their grade using defaultdict.

    Args:
        students: List of student dictionaries with name and grade

    Returns:
        Dictionary with grades as keys and student lists as values
    """
    groups = defaultdict(list)
    for student in students:
        groups[student['grade']].append(student['name'])
    return dict(groups)


def build_word_index(words: list[str]) -> dict:
    """
    Build an inverted index mapping words to their positions.
    Used in search engines and NLP!

    Args:
        words: List of words to index

    Returns:
        Dictionary mapping each word to list of positions
    """
    index = defaultdict(list)
    for position, word in enumerate(words):
        index[word.lower()].append(position)
    return dict(index)


if __name__ == "__main__":
    # Group students
    students = [
        {"name": "Bala", "grade": "A"},
        {"name": "Ravi", "grade": "B"},
        {"name": "Kumar", "grade": "A"},
        {"name": "Hari", "grade": "B"},
        {"name": "Priya", "grade": "A"}
    ]

    print("=== Students by Grade ===")
    groups = group_students_by_grade(students)
    for grade, names in groups.items():
        print(f"Grade {grade}: {names}")

    print("\n=== Word Index (Search Engine Style) ===")
    sentence = "AI is amazing and AI will change the world".split()
    index = build_word_index(sentence)
    for word, positions in index.items():
        print(f"'{word}' found at positions: {positions}")
