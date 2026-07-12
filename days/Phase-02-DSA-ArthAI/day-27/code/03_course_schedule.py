"""
Day 27 — DSA: Graphs
Topic: Course Schedule — LeetCode #207
Date: 14 June 2026
Author: Bala Ravi

Difficulty: Medium
Pattern: Topological Sort + Cycle Detection

Can you finish all courses given prerequisites?
If there's a cycle → impossible!

Time Complexity:  O(V + E)
Space Complexity: O(V + E)

Real World Connection:
    Course scheduling IS ML pipeline scheduling!
    Can you run step B before step A finishes?
    Topological sort answers this for ML systems!

    Used in:
    - ML pipeline dependency resolution
    - Neural network layer ordering
    - Build systems for ML deployment!
"""
from collections import deque


def can_finish(num_courses: int,
               prerequisites: list[list[int]]) -> bool:
    """
    Check if all courses can be finished.
    Uses topological sort (Kahn's algorithm)!

    Key insight:
        Build dependency graph.
        If cycle exists → can't finish all courses!
        Process nodes with no dependencies first.

    Args:
        num_courses: Total number of courses
        prerequisites: [course, prerequisite] pairs

    Returns:
        True if all courses can be finished

    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    """
    # Build graph and in-degree count
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with courses that have no prerequisites
    queue = deque()
    for i in range(num_courses):
        if in_degree[i] == 0:
            queue.append(i)

    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1

        # Reduce in-degree of dependent courses
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return completed == num_courses


def find_order(num_courses: int,
               prerequisites: list[list[int]]) -> list[int]:
    """
    Find order to take all courses — LeetCode #210.

    Args:
        num_courses: Total courses
        prerequisites: Dependencies

    Returns:
        Valid course order, empty if impossible

    Time Complexity: O(V + E)
    """
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue = deque()
    for i in range(num_courses):
        if in_degree[i] == 0:
            queue.append(i)

    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return order if len(order) == num_courses else []


if __name__ == "__main__":
    print("=== Course Schedule — LeetCode #207 ===")

    test_cases = [
        (2, [[1, 0]]),           # True
        (2, [[1, 0], [0, 1]]),   # False — cycle!
        (4, [[1, 0], [2, 1], [3, 2]])  # True
    ]

    for n, prereqs in test_cases:
        result = can_finish(n, prereqs)
        print(f"courses={n}, prereqs={prereqs} → {result}")

    print("\n=== Course Order — LeetCode #210 ===")
    order = find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]])
    print(f"Valid course order: {order}")

    print("\n=== ML Pipeline Scheduling ===")
    steps = {
        "load_data": 0,
        "preprocess": 1,
        "feature_eng": 2,
        "train": 3,
        "evaluate": 4,
        "deploy": 5
    }

    # preprocess needs load_data
    # feature_eng needs preprocess
    # train needs feature_eng
    # evaluate needs train
    # deploy needs evaluate
    ml_deps = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]

    can_run = can_finish(6, ml_deps)
    run_order = find_order(6, ml_deps)
    step_names = list(steps.keys())

    print(f"Can run ML pipeline: {can_run}")
    print(f"Execution order: {[step_names[i] for i in run_order]}")
