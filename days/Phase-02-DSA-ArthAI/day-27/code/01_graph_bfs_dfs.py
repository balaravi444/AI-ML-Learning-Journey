"""
Day 27 — DSA: Graphs — BFS & DFS
Topic: Graph Implementation + BFS + DFS
Date: 14 June 2026
Author: Bala Ravi

Time Complexity: O(V + E) for both BFS and DFS
Space Complexity: O(V)

Real World Connection:
    BFS finds shortest path — used in GPS navigation!
    DFS finds all paths — used in maze solving!
    Both used in knowledge graph reasoning in LLMs!
"""
from collections import deque


class Graph:
    """
    Undirected graph using adjacency list.
    Most efficient for sparse graphs (real world!)
    """

    def __init__(self) -> None:
        """Initialize empty graph."""
        self.adjacency_list: dict[int, list[int]] = {}

    def add_vertex(self, vertex: int) -> None:
        """
        Add vertex to graph.

        Args:
            vertex: Vertex to add

        Time Complexity: O(1)
        """
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self,
                 vertex1: int,
                 vertex2: int) -> None:
        """
        Add undirected edge between vertices.

        Args:
            vertex1: First vertex
            vertex2: Second vertex

        Time Complexity: O(1)
        """
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        self.adjacency_list[vertex1].append(vertex2)
        self.adjacency_list[vertex2].append(vertex1)

    def bfs(self, start: int) -> list[int]:
        """
        Breadth First Search — explores level by level.
        Uses QUEUE — FIFO!
        Finds SHORTEST path in unweighted graph!

        Args:
            start: Starting vertex

        Returns:
            Vertices in BFS order

        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = set([start])
        queue = deque([start])
        order = []

        while queue:
            vertex = queue.popleft()
            order.append(vertex)

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return order

    def dfs(self, start: int) -> list[int]:
        """
        Depth First Search — explores deep first.
        Uses STACK (via recursion)!
        Finds all paths and detects cycles!

        Args:
            start: Starting vertex

        Returns:
            Vertices in DFS order

        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = set()
        order = []

        def explore(vertex: int) -> None:
            visited.add(vertex)
            order.append(vertex)

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    explore(neighbor)

        explore(start)
        return order

    def dfs_iterative(self, start: int) -> list[int]:
        """
        Iterative DFS using explicit stack.
        Avoids recursion limit for large graphs!

        Args:
            start: Starting vertex

        Returns:
            Vertices in DFS order

        Time Complexity: O(V + E)
        Space Complexity: O(V)
        """
        visited = set()
        stack = [start]
        order = []

        while stack:
            vertex = stack.pop()

            if vertex not in visited:
                visited.add(vertex)
                order.append(vertex)

                for neighbor in self.adjacency_list[vertex]:
                    if neighbor not in visited:
                        stack.append(neighbor)

        return order

    def has_path(self,
                 start: int,
                 end: int) -> bool:
        """
        Check if path exists between two vertices.

        Args:
            start: Source vertex
            end: Destination vertex

        Returns:
            True if path exists

        Time Complexity: O(V + E)
        """
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            vertex = queue.popleft()

            if vertex == end:
                return True

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def shortest_path(self,
                      start: int,
                      end: int) -> list[int]:
        """
        Find shortest path using BFS.

        Args:
            start: Source vertex
            end: Destination vertex

        Returns:
            Shortest path as list, empty if no path

        Time Complexity: O(V + E)
        """
        visited = set([start])
        queue = deque([(start, [start])])

        while queue:
            vertex, path = queue.popleft()

            if vertex == end:
                return path

            for neighbor in self.adjacency_list[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor,
                                  path + [neighbor]))

        return []

    def count_components(self) -> int:
        """
        Count connected components in graph.

        Returns:
            Number of connected components

        Time Complexity: O(V + E)
        """
        visited = set()
        components = 0

        for vertex in self.adjacency_list:
            if vertex not in visited:
                # DFS from unvisited vertex
                stack = [vertex]
                while stack:
                    v = stack.pop()
                    if v not in visited:
                        visited.add(v)
                        stack.extend(
                            self.adjacency_list[v])
                components += 1

        return components

    def __str__(self) -> str:
        result = "Graph:\n"
        for vertex, neighbors in self.adjacency_list.items():
            result += f"  {vertex} → {neighbors}\n"
        return result


if __name__ == "__main__":
    print("=== Graph BFS & DFS ===")

    g = Graph()
    edges = [(1, 2), (1, 3), (2, 4),
             (3, 4), (4, 5), (5, 6)]

    for v1, v2 in edges:
        g.add_edge(v1, v2)

    print(g)

    print(f"BFS from 1: {g.bfs(1)}")
    print(f"DFS from 1: {g.dfs(1)}")
    print(f"DFS iterative: {g.dfs_iterative(1)}")

    print(f"\nPath 1→6 exists: {g.has_path(1, 6)}")
    print(f"Shortest path 1→6: {g.shortest_path(1, 6)}")
    print(f"Connected components: {g.count_components()}")

    print("\n=== Knowledge Graph (AI Connection) ===")
    kg = Graph()
    kg.add_edge("Python", "AI")
    kg.add_edge("Python", "ML")
    kg.add_edge("AI", "ML")
    kg.add_edge("ML", "Deep Learning")
    kg.add_edge("Deep Learning", "NLP")
    kg.add_edge("NLP", "ChatGPT")

    path = kg.shortest_path("Python", "ChatGPT")
    print(f"Python → ChatGPT path: {path}")
