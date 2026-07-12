"""
Day 27 — DSA: Graphs
Topic: Knowledge Graph — Real AI Application!
Date: 14 June 2026
Author: Bala Ravi

This is a simplified version of how
knowledge graphs power LLMs like ChatGPT!

Real World Connection:
    Google's Knowledge Graph has 500+ billion facts!
    LLMs use knowledge graphs for reasoning!
    Drug discovery uses molecular graphs + GNNs!
"""
from collections import deque


class KnowledgeGraph:
    """
    Knowledge graph for AI/ML domain.
    Simplified version of how LLMs store knowledge!
    """

    def __init__(self) -> None:
        """Initialize empty knowledge graph."""
        self.graph: dict[str, list[str]] = {}
        self.relations: dict[tuple, str] = {}

    def add_fact(self,
                 entity1: str,
                 relation: str,
                 entity2: str) -> None:
        """
        Add a fact to knowledge graph.
        Format: entity1 [relation] entity2

        Args:
            entity1: Source entity
            relation: Relationship type
            entity2: Target entity
        """
        if entity1 not in self.graph:
            self.graph[entity1] = []
        if entity2 not in self.graph:
            self.graph[entity2] = []

        self.graph[entity1].append(entity2)
        self.relations[(entity1, entity2)] = relation

    def find_connection(self,
                        start: str,
                        end: str) -> list[str]:
        """
        Find shortest connection between concepts.
        How LLMs reason about relationships!

        Args:
            start: Starting concept
            end: Target concept

        Returns:
            Shortest path between concepts
        """
        if start not in self.graph:
            return []

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            node, path = queue.popleft()

            if node == end:
                return path

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor,
                                  path + [neighbor]))

        return []

    def get_all_related(self, concept: str) -> list[str]:
        """
        Get all concepts related to given concept.
        Uses BFS to find all reachable concepts.

        Args:
            concept: Starting concept

        Returns:
            All related concepts
        """
        if concept not in self.graph:
            return []

        visited = set()
        queue = deque([concept])
        related = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node != concept:
                    related.append(node)
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

        return related

    def explain_connection(self,
                           start: str,
                           end: str) -> str:
        """
        Explain the connection between two concepts.

        Args:
            start: Starting concept
            end: Target concept

        Returns:
            Human readable explanation
        """
        path = self.find_connection(start, end)

        if not path:
            return f"No connection found between {start} and {end}"

        explanation = []
        for i in range(len(path) - 1):
            relation = self.relations.get(
                (path[i], path[i + 1]), "related to")
            explanation.append(
                f"{path[i]} --[{relation}]--> {path[i+1]}")

        return " | ".join(explanation)


if __name__ == "__main__":
    print("=== Knowledge Graph — AI Domain ===\n")

    kg = KnowledgeGraph()

    # Add AI/ML knowledge
    facts = [
        ("Python", "used_for", "AI"),
        ("Python", "used_for", "ML"),
        ("Python", "used_for", "Data Science"),
        ("AI", "includes", "ML"),
        ("AI", "includes", "NLP"),
        ("AI", "includes", "Computer Vision"),
        ("ML", "includes", "Deep Learning"),
        ("ML", "uses", "NumPy"),
        ("ML", "uses", "Pandas"),
        ("ML", "uses", "Scikit-learn"),
        ("Deep Learning", "uses", "TensorFlow"),
        ("Deep Learning", "uses", "PyTorch"),
        ("NLP", "powers", "ChatGPT"),
        ("NLP", "powers", "BERT"),
        ("Transformer", "is_basis_of", "ChatGPT"),
        ("Transformer", "is_basis_of", "BERT"),
        ("Deep Learning", "uses", "Transformer"),
    ]

    for e1, rel, e2 in facts:
        kg.add_fact(e1, rel, e2)

    print("=== Finding Connections ===")
    connections = [
        ("Python", "ChatGPT"),
        ("Python", "TensorFlow"),
        ("ML", "BERT"),
    ]

    for start, end in connections:
        explanation = kg.explain_connection(start, end)
        print(f"\n{start} → {end}:")
        print(f"  {explanation}")

    print("\n=== All Concepts Related to ML ===")
    related = kg.get_all_related("ML")
    print(f"ML is related to: {related}")
