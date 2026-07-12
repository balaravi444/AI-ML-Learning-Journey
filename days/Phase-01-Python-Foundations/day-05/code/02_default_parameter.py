"""
Day 05 — Advanced Functions
Topic: Default Parameter Function
Date: 23 May 2026
Author: Bala Ravi

Real World Connection:
    Scikit-learn uses default parameters everywhere!
    model = RandomForestClassifier(n_estimators=100)
    n_estimators=100 is a default parameter!
"""


def greet(name: str = "Bala") -> None:
    """
    Greet a person by name.
    Uses default parameter if no name provided.

    Args:
        name: Name to greet. Defaults to "Bala"

    Returns:
        None — prints greeting directly
    """
    print(f"Hello, {name}!")


if __name__ == "__main__":
    greet()           # uses default
    greet("Ravi")     # overrides default
    greet("Kumar")    # overrides default
