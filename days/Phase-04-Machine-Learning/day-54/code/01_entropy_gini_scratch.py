"""
Day 54 — Decision Trees
Topic: Entropy, Gini, Information Gain from Scratch
Date: 11 July 2026
Author: Bala Ravi

The math that decides HOW trees split data!
Understanding this = understanding all tree models:
Decision Trees, Random Forest, XGBoost!
"""
import numpy as np
from collections import Counter


def entropy(labels: list) -> float:
    """
    Calculate Shannon entropy of a label set.

    H = -Σ p(c) * log₂(p(c))

    Pure node (all same class) → entropy = 0
    Perfectly mixed (50/50)   → entropy = 1

    Args:
        labels: List of class labels

    Returns:
        Entropy value (0 to 1)
    """
    if len(labels) == 0:
        return 0.0

    counts = Counter(labels)
    total = len(labels)
    h = 0.0

    for count in counts.values():
        p = count / total
        if p > 0:
            h -= p * np.log2(p)

    return h


def gini_impurity(labels: list) -> float:
    """
    Calculate Gini impurity.

    Gini = 1 - Σ p(c)²

    Pure node  → Gini = 0
    Mixed node → Gini approaches 0.5

    Sklearn uses Gini by default (faster than entropy!)

    Args:
        labels: List of class labels

    Returns:
        Gini impurity (0 to 0.5 for binary)
    """
    if len(labels) == 0:
        return 0.0

    counts = Counter(labels)
    total = len(labels)

    return 1.0 - sum(
        (count / total) ** 2
        for count in counts.values())


def information_gain(
        parent_labels: list,
        left_labels: list,
        right_labels: list,
        criterion: str = 'entropy') -> float:
    """
    Calculate Information Gain of a split.

    IG = H(parent) - [weighted average H(children)]

    Best split = MAXIMUM information gain!

    Args:
        parent_labels: Labels before split
        left_labels: Labels in left branch
        right_labels: Labels in right branch
        criterion: 'entropy' or 'gini'

    Returns:
        Information gain value
    """
    metric = entropy if criterion == 'entropy' else gini_impurity

    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    if n == 0:
        return 0.0

    parent_impurity = metric(parent_labels)
    weighted_child = (
        (n_left / n) * metric(left_labels) +
        (n_right / n) * metric(right_labels))

    return parent_impurity - weighted_child


def find_best_split(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list) -> dict:
    """
    Find the best feature and threshold to split on.

    Tries every feature and every possible threshold.
    Picks the split with highest Information Gain!

    Args:
        X: Feature matrix
        y: Labels
        feature_names: Names of features

    Returns:
        Best split info dictionary
    """
    best_ig = -1
    best_split = {}
    parent_labels = y.tolist()

    for feature_idx in range(X.shape[1]):
        values = X[:, feature_idx]
        thresholds = np.unique(values)

        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            left_labels = y[left_mask].tolist()
            right_labels = y[right_mask].tolist()

            ig = information_gain(
                parent_labels,
                left_labels,
                right_labels)

            if ig > best_ig:
                best_ig = ig
                best_split = {
                    'feature_idx': feature_idx,
                    'feature_name': feature_names[feature_idx],
                    'threshold': threshold,
                    'information_gain': ig,
                    'left_size': left_mask.sum(),
                    'right_size': right_mask.sum()
                }

    return best_split


def demonstrate_entropy() -> None:
    """Show entropy for different distributions."""
    print("=== Entropy Demonstration ===\n")

    scenarios = [
        ([1, 1, 1, 1, 1], "Pure (all PASS)"),
        ([0, 0, 0, 0, 0], "Pure (all FAIL)"),
        ([1, 1, 1, 0, 0], "Slightly mixed (3P, 2F)"),
        ([1, 1, 0, 0, 0], "Mostly mixed (2P, 3F)"),
        ([1, 0, 1, 0, 1], "Perfectly mixed (3P, 2F)"),
        ([1, 0, 1, 0], "Perfectly 50/50")
    ]

    print(f"{'Labels':<25} | {'Entropy':>8} | "
          f"{'Gini':>8} | {'Description'}")
    print("-" * 70)

    for labels, desc in scenarios:
        e = entropy(labels)
        g = gini_impurity(labels)
        print(f"{str(labels):<25} | {e:>8.4f} | "
              f"{g:>8.4f} | {desc}")

    print(f"\n💡 Key Insight:")
    print(f"   Entropy = 0 → Pure node (perfect!)")
    print(f"   Entropy = 1 → Maximum disorder")
    print(f"   We want splits that REDUCE entropy!")


def demonstrate_information_gain() -> None:
    """Show information gain for different splits."""
    print("\n=== Information Gain Demo ===\n")
    print("Parent node: 10 students (5 pass, 5 fail)")
    print("Trying different splits...\n")

    parent = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    splits = [
        ("Random split", [1, 0, 1, 0], [1, 0, 1, 0]),
        ("Good split",   [1, 1, 1, 1], [0, 0, 0, 0, 1]),
        ("Perfect split",[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]),
        ("Bad split",    [1, 0, 1, 0, 1], [0, 1, 0])
    ]

    print(f"{'Split Type':<20} | {'Parent H':>9} | "
          f"{'Child H':>9} | {'IG':>8}")
    print("-" * 55)

    for split_name, left, right in splits:
        parent_h = entropy(parent)
        n = len(parent)
        child_h = (
            (len(left)/n) * entropy(left) +
            (len(right)/n) * entropy(right))
        ig = parent_h - child_h
        print(f"{split_name:<20} | {parent_h:>9.4f} | "
              f"{child_h:>9.4f} | {ig:>8.4f}")

    print(f"\n✅ Perfect split → IG = {entropy(parent):.4f}")
    print(f"   Always pick the split with MAX IG!")


def find_best_split_demo() -> None:
    """Demo finding best split on student data."""
    print("\n=== Finding Best Split ===\n")

    np.random.seed(42)
    n = 100

    X = np.column_stack([
        np.random.uniform(1, 10, n),   # study_hours
        np.random.uniform(40, 100, n), # attendance
        np.random.uniform(30, 95, n)   # prev_score
    ])

    # Pass if study > 6 AND attendance > 70
    passed = (
        (X[:, 0] > 6) & (X[:, 1] > 70)).astype(int)

    feature_names = [
        'study_hours', 'attendance_pct', 'prev_score']

    print("Searching for best split...")
    best = find_best_split(X, passed, feature_names)

    print(f"\nBest Split Found:")
    print(f"  Feature:          {best['feature_name']}")
    print(f"  Threshold:        {best['threshold']:.2f}")
    print(f"  Information Gain: {best['information_gain']:.4f}")
    print(f"  Left branch:      {best['left_size']} samples")
    print(f"  Right branch:     {best['right_size']} samples")
    print(f"\n✅ Algorithm correctly identified the")
    print(f"   most important splitting feature!")


if __name__ == "__main__":
    demonstrate_entropy()
    demonstrate_information_gain()
    find_best_split_demo()
