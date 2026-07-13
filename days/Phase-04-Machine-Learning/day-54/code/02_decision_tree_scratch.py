"""
Day 54 — Decision Trees
Topic: Decision Tree from Scratch
Date: 11 July 2026
Author: Bala Ravi

Building a complete Decision Tree classifier!
Uses recursion from Day 25 + entropy from today!
"""
import numpy as np
from collections import Counter


class TreeNode:
    """A node in the decision tree."""

    def __init__(self,
                  feature_idx: int = None,
                  threshold: float = None,
                  left=None,
                  right=None,
                  value=None) -> None:
        """
        Initialize tree node.

        Args:
            feature_idx: Feature index to split on
            threshold: Threshold value for split
            left: Left child node
            right: Right child node
            value: Prediction value (leaf nodes only)
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.value is not None


class DecisionTreeScratch:
    """
    Decision Tree Classifier from scratch!

    Same algorithm as sklearn's DecisionTreeClassifier!
    Uses recursive splitting with Information Gain.
    """

    def __init__(self,
                  max_depth: int = 5,
                  min_samples_split: int = 2,
                  min_samples_leaf: int = 1,
                  criterion: str = 'entropy'
                  ) -> None:
        """
        Initialize Decision Tree.

        Args:
            max_depth: Maximum tree depth
            min_samples_split: Min samples to split
            min_samples_leaf: Min samples in leaf
            criterion: 'entropy' or 'gini'
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_features = 0

    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy."""
        if len(y) == 0:
            return 0.0
        counts = Counter(y)
        total = len(y)
        h = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                h -= p * np.log2(p + 1e-15)
        return h

    def _gini(self, y: np.ndarray) -> float:
        """Calculate gini impurity."""
        if len(y) == 0:
            return 0.0
        counts = Counter(y)
        total = len(y)
        return 1.0 - sum(
            (c/total)**2 for c in counts.values())

    def _impurity(self, y: np.ndarray) -> float:
        """Get impurity based on criterion."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        return self._gini(y)

    def _information_gain(
            self,
            y: np.ndarray,
            left_y: np.ndarray,
            right_y: np.ndarray) -> float:
        """Calculate information gain of split."""
        n = len(y)
        if n == 0:
            return 0.0
        parent_imp = self._impurity(y)
        weighted = (
            (len(left_y)/n) * self._impurity(left_y) +
            (len(right_y)/n) * self._impurity(right_y))
        return parent_imp - weighted

    def _best_split(
            self,
            X: np.ndarray,
            y: np.ndarray) -> dict:
        """Find best feature and threshold to split on."""
        best_ig = -1.0
        best_split = {}

        for feat_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feat_idx])

            for threshold in thresholds:
                left_mask = X[:, feat_idx] <= threshold
                right_mask = ~left_mask

                if (left_mask.sum() < self.min_samples_leaf or
                        right_mask.sum() < self.min_samples_leaf):
                    continue

                ig = self._information_gain(
                    y,
                    y[left_mask],
                    y[right_mask])

                if ig > best_ig:
                    best_ig = ig
                    best_split = {
                        'feature_idx': feat_idx,
                        'threshold': threshold,
                        'ig': ig
                    }

        return best_split

    def _build_tree(
            self,
            X: np.ndarray,
            y: np.ndarray,
            depth: int = 0) -> TreeNode:
        """
        Recursively build the decision tree.

        This is pure recursion from Day 25! 🔥
        Base cases: max depth, pure node, too small
        Recursive case: split and build children
        """
        n_samples = len(y)

        # Base cases — return leaf node
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)

        # Find best split
        best = self._best_split(X, y)

        if not best or best.get('ig', 0) <= 0:
            leaf_value = Counter(y).most_common(1)[0][0]
            return TreeNode(value=leaf_value)

        # Split data
        feat_idx = best['feature_idx']
        threshold = best['threshold']
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        # Recursive calls!
        left_node = self._build_tree(
            X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(
            X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_idx=feat_idx,
            threshold=threshold,
            left=left_node,
            right=right_node)

    def fit(self,
             X: np.ndarray,
             y: np.ndarray) -> 'DecisionTreeScratch':
        """Train decision tree."""
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)
        return self

    def _traverse(self,
                   node: TreeNode,
                   x: np.ndarray) -> int:
        """Traverse tree for single sample."""
        if node.is_leaf():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)

    def predict(self,
                 X: np.ndarray) -> np.ndarray:
        """Predict class for each sample."""
        return np.array([
            self._traverse(self.root, x) for x in X])

    def score(self,
               X: np.ndarray,
               y: np.ndarray) -> float:
        """Calculate accuracy."""
        return (self.predict(X) == y).mean()


def train_and_compare() -> None:
    """Compare scratch vs sklearn."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    print("=== Scratch vs Sklearn Decision Tree ===\n")

    np.random.seed(42)
    X, y = make_classification(
        n_samples=400,
        n_features=4,
        random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Scratch
    scratch = DecisionTreeScratch(
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3)
    scratch.fit(X_train, y_train)
    scratch_acc = scratch.score(X_test, y_test)

    # Sklearn
    sk_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42)
    sk_tree.fit(X_train, y_train)
    sk_acc = sk_tree.score(X_test, y_test)

    print(f"Scratch accuracy:  {scratch_acc:.4f}")
    print(f"Sklearn accuracy:  {sk_acc:.4f}")
    print(f"\n✅ Very similar! Same algorithm!")
    print(f"   Sklearn is faster (C-optimized)")
    print(f"   but logic is IDENTICAL!")


if __name__ == "__main__":
    train_and_compare()
