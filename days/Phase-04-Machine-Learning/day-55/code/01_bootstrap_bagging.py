"""
Day 55 — Random Forest & Ensemble Methods
Topic: Bootstrap Sampling and Bagging from Scratch
Date: 12 July 2026
Author: Bala Ravi

Understanding WHY Random Forest works!
Bootstrap + Random features = decorrelated trees
Decorrelated trees + voting = better accuracy!
"""
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def demonstrate_bootstrap() -> None:
    """Show bootstrap sampling concept."""
    print("=== Bootstrap Sampling ===\n")

    data = list(range(10))
    n_bootstraps = 5

    print(f"Original data: {data}")
    print(f"Size: {len(data)}\n")

    oob_counts = Counter()

    for i in range(n_bootstraps):
        bootstrap = list(
            np.random.choice(data, size=len(data),
                             replace=True))
        oob = [x for x in data if x not in bootstrap]

        print(f"Bootstrap {i+1}: {bootstrap}")
        print(f"  OOB (not seen): {oob}")

        for x in oob:
            oob_counts[x] += 1

    print(f"\nOn average each sample is OOB")
    print(f"in ~{1 - (1-1/len(data))**len(data)*100:.0f}% "
          f"of trees (theoretically 36.8%)")
    print(f"\nThis is the FREE validation!")
    print(f"No need for separate val set! 🔥")


class BaggingClassifierScratch:
    """
    Bagging from scratch.
    Same idea as sklearn's BaggingClassifier!
    Base of Random Forest.
    """

    def __init__(self,
                 n_estimators: int = 10,
                 max_depth: int = 5,
                 random_state: int = 42) -> None:
        """Initialize bagging classifier."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.oob_predictions = {}

    def fit(self,
             X: np.ndarray,
             y: np.ndarray) -> 'BaggingClassifierScratch':
        """Train ensemble of trees on bootstrap samples."""
        np.random.seed(self.random_state)
        n = len(X)
        self.n_samples = n

        # Store OOB predictions for each sample
        oob_votes = {i: [] for i in range(n)}

        for t in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(
                n, size=n, replace=True)
            oob_indices = list(
                set(range(n)) - set(indices))

            X_boot = X[indices]
            y_boot = y[indices]

            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=t)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            # OOB predictions
            if oob_indices:
                oob_preds = tree.predict(
                    X[oob_indices])
                for idx, pred in zip(
                        oob_indices, oob_preds):
                    oob_votes[idx].append(pred)

        # OOB score
        oob_correct = 0
        oob_total = 0
        for idx, votes in oob_votes.items():
            if votes:
                majority = Counter(votes).most_common(1)[0][0]
                if majority == y[idx]:
                    oob_correct += 1
                oob_total += 1

        self.oob_score_ = (oob_correct / oob_total
                           if oob_total > 0 else 0)
        return self

    def predict(self,
                 X: np.ndarray) -> np.ndarray:
        """Majority vote from all trees."""
        all_preds = np.array([
            tree.predict(X) for tree in self.trees])

        # Majority vote for each sample
        result = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            majority = Counter(votes).most_common(1)[0][0]
            result.append(majority)

        return np.array(result)

    def score(self,
               X: np.ndarray,
               y: np.ndarray) -> float:
        """Calculate accuracy."""
        return accuracy_score(y, self.predict(X))


def bagging_vs_single_tree() -> None:
    """Compare single tree vs bagging."""
    print("\n=== Single Tree vs Bagging ===\n")

    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Single tree
    single = DecisionTreeClassifier(
        random_state=42)
    single.fit(X_train, y_train)
    single_train = single.score(X_train, y_train)
    single_test = single.score(X_test, y_test)

    # Bagging from scratch
    bagging = BaggingClassifierScratch(
        n_estimators=50,
        max_depth=None,
        random_state=42)
    bagging.fit(X_train, y_train)
    bagging_train = bagging.score(X_train, y_train)
    bagging_test = bagging.score(X_test, y_test)

    print(f"{'Model':<25} | {'Train Acc':>10} | "
          f"{'Test Acc':>9} | {'OOB':>8}")
    print("-" * 60)
    print(f"{'Single Decision Tree':<25} | "
          f"{single_train:>10.4f} | "
          f"{single_test:>9.4f} | {'N/A':>8}")
    print(f"{'Bagging (50 trees)':<25} | "
          f"{bagging_train:>10.4f} | "
          f"{bagging_test:>9.4f} | "
          f"{bagging.oob_score_:>8.4f}")

    improvement = bagging_test - single_test
    print(f"\n✅ Bagging improves test accuracy by "
          f"{improvement*100:.1f}%!")
    print(f"   Errors cancel out when trees vote! 🔥")


if __name__ == "__main__":
    demonstrate_bootstrap()
    bagging_vs_single_tree()
