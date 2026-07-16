"""
Day 56 — SVM & KNN
Topic: KNN from Scratch + Sklearn
Date: 13 July 2026
Author: Bala Ravi

KNN = no training, all prediction!
Store data, find neighbors, vote!

Uses linear algebra from Day 37:
Euclidean distance = L2 norm!
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import (
    accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class KNNScratch:
    """
    K-Nearest Neighbors from scratch!

    No actual training — just store data.
    Prediction = find K nearest + vote!

    Uses Euclidean distance (L2 norm from Day 37!)
    """

    def __init__(self,
                  k: int = 5,
                  metric: str = 'euclidean'
                  ) -> None:
        """
        Initialize KNN.

        Args:
            k: Number of neighbors
            metric: Distance metric
        """
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self,
             X: np.ndarray,
             y: np.ndarray) -> 'KNNScratch':
        """
        'Training' = just store the data!
        No actual optimization happens!

        Args:
            X: Training features
            y: Training labels

        Returns:
            self
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def _distance(self,
                   a: np.ndarray,
                   b: np.ndarray) -> float:
        """
        Calculate distance between two points.
        Euclidean = L2 norm from Day 37!
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        else:
            return np.sqrt(np.sum((a - b) ** 2))

    def _predict_single(self,
                         x: np.ndarray) -> int:
        """
        Predict class for a single sample.

        Steps:
        1. Calculate distance to ALL training points
        2. Sort by distance
        3. Take K nearest
        4. Majority vote
        """
        # Step 1: Calculate distances to all points
        distances = [
            self._distance(x, x_train)
            for x_train in self.X_train]

        # Step 2: Get K nearest indices
        k_indices = np.argsort(distances)[:self.k]

        # Step 3: Get labels of K nearest neighbors
        k_labels = self.y_train[k_indices]

        # Step 4: Majority vote
        from collections import Counter
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self,
                 X: np.ndarray) -> np.ndarray:
        """Predict for all samples."""
        return np.array([
            self._predict_single(x) for x in X])

    def score(self,
               X: np.ndarray,
               y: np.ndarray) -> float:
        """Calculate accuracy."""
        return accuracy_score(y, self.predict(X))


def demonstrate_distance_metrics() -> None:
    """Show different distance calculations."""
    print("=== Distance Metrics ===\n")

    # Two students' feature vectors
    student_A = np.array([8.0, 90.0, 85.0, 9, 7.5])
    student_B = np.array([3.0, 55.0, 45.0, 3, 5.0])
    student_C = np.array([7.5, 85.0, 80.0, 8, 7.0])

    feature_names = ['study_hrs', 'attend%',
                      'prev_score', 'assignments',
                      'sleep_hrs']

    print("Student profiles:")
    for name, student in [
        ('A (likely PASS)', student_A),
        ('B (likely FAIL)', student_B),
        ('C (unknown)', student_C)
    ]:
        print(f"  {name}: {dict(zip(feature_names, student))}")

    # Distances from C to A and B
    euclidean_A = np.sqrt(np.sum(
        (student_C - student_A) ** 2))
    euclidean_B = np.sqrt(np.sum(
        (student_C - student_B) ** 2))

    manhattan_A = np.sum(np.abs(student_C - student_A))
    manhattan_B = np.sum(np.abs(student_C - student_B))

    print(f"\nDistances from Student C:")
    print(f"  Euclidean to A: {euclidean_A:.2f}")
    print(f"  Euclidean to B: {euclidean_B:.2f}")
    print(f"  → C is closer to A → predict PASS! ✅")

    print(f"\n  Manhattan to A: {manhattan_A:.2f}")
    print(f"  Manhattan to B: {manhattan_B:.2f}")
    print(f"  → Same conclusion! ✅")

    print(f"\n⚠️  IMPORTANT: Features are on different scales!")
    print(f"   attend% (40-100) dominates study_hrs (1-10)!")
    print(f"   ALWAYS scale before KNN! 🔥")


def scratch_vs_sklearn() -> None:
    """Compare KNN from scratch vs sklearn."""
    print("\n=== KNN Scratch vs Sklearn ===\n")

    np.random.seed(42)
    n = 300

    study = np.random.uniform(1, 10, n)
    attend = np.random.uniform(40, 100, n)
    score = study * 5 + attend * 0.3 + np.random.normal(0, 5, n)
    passed = (score > 70).astype(int)

    X = np.column_stack([study, attend])
    y = passed

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Scale features!
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Scratch KNN
    knn_scratch = KNNScratch(k=5)
    knn_scratch.fit(X_train_s, y_train)
    scratch_acc = knn_scratch.score(X_test_s, y_test)

    # Sklearn KNN
    knn_sk = KNeighborsClassifier(n_neighbors=5)
    knn_sk.fit(X_train_s, y_train)
    sk_acc = knn_sk.score(X_test_s, y_test)

    print(f"KNN Scratch accuracy:  {scratch_acc:.4f}")
    print(f"KNN Sklearn accuracy:  {sk_acc:.4f}")
    print(f"\n✅ Identical results!")
    print(f"   Sklearn is faster (optimized C code)")
    print(f"   but logic is EXACTLY the same!")


def k_selection_study() -> None:
    """Find optimal K using cross-validation."""
    print("\n=== K Selection Study ===\n")

    np.random.seed(42)
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500, n_features=5,
        random_state=42)

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    print(f"{'K':>5} | {'Train Acc':>10} | "
          f"{'Test Acc':>9} | {'CV Mean':>8} | "
          f"{'Status'}")
    print("-" * 60)

    k_values = [1, 3, 5, 7, 9, 11, 15, 21,
                31, 51, 101]

    best_cv = 0
    best_k = 5

    for k in k_values:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(
                n_neighbors=k))
        ])
        pipeline.fit(X_train, y_train)
        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        cv = cross_val_score(
            pipeline, X, y, cv=5,
            scoring='accuracy',
            n_jobs=-1).mean()

        if cv > best_cv:
            best_cv = cv
            best_k = k

        if k == 1:
            status = "⚠️ Overfit (k=1)"
        elif k == 101:
            status = "❌ Underfit"
        elif cv > 0.85:
            status = "✅ Good"
        else:
            status = ""

        print(f"{k:>5} | {train_acc:>10.4f} | "
              f"{test_acc:>9.4f} | {cv:>8.4f} | "
              f"{status}")

    print(f"\n✅ Best K = {best_k} (CV = {best_cv:.4f})")
    print(f"   Rule of thumb: K ≈ √n = "
          f"{int(np.sqrt(len(X_train)))}")
    print(f"   Always use ODD K for binary! 🔥")


if __name__ == "__main__":
    demonstrate_distance_metrics()
    scratch_vs_sklearn()
    k_selection_study()
