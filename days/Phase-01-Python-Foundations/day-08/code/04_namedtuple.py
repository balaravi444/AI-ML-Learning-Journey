"""
Day 08 — Collections Module
Topic: namedtuple for readable data structures
Date: 26 May 2026
Author: Bala Ravi

Real World Connection:
    namedtuple is more readable than tuple and
    more memory efficient than class!
    Used for storing ML model evaluation metrics.
"""
from collections import namedtuple


# Define namedtuples
Student = namedtuple("Student", ["name", "age", "course", "marks"])
ModelMetrics = namedtuple("ModelMetrics", ["accuracy", "precision", "recall", "f1"])


def evaluate_model(
    correct: int,
    total: int,
    true_pos: int,
    false_pos: int,
    false_neg: int
) -> ModelMetrics:
    """
    Calculate ML model evaluation metrics.

    Args:
        correct: Number of correct predictions
        total: Total predictions
        true_pos: True positives
        false_pos: False positives
        false_neg: False negatives

    Returns:
        ModelMetrics namedtuple with all metrics
    """
    accuracy = correct / total
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * (precision * recall) / (precision + recall)

    return ModelMetrics(
        accuracy=round(accuracy, 3),
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3)
    )


if __name__ == "__main__":
    print("=== Student Records ===")
    students = [
        Student("Bala", 20, "AI Engineering", 85),
        Student("Ravi", 21, "Data Science", 92),
        Student("Kumar", 19, "ML", 78)
    ]

    for s in students:
        print(f"Name: {s.name} | Course: {s.course} | Marks: {s.marks}")

    print("\n=== ML Model Metrics ===")
    metrics = evaluate_model(
        correct=95,
        total=100,
        true_pos=45,
        false_pos=3,
        false_neg=5
    )
    print(f"Accuracy  : {metrics.accuracy}")
    print(f"Precision : {metrics.precision}")
    print(f"Recall    : {metrics.recall}")
    print(f"F1 Score  : {metrics.f1}")
