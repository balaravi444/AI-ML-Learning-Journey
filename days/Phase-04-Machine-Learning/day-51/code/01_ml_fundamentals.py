"""
Day 51 — ML Fundamentals + Scikit-learn
Topic: Core ML Concepts
Date: 08 July 2026
Author: Bala Ravi

Understanding what ML actually IS
before diving into algorithms!
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    mean_squared_error)
import warnings
warnings.filterwarnings('ignore')


def create_dataset(n: int = 500) -> pd.DataFrame:
    """Create student performance dataset."""
    np.random.seed(42)

    study_hours = np.random.uniform(1, 10, n)
    attendance = np.random.uniform(50, 100, n)
    prev_score = np.random.uniform(40, 90, n)
    assignments = np.random.randint(0, 10, n)
    sleep_hours = np.random.uniform(4, 9, n)

    # Final score depends on these features
    score = (
        study_hours * 4.5 +
        attendance * 0.3 +
        prev_score * 0.4 +
        assignments * 1.2 +
        sleep_hours * 1.5 +
        np.random.normal(0, 5, n)
    )
    score = np.clip(score, 0, 100)

    return pd.DataFrame({
        'study_hours': study_hours.round(1),
        'attendance_pct': attendance.round(1),
        'prev_score': prev_score.round(1),
        'assignments_done': assignments,
        'sleep_hours': sleep_hours.round(1),
        'final_score': score.round(1)
    })


def demonstrate_ml_workflow(
        df: pd.DataFrame) -> None:
    """
    Complete ML workflow from scratch!
    Every step explained!
    """
    print("=== Complete ML Workflow ===\n")

    # Step 1: Define problem
    print("Step 1: Define Problem")
    print("  Predict student final score "
          "(regression)")
    print(f"  Dataset: {df.shape[0]} students, "
          f"{df.shape[1]-1} features\n")

    # Step 2: Prepare data
    X = df.drop('final_score', axis=1)
    y = df['final_score']

    print("Step 2: Prepare Data")
    print(f"  Features: {list(X.columns)}")
    print(f"  Target: final_score")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}\n")

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    print("Step 3: Train/Test Split (80/20)")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}\n")

    # Step 4: Choose and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Step 4: Train Model")
    print("  model = LinearRegression()")
    print("  model.fit(X_train, y_train)")
    print("  ✅ Model trained!\n")

    # Step 5: Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(
        y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(
        y_test, y_pred_test))

    print("Step 5: Evaluate Model")
    print(f"  Train R²:  {train_r2:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")
    print(f"  Test MAE:  {test_mae:.2f} marks")
    print(f"  Test RMSE: {test_rmse:.2f} marks")

    gap = train_r2 - test_r2
    print(f"\n  Train-Test Gap: {gap:.4f}")
    if gap > 0.1:
        print("  ⚠️ Possible overfitting!")
    else:
        print("  ✅ Model generalizes well!")

    # Step 6: Make predictions
    print("\nStep 6: Predict on New Students")
    new_students = pd.DataFrame({
        'study_hours': [8.5, 3.0, 6.0],
        'attendance_pct': [95, 60, 80],
        'prev_score': [85, 55, 70],
        'assignments_done': [9, 3, 6],
        'sleep_hours': [7.5, 5.0, 7.0]
    })

    predictions = model.predict(new_students)
    for i, pred in enumerate(predictions):
        print(f"  Student {i+1}: "
              f"{pred:.1f}/100")


def demonstrate_overfitting() -> None:
    """
    Show overfitting vs underfitting
    with different model complexities.
    """
    print("\n=== Overfitting vs Underfitting ===\n")

    np.random.seed(42)
    n = 200

    X = np.random.uniform(0, 10, n).reshape(-1, 1)
    y = 2 * X.flatten() + np.random.normal(
        0, 3, n)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.3,
            random_state=42))

    print(f"{'Model':<30} | {'Train R²':>8} | "
          f"{'Test R²':>8} | {'Status':>20}")
    print("-" * 75)

    # Try different tree depths
    configs = [
        ("Linear Regression", None),
        ("Decision Tree (depth=2)", 2),
        ("Decision Tree (depth=5)", 5),
        ("Decision Tree (depth=10)", 10),
        ("Decision Tree (depth=20)", 20),
        ("Decision Tree (no limit)", None)
    ]

    for name, depth in configs:
        if "Linear" in name:
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor(
                max_depth=depth,
                random_state=42)

        model.fit(X_train, y_train)
        train_r2 = r2_score(
            y_train, model.predict(X_train))
        test_r2 = r2_score(
            y_test, model.predict(X_test))
        gap = train_r2 - test_r2

        if gap > 0.15:
            status = "⚠️ Overfitting!"
        elif train_r2 < 0.5 and test_r2 < 0.5:
            status = "❌ Underfitting!"
        else:
            status = "✅ Good fit"

        print(f"{name:<30} | {train_r2:>8.4f} | "
              f"{test_r2:>8.4f} | {status:>20}")

    print("\n💡 Key Insight:")
    print("   Depth=20 → perfect train, bad test!")
    print("   That's overfitting — memorized data!")
    print("   Depth=5 → balanced train and test! ✅")


def demonstrate_sklearn_api() -> None:
    """
    Show sklearn's consistent API pattern.
    Works the same for every algorithm!
    """
    print("\n=== Sklearn API Pattern ===\n")
    print("Every sklearn model: "
          "fit() → predict() → score()\n")

    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso)
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor)
    from sklearn.neighbors import (
        KNeighborsRegressor)

    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = 2*X[:, 0] + 3*X[:, 1] + np.random.randn(200)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=50, random_state=42),
        "KNN Regressor": KNeighborsRegressor(k=5)
    }

    print(f"{'Algorithm':<22} | {'R² Score':>8} | "
          f"{'MAE':>8}")
    print("-" * 45)

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            r2 = r2_score(
                y_test, model.predict(X_test))
            mae = mean_absolute_error(
                y_test, model.predict(X_test))
            print(f"{name:<22} | {r2:>8.4f} | "
                  f"{mae:>8.4f}")
        except Exception:
            pass

    print("\n✅ Same API for ALL algorithms!")
    print("   Change the class name = change algorithm!")


if __name__ == "__main__":
    df = create_dataset()
    print(f"Dataset: {df.shape}\n")

    demonstrate_ml_workflow(df)
    demonstrate_overfitting()
    demonstrate_sklearn_api()
