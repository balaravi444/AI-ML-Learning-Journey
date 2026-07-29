"""
Day 69 — Autonomous Data Scientist
Topic: SHAP Explainability
Date: 26 July 2026
Author: Bala Ravi

SHAP = SHapley Additive exPlanations
Why did the model predict X?
Global + local explanations for any model!
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed.")
    print("    Run: pip install shap\n")


def create_churn_dataset(
        n: int = 500) -> tuple:
    """Create employee churn dataset."""
    np.random.seed(42)

    satisfaction = np.random.uniform(1, 5, n)
    performance = np.random.uniform(1, 5, n)
    years_exp = np.random.exponential(5, n).clip(0, 20)
    salary = np.clip(
        400000 + years_exp * 40000 +
        np.random.normal(0, 80000, n),
        200000, 2000000)
    overtime = np.random.choice([0, 1], n, p=[0.6, 0.4])
    num_projects = np.random.randint(1, 8, n)
    distance_km = np.random.exponential(15, n).clip(1, 60)

    churn_prob = np.clip(
        0.1 +
        0.15 * (satisfaction < 2.5) +
        0.12 * overtime +
        0.08 * (salary < 500000) +
        0.06 * (distance_km > 30) +
        0.05 * (performance < 2.0),
        0, 0.85)
    churn = np.random.binomial(1, churn_prob)

    feature_names = [
        'satisfaction_score', 'performance_score',
        'years_experience', 'salary',
        'overtime', 'num_projects', 'distance_km']

    X = np.column_stack([
        satisfaction, performance, years_exp,
        salary, overtime, num_projects, distance_km])
    y = churn

    return X, y, feature_names


def compute_shap_values(
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: list) -> dict:
    """
    Compute SHAP values for a trained model.

    Args:
        model: Trained sklearn model
        X_train: Training data (for background)
        X_test: Test data to explain
        feature_names: Feature names

    Returns:
        Dictionary with SHAP values and summary
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available — showing simulation")
        # Simulate SHAP values for demo
        n_features = X_test.shape[1]
        shap_values = np.random.randn(
            len(X_test), n_features) * 0.1
        return {
            'shap_values': shap_values,
            'expected_value': 0.15,
            'feature_names': feature_names,
            'available': False
        }

    # TreeExplainer for tree models (fast!)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification — take class 1
        if isinstance(shap_values, list):
            shap_values_use = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_values_use = shap_values
            expected_value = explainer.expected_value

        return {
            'shap_values': shap_values_use,
            'expected_value': float(expected_value),
            'feature_names': feature_names,
            'available': True,
            'explainer': explainer
        }
    except Exception:
        # KernelExplainer fallback (model-agnostic)
        background = shap.sample(X_train, 50)
        explainer = shap.KernelExplainer(
            model.predict_proba, background)
        shap_values = explainer.shap_values(
            X_test[:20])
        return {
            'shap_values': shap_values[1],
            'expected_value': float(
                explainer.expected_value[1]),
            'feature_names': feature_names,
            'available': True
        }


def global_feature_importance(
        shap_data: dict) -> None:
    """Show global feature importance via SHAP."""
    print("=== Global Feature Importance (SHAP) ===\n")
    print("Mean |SHAP value| per feature")
    print("Higher = more important overall\n")

    shap_values = shap_data['shap_values']
    feature_names = shap_data['feature_names']

    # Mean absolute SHAP per feature
    mean_shap = np.abs(shap_values).mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_shap)[::-1]

    print(f"{'Feature':<25} | "
          f"{'Mean |SHAP|':>12} | "
          f"{'Importance Bar'}")
    print("-" * 60)

    max_val = mean_shap.max()
    for idx in sorted_idx:
        name = feature_names[idx]
        val = mean_shap[idx]
        bar = '█' * int(val / max_val * 30)
        print(f"{name:<25} | "
              f"{val:>12.4f} | "
              f"{bar}")

    print(f"\n💡 satisfaction_score and overtime")
    print(f"   are the biggest drivers of churn!")
    print(f"   HR should focus there first!")


def local_explanation(
        shap_data: dict,
        X_test: np.ndarray,
        predictions: np.ndarray,
        row_idx: int = 0) -> None:
    """Show SHAP explanation for one prediction."""
    print(f"\n=== Local Explanation: "
          f"Employee #{row_idx} ===\n")

    shap_values = shap_data['shap_values']
    expected_value = shap_data['expected_value']
    feature_names = shap_data['feature_names']

    row_shap = shap_values[row_idx]
    row_features = X_test[row_idx]
    prediction = predictions[row_idx]

    print(f"Prediction: {'CHURN' if prediction == 1 else 'STAY'}")
    print(f"Base rate (avg churn): "
          f"{expected_value:.3f} ({expected_value*100:.1f}%)\n")

    print(f"{'Feature':<25} | "
          f"{'Value':>10} | "
          f"{'SHAP':>8} | "
          f"{'Direction':>12}")
    print("-" * 65)

    # Sort by absolute SHAP
    sorted_idx = np.argsort(np.abs(row_shap))[::-1]

    for idx in sorted_idx:
        name = feature_names[idx]
        feat_val = row_features[idx]
        shap_val = row_shap[idx]

        direction = (
            "↑ toward CHURN"
            if shap_val > 0.01 else
            "↓ away from CHURN"
            if shap_val < -0.01 else
            "→ neutral")

        print(f"{name:<25} | "
              f"{feat_val:>10.3f} | "
              f"{shap_val:>8.4f} | "
              f"{direction}")

    total = expected_value + row_shap.sum()
    print(f"\n  Base rate:     {expected_value:.4f}")
    print(f"  SHAP sum:     {row_shap.sum():+.4f}")
    print(f"  Final score:  {total:.4f}")
    print(f"  Prediction:   "
          f"{'CHURN' if prediction == 1 else 'STAY'}")


def demonstrate_shap() -> None:
    """Full SHAP demonstration."""
    print("=== SHAP Explainability Demo ===\n")

    X, y, feature_names = create_churn_dataset()
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42, stratify=y))

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.4f}\n")

    # Compute SHAP
    shap_data = compute_shap_values(
        model, X_train, X_test, feature_names)

    # Global importance
    global_feature_importance(shap_data)

    # Compare with RF built-in importance
    print(f"\n=== RF Built-in vs SHAP Importance ===\n")
    rf_imp = model.feature_importances_
    shap_imp = np.abs(
        shap_data['shap_values']).mean(axis=0)

    print(f"{'Feature':<25} | "
          f"{'RF Built-in':>12} | "
          f"{'SHAP':>8}")
    print("-" * 50)

    for i, name in enumerate(feature_names):
        print(f"{name:<25} | "
              f"{rf_imp[i]:>12.4f} | "
              f"{shap_imp[i]:>8.4f}")

    print(f"\n💡 SHAP is more reliable than RF importance!")
    print(f"   RF importance has bias toward high-cardinality features")
    print(f"   SHAP is unbiased and model-agnostic! ✅")

    # Local explanations
    predictions = model.predict(X_test)
    local_explanation(
        shap_data, X_test, predictions, row_idx=0)

    # Find interesting case — high confidence churn
    churn_proba = model.predict_proba(X_test)[:, 1]
    high_churn_idx = np.argmax(churn_proba)
    print(f"\nHighest churn risk employee:")
    local_explanation(
        shap_data, X_test, predictions,
        row_idx=high_churn_idx)


if __name__ == "__main__":
    demonstrate_shap()
