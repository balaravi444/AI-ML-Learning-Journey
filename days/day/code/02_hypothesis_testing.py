"""
Day 45 — Statistics for ML
Topic: Hypothesis Testing
Date: 02 July 2026
Author: Bala Ravi

Hypothesis testing = making data-driven decisions!
Used in: A/B testing, feature selection,
         model comparison, business decisions!
"""
import numpy as np
import pandas as pd
from scipy import stats


def ttest_demo() -> None:
    """
    T-test — compare means of two groups.
    Most common hypothesis test in ML!
    """
    print("=== T-Test — Group Comparison ===\n")

    np.random.seed(42)

    # Question: Does remote work pay more?
    remote_salary = np.random.normal(23, 4, 50)
    onsite_salary = np.random.normal(20, 4, 50)

    print("Hypothesis:")
    print("  H0 (Null): Remote and onsite "
          "salaries are EQUAL")
    print("  H1 (Alt):  Remote pays MORE "
          "than onsite\n")

    print(f"Remote salary:  "
          f"₹{remote_salary.mean():.1f} ± "
          f"₹{remote_salary.std():.1f} LPA")
    print(f"Onsite salary:  "
          f"₹{onsite_salary.mean():.1f} ± "
          f"₹{onsite_salary.std():.1f} LPA\n")

    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(
        remote_salary, onsite_salary)

    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value:     {p_value:.4f}")
    print()

    if p_value < 0.05:
        print("✅ Reject H0 — "
              "Remote DOES pay significantly more!")
        print(f"   Difference: "
              f"₹{remote_salary.mean() - onsite_salary.mean():.1f} LPA")
    else:
        print("❌ Fail to reject H0 — "
              "No significant difference")

    print(f"\n📌 Significance level α = 0.05")
    print(f"   p={p_value:.4f} < 0.05 → significant!")


def ab_test_models() -> None:
    """
    A/B Test — compare two ML models!
    Data-driven model selection!
    """
    print("\n=== A/B Testing ML Models ===\n")

    np.random.seed(42)

    # Model performance across 10 cross-validation folds
    model_a = np.random.normal(0.84, 0.02, 10)
    model_b = np.random.normal(0.88, 0.02, 10)

    print("Model A (Random Forest):")
    print(f"  Scores: {model_a.round(3)}")
    print(f"  Mean: {model_a.mean():.4f} "
          f"± {model_a.std():.4f}")

    print("\nModel B (XGBoost):")
    print(f"  Scores: {model_b.round(3)}")
    print(f"  Mean: {model_b.mean():.4f} "
          f"± {model_b.std():.4f}")

    # Paired t-test (same folds = paired!)
    t_stat, p_value = stats.ttest_rel(
        model_b, model_a)

    print(f"\nPaired t-test result:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")

    if p_value < 0.05:
        better = ("Model B" if
                  model_b.mean() > model_a.mean()
                  else "Model A")
        print(f"\n✅ {better} is significantly better!")
        print(f"   Improvement: "
              f"{abs(model_b.mean() - model_a.mean())*100:.1f}%")
    else:
        print("\n❌ No significant difference "
              "between models")
        print("   Either model can be deployed!")


def correlation_significance() -> None:
    """
    Test if correlation is statistically significant.
    Used in feature selection!
    """
    print("\n=== Correlation Significance ===\n")

    np.random.seed(42)
    n = 100

    features = {
        'experience': np.random.randint(0, 12, n),
        'skills': np.random.randint(2, 10, n),
        'rating': np.random.uniform(3, 5, n),
        'random_noise': np.random.randn(n)
    }

    salary = (10 + features['experience'] * 1.8 +
              features['skills'] * 0.5 +
              np.random.normal(0, 2, n))

    print("Feature Correlation Significance Test:")
    print(f"{'Feature':<15} | {'Corr':>6} | "
          f"{'p-value':>8} | {'Decision':>20}")
    print("-" * 60)

    for feat_name, feat_vals in features.items():
        corr, p = stats.pearsonr(feat_vals, salary)
        decision = ("✅ KEEP" if p < 0.05
                   else "❌ DROP")
        print(f"{feat_name:<15} | {corr:>6.3f} | "
              f"{p:>8.4f} | {decision:>20}")

    print("\n💡 random_noise has no significant")
    print("   correlation → safely DROP it!")
    print("   This prevents overfitting! 🔥")


if __name__ == "__main__":
    ttest_demo()
    ab_test_models()
    correlation_significance()
