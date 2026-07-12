"""
Day 44 — Feature Engineering
Topic: Creating New Features
Date: 01 July 2026
Author: Bala Ravi

Raw features → engineered features → better model!
Interaction features, ratios, log transforms,
binning — the tools of feature engineering!
"""
import pandas as pd
import numpy as np


def create_dataset(n: int = 300) -> pd.DataFrame:
    """Create job market dataset."""
    np.random.seed(42)
    experience = np.random.randint(0, 12, n)
    salary = np.clip(
        10 + experience * 1.8 +
        np.random.normal(0, 3, n), 5, 55).round(1)

    return pd.DataFrame({
        'salary_lpa': salary,
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n),
        'rating': np.round(
            np.random.uniform(3.0, 5.0, n), 1),
        'posted_days_ago': np.random.randint(1, 30, n)
    })


def create_new_features(
        df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all types of engineered features.
    Each type serves a different purpose!
    """
    print("=== Feature Creation ===\n")
    df = df.copy()

    original_cols = set(df.columns)

    # 1. Log transform — fix skewness!
    df['salary_log'] = np.log1p(df['salary_lpa'])
    print("1. Log Transform (fix skewness):")
    print(f"   salary_lpa skew:  "
          f"{df['salary_lpa'].skew():.3f}")
    print(f"   salary_log skew:  "
          f"{df['salary_log'].skew():.3f}")
    print(f"   → Closer to 0 = better for "
          f"linear models!\n")

    # 2. Interaction features
    df['exp_x_skills'] = (df['experience_years'] *
                          df['skills_count'])
    df['exp_x_rating'] = (df['experience_years'] *
                          df['rating'])
    print("2. Interaction Features:")
    print(f"   exp_x_skills corr with salary: "
          f"{df['exp_x_skills'].corr(df['salary_lpa']):.3f}")
    print(f"   experience corr with salary:   "
          f"{df['experience_years'].corr(df['salary_lpa']):.3f}")
    print(f"   → Interaction can be stronger!\n")

    # 3. Ratio features
    df['salary_per_exp'] = (
        df['salary_lpa'] /
        (df['experience_years'] + 1))
    df['skills_per_exp'] = (
        df['skills_count'] /
        (df['experience_years'] + 1))
    print("3. Ratio Features:")
    print(f"   salary_per_exp mean: "
          f"₹{df['salary_per_exp'].mean():.1f}L per yr")
    print(f"   → Useful for spotting "
          f"high/low value hires!\n")

    # 4. Binning — group continuous values
    df['exp_level'] = pd.cut(
        df['experience_years'],
        bins=[0, 2, 5, 8, 12],
        labels=['Junior', 'Mid', 'Senior', 'Expert'],
        include_lowest=True)

    df['salary_band'] = pd.cut(
        df['salary_lpa'],
        bins=[0, 15, 25, 40, 100],
        labels=['Entry', 'Mid', 'Senior', 'Top'])

    print("4. Binning:")
    print("   Experience levels:")
    print(f"   {df['exp_level'].value_counts().to_dict()}")
    print("   Salary bands:")
    print(f"   {df['salary_band'].value_counts().to_dict()}\n")

    # 5. Recency feature
    df['posting_recency'] = np.where(
        df['posted_days_ago'] <= 7,
        'Recent', np.where(
            df['posted_days_ago'] <= 14,
            'Week Old', 'Old'))
    print("5. Recency Feature from date:")
    print(f"   {df['posting_recency'].value_counts().to_dict()}\n")

    new_cols = set(df.columns) - original_cols
    print(f"✅ Created {len(new_cols)} new features!")
    print(f"   {sorted(new_cols)}")

    return df


def polynomial_features_demo(
        df: pd.DataFrame) -> None:
    """
    Polynomial features — capture non-linear patterns!
    """
    print("\n=== Polynomial Features ===\n")

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X = df[['experience_years']].values
    y = df['salary_lpa'].values

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    results = {}
    for degree in [1, 2, 3]:
        poly = PolynomialFeatures(
            degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
        r2 = r2_score(y_test, y_pred)
        results[degree] = r2

    print("R² Score by Polynomial Degree:")
    for degree, r2 in results.items():
        bar = '█' * int(r2 * 30)
        print(f"  degree={degree}: {r2:.4f} {bar}")

    best = max(results, key=results.get)
    print(f"\nBest degree: {best}")
    print(f"Higher degree = captures non-linear "
          f"salary patterns!")


if __name__ == "__main__":
    df = create_dataset()
    print(f"Dataset: {df.shape}\n")

    df_engineered = create_new_features(df)
    polynomial_features_demo(df)
