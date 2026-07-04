"""
Day 46 — Data Preprocessing Pipeline
Topic: Custom Transformers
Date: 03 July 2026
Author: Bala Ravi

Custom transformers = plug any transformation
into a sklearn Pipeline!

Used for domain-specific feature engineering!
"""
import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator, TransformerMixin)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score)
from sklearn.metrics import r2_score


class LogTransformer(BaseEstimator,
                      TransformerMixin):
    """
    Log transform — fixes right-skewed distributions!
    Uses log1p to handle zeros safely.
    """

    def __init__(self, columns: list = None):
        self.columns = columns

    def fit(self, X, y=None):
        """Nothing to learn — stateless transform."""
        return self

    def transform(self, X):
        """Apply log1p transform."""
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            if self.columns:
                X[self.columns] = np.log1p(
                    X[self.columns])
            else:
                X = pd.DataFrame(
                    np.log1p(X.values),
                    columns=X.columns,
                    index=X.index)
        else:
            X = np.log1p(X)
        return X


class OutlierCapper(BaseEstimator,
                    TransformerMixin):
    """
    Cap outliers using IQR method.
    Learned from training data only!
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        """Learn IQR bounds from training data."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        self.Q1_ = np.percentile(
            X_arr, 25, axis=0)
        self.Q3_ = np.percentile(
            X_arr, 75, axis=0)
        self.IQR_ = self.Q3_ - self.Q1_
        self.lower_ = (self.Q1_ -
                       self.factor * self.IQR_)
        self.upper_ = (self.Q3_ +
                       self.factor * self.IQR_)
        return self

    def transform(self, X):
        """Apply learned bounds to cap outliers."""
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.copy()
            result = np.clip(
                X_arr, self.lower_, self.upper_)
            return pd.DataFrame(
                result,
                columns=X.columns,
                index=X.index)
        return np.clip(X, self.lower_, self.upper_)


class FinancialFeatureEngineer(
        BaseEstimator, TransformerMixin):
    """
    Custom feature engineering for job market data.
    Creates interaction and ratio features!
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Create new features from existing ones."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = pd.DataFrame(X)

        if ('experience_years' in X.columns and
                'skills_count' in X.columns):
            X['exp_x_skills'] = (
                X['experience_years'] *
                X['skills_count'])
            X['skills_per_exp'] = (
                X['skills_count'] /
                (X['experience_years'] + 1))

        if 'experience_years' in X.columns:
            X['exp_squared'] = (
                X['experience_years'] ** 2)

        return X


def demo_custom_transformers() -> None:
    """Demonstrate all custom transformers."""
    print("=== Custom Transformers Demo ===\n")

    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'experience_years': np.random.randint(0, 12, n)
                                       .astype(float),
        'skills_count': np.random.randint(2, 10, n)
                                  .astype(float),
        'salary_lpa': np.clip(
            np.random.normal(20, 8, n), 5, 55)
    })

    # Add outliers
    df.loc[0, 'salary_lpa'] = 150
    df.loc[1, 'salary_lpa'] = 200

    print("Before transformers:")
    print(f"  salary skew: {df['salary_lpa'].skew():.3f}")
    print(f"  salary max:  ₹{df['salary_lpa'].max():.1f}L")

    # Apply OutlierCapper
    capper = OutlierCapper(factor=1.5)
    df_capped = capper.fit_transform(df)
    print(f"\nAfter OutlierCapper:")
    print(f"  salary max: ₹{df_capped['salary_lpa'].max():.1f}L")

    # Apply LogTransformer
    log_t = LogTransformer(
        columns=['salary_lpa'])
    df_log = log_t.fit_transform(df_capped)
    print(f"\nAfter LogTransformer:")
    print(f"  salary_lpa skew: "
          f"{df_log['salary_lpa'].skew():.3f}")

    # Apply FinancialFeatureEngineer
    feat_eng = FinancialFeatureEngineer()
    df_feat = feat_eng.fit_transform(df)
    print(f"\nAfter FinancialFeatureEngineer:")
    print(f"  New features: "
          f"{[c for c in df_feat.columns if c not in df.columns]}")


def build_custom_pipeline() -> None:
    """Build pipeline with all custom transformers."""
    print("\n=== Custom Pipeline ===\n")

    np.random.seed(42)
    n = 400

    cities = ['Bangalore', 'Mumbai', 'Delhi',
              'Hyderabad', 'Pune']
    roles = ['Data Scientist', 'ML Engineer',
             'Data Analyst', 'AI Engineer']

    experience = np.random.randint(0, 12, n).astype(float)
    city = np.random.choice(cities, n)
    role = np.random.choice(roles, n)

    city_b = {'Bangalore': 3, 'Mumbai': 2,
               'Delhi': 1, 'Hyderabad': 0.5, 'Pune': 0}
    role_b = {'Data Scientist': 3,
               'ML Engineer': 4,
               'Data Analyst': -2, 'AI Engineer': 5}

    salary = np.clip(
        10 + experience * 1.8 +
        np.array([city_b[c] for c in city]) +
        np.array([role_b[r] for r in role]) +
        np.random.normal(0, 2, n), 5, 55).round(1)

    df = pd.DataFrame({
        'experience_years': experience,
        'skills_count': np.random.randint(2, 10, n)
                                  .astype(float),
        'salary_lpa': salary
    })

    X = df[['experience_years', 'skills_count']]
    y = df['salary_lpa']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42))

    # Pipeline with custom transformers!
    pipeline = Pipeline([
        ('outlier_capper', OutlierCapper()),
        ('feature_engineer',
         FinancialFeatureEngineer()),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(
        pipeline, X, y, cv=5, scoring='r2')

    print(f"Pipeline steps:")
    for name, step in pipeline.steps:
        print(f"  {name}: {type(step).__name__}")

    print(f"\nTest R²: {r2:.4f}")
    print(f"CV R²:   {cv_scores.mean():.4f} "
          f"± {cv_scores.std():.4f}")
    print(f"\n✅ Custom transformers work in pipeline!")


if __name__ == "__main__":
    demo_custom_transformers()
    build_custom_pipeline()
