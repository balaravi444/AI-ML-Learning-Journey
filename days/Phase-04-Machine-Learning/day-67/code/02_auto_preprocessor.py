"""
Day 67 — Autonomous Data Scientist
Topic: Auto Preprocessing Pipeline Builder
Date: 24 July 2026
Author: Bala Ravi

Given a DataProfile → automatically build
the optimal sklearn preprocessing pipeline!
No manual feature engineering needed.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder,
    FunctionTransformer, LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class LogTransformer(BaseEstimator, TransformerMixin):
    """Log1p transformer for skewed features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(np.abs(X))

    def inverse_transform(self, X):
        return np.expm1(X)


class DatetimeFeatureExtractor(
        BaseEstimator, TransformerMixin):
    """Extract features from datetime columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, 'iloc'):
            X = X.iloc[:, 0]
        X = pd.to_datetime(X, errors='coerce')
        features = np.column_stack([
            X.dt.year.fillna(0).values,
            X.dt.month.fillna(0).values,
            X.dt.day.fillna(0).values,
            X.dt.dayofweek.fillna(0).values,
            X.dt.quarter.fillna(0).values
        ])
        return features


class AutoPreprocessor:
    """
    Automatically builds sklearn preprocessing pipeline
    from a DataProfile.

    Applies ML best practices automatically:
    - Median imputation for numeric
    - Mode imputation for categorical
    - Log transform for skewed features
    - StandardScaler for numeric
    - OneHotEncoder for categorical
    - Drop useless columns (constant, ID)
    """

    def __init__(self) -> None:
        """Initialize auto preprocessor."""
        self.pipeline = None
        self.feature_names_out = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        self.drop_cols = []
        self.skewed_cols = []
        self.target_col = None
        self.label_encoder = None
        self.task_type = None  # classification/regression

    def _detect_task_type(
            self,
            y: pd.Series) -> str:
        """Detect if task is classification or regression."""
        n_unique = y.nunique()
        if n_unique <= 20 or y.dtype == object:
            return 'classification'
        return 'regression'

    def analyze_columns(
            self,
            df: pd.DataFrame,
            target_col: str) -> None:
        """
        Analyze columns and categorize them.

        Args:
            df: Input DataFrame
            target_col: Target column name
        """
        self.target_col = target_col

        for col in df.columns:
            if col == target_col:
                continue

            series = df[col]
            n = len(series)
            n_unique = series.nunique()

            # Drop useless
            if n_unique <= 1:
                self.drop_cols.append(col)
                continue

            if (n_unique / n > 0.95 and
                    n_unique > 100):
                self.drop_cols.append(col)
                continue

            if series.isna().mean() > 0.5:
                self.drop_cols.append(col)
                continue

            # Detect type
            if pd.api.types.is_datetime64_any_dtype(
                    series):
                self.datetime_cols.append(col)
                continue

            if pd.api.types.is_numeric_dtype(series):
                self.numeric_cols.append(col)
                # Check skewness
                clean = series.dropna()
                if (len(clean) > 0 and
                        abs(clean.skew()) > 1.0):
                    self.skewed_cols.append(col)
                continue

            if series.dtype == object:
                if n_unique <= 50:
                    self.categorical_cols.append(col)
                else:
                    self.drop_cols.append(col)
                continue

    def build_pipeline(
            self,
            df: pd.DataFrame,
            target_col: str) -> Pipeline:
        """
        Build complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            target_col: Target column name

        Returns:
            Fitted sklearn Pipeline
        """
        self.analyze_columns(df, target_col)

        print(f"\nColumn Analysis:")
        print(f"  Numeric:     {self.numeric_cols}")
        print(f"  Categorical: {self.categorical_cols}")
        print(f"  Datetime:    {self.datetime_cols}")
        print(f"  Dropped:     {self.drop_cols}")
        print(f"  Skewed:      {self.skewed_cols}")

        transformers = []

        # Numeric pipeline
        if self.numeric_cols:
            # Split skewed vs normal
            normal_num = [
                c for c in self.numeric_cols
                if c not in self.skewed_cols]
            skewed_num = self.skewed_cols

            if normal_num:
                num_pipe = Pipeline([
                    ('imputer', SimpleImputer(
                        strategy='median')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(
                    ('numeric', num_pipe, normal_num))

            if skewed_num:
                skew_pipe = Pipeline([
                    ('imputer', SimpleImputer(
                        strategy='median')),
                    ('log', LogTransformer()),
                    ('scaler', StandardScaler())
                ])
                transformers.append(
                    ('skewed', skew_pipe, skewed_num))

        # Categorical pipeline
        if self.categorical_cols:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(
                    strategy='most_frequent')),
                ('encoder', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    drop='first'))
            ])
            transformers.append(
                ('categorical', cat_pipe,
                 self.categorical_cols))

        # Datetime pipeline
        if self.datetime_cols:
            for dt_col in self.datetime_cols:
                dt_pipe = Pipeline([
                    ('extractor',
                     DatetimeFeatureExtractor())
                ])
                transformers.append(
                    (f'datetime_{dt_col}',
                     dt_pipe, [dt_col]))

        if not transformers:
            raise ValueError(
                "No valid columns found for preprocessing!")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop')

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])

        # Prepare target
        X = df.drop(columns=[target_col] +
                    self.drop_cols)
        y = df[target_col]

        self.task_type = self._detect_task_type(y)
        print(f"\n  Task type: {self.task_type}")

        if self.task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)

        # Fit pipeline
        self.pipeline.fit(X)
        X_transformed = self.pipeline.transform(X)

        print(f"\n  Input shape:  {X.shape}")
        print(f"  Output shape: {X_transformed.shape}")
        print(f"  Features created: "
              f"{X_transformed.shape[1]}")

        return self.pipeline, y

    def transform(
            self,
            df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError(
                "Call build_pipeline first!")

        # Drop same columns as training
        cols_to_drop = [
            c for c in self.drop_cols
            if c in df.columns]
        if self.target_col in df.columns:
            cols_to_drop.append(self.target_col)

        X = df.drop(
            columns=cols_to_drop,
            errors='ignore')
        return self.pipeline.transform(X)

    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True)
        joblib.dump(self, path)
        print(f"✅ Preprocessor saved: {path}")

    @classmethod
    def load(cls, path: str) -> 'AutoPreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(path)


def demonstrate_auto_preprocessing() -> None:
    """Show auto preprocessing on sample data."""
    print("=== Auto Preprocessing Pipeline ===\n")

    # Create messy dataset
    np.random.seed(42)
    n = 300

    df = pd.DataFrame({
        'employee_id': range(1, n + 1),
        'age': np.random.normal(35, 8, n),
        'salary': np.random.lognormal(11, 0.5, n),
        'years_exp': np.random.exponential(4, n),
        'performance': np.random.uniform(1, 5, n),
        'department': np.random.choice(
            ['Engineering', 'Sales', 'HR',
             'Marketing', 'Finance'], n),
        'city': np.random.choice(
            ['Bangalore', 'Mumbai', 'Delhi'], n),
        'constant_col': 1,
        'churn': np.random.choice([0, 1], n)
    })

    # Add missing values
    df.loc[
        np.random.choice(n, 30, replace=False),
        'age'] = np.nan
    df.loc[
        np.random.choice(n, 40, replace=False),
        'years_exp'] = np.nan
    df.loc[
        np.random.choice(n, 20, replace=False),
        'department'] = np.nan

    print(f"Input dataset: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n"
          f"{df.isnull().sum()[df.isnull().sum() > 0]}")

    preprocessor = AutoPreprocessor()
    pipeline, y = preprocessor.build_pipeline(
        df, target_col='churn')

    print(f"\n✅ Pipeline built automatically!")
    print(f"   No manual feature engineering!")
    print(f"   Task: {preprocessor.task_type}")
    print(f"\nPreprocessing steps applied:")
    print(f"  ✅ Dropped: {preprocessor.drop_cols}")
    print(f"  ✅ Log transformed: "
          f"{preprocessor.skewed_cols}")
    print(f"  ✅ StandardScaled: "
          f"{preprocessor.numeric_cols}")
    print(f"  ✅ OneHotEncoded: "
          f"{preprocessor.categorical_cols}")

    # Transform new data
    new_data = df.head(5)
    transformed = preprocessor.transform(new_data)
    print(f"\nNew data transformed: "
          f"{new_data.shape} → {transformed.shape}")


if __name__ == "__main__":
    demonstrate_auto_preprocessing()
