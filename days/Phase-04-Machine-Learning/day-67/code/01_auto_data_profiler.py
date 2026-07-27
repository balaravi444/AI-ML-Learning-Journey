"""
Day 67 — Autonomous Data Scientist
Topic: Auto Data Profiler
Date: 24 July 2026
Author: Bala Ravi

Given ANY CSV → instant data profile!
Detects types, missing values, outliers,
correlations, distributions automatically.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str
    dtype: str
    inferred_type: str       # numeric / categorical / datetime / id / constant
    n_missing: int
    pct_missing: float
    n_unique: int
    pct_unique: float
    is_constant: bool
    is_id_column: bool

    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    n_outliers: Optional[int] = None

    # Categorical stats
    top_values: Optional[List] = None
    top_counts: Optional[List] = None

    # Quality flags
    has_missing: bool = False
    has_outliers: bool = False
    high_cardinality: bool = False
    high_skew: bool = False

    # Recommendation
    recommendation: str = ""


@dataclass
class DataProfile:
    """Complete profile of a dataset."""
    n_rows: int
    n_cols: int
    n_duplicates: int
    pct_duplicates: float
    memory_mb: float
    column_profiles: Dict[str, ColumnProfile] = (
        field(default_factory=dict))
    target_suggestion: Optional[str] = None
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = (
        field(default_factory=list))


class AutoDataProfiler:
    """
    Automatically profiles any pandas DataFrame.

    Detects column types, missing values,
    outliers, distributions, correlations.
    Zero configuration needed!
    """

    def __init__(self,
                 outlier_threshold: float = 1.5,
                 high_cardinality_threshold: int = 50,
                 high_skew_threshold: float = 1.0,
                 id_uniqueness_threshold: float = 0.95
                 ) -> None:
        """
        Initialize profiler with thresholds.

        Args:
            outlier_threshold: IQR multiplier
            high_cardinality_threshold: Max unique
                for categorical
            high_skew_threshold: Skew threshold
            id_uniqueness_threshold: % unique to
                classify as ID column
        """
        self.outlier_threshold = outlier_threshold
        self.high_card_thresh = (
            high_cardinality_threshold)
        self.high_skew_thresh = high_skew_threshold
        self.id_thresh = id_uniqueness_threshold

    def _infer_type(self,
                    series: pd.Series) -> str:
        """
        Infer semantic type of a column.

        Returns:
            'numeric', 'categorical', 'datetime',
            'id', 'constant', 'text'
        """
        n = len(series)
        n_unique = series.nunique()

        # Constant column
        if n_unique <= 1:
            return 'constant'

        # ID column (almost all unique)
        if (n_unique / n > self.id_thresh and
                n_unique > 100):
            return 'id'

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(
                series):
            return 'datetime'

        # Try parsing as datetime
        if series.dtype == object:
            sample = series.dropna().head(10)
            try:
                pd.to_datetime(sample)
                return 'datetime'
            except Exception:
                pass

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'

        # Text (long strings)
        if series.dtype == object:
            avg_len = series.dropna().str.len().mean()
            if avg_len > 50:
                return 'text'
            return 'categorical'

        return 'categorical'

    def _profile_numeric(
            self,
            series: pd.Series) -> dict:
        """Calculate numeric statistics."""
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.outlier_threshold * iqr
        upper = q3 + self.outlier_threshold * iqr
        n_outliers = int(
            ((clean < lower) | (clean > upper)).sum())

        return {
            'mean': float(clean.mean()),
            'std': float(clean.std()),
            'min_val': float(clean.min()),
            'max_val': float(clean.max()),
            'skewness': float(clean.skew()),
            'kurtosis': float(clean.kurtosis()),
            'n_outliers': n_outliers,
            'has_outliers': n_outliers > 0,
            'high_skew': abs(clean.skew()) > (
                self.high_skew_thresh)
        }

    def _profile_categorical(
            self,
            series: pd.Series,
            top_n: int = 5) -> dict:
        """Calculate categorical statistics."""
        counts = series.value_counts()
        return {
            'top_values': list(
                counts.index[:top_n]),
            'top_counts': list(
                counts.values[:top_n]),
            'high_cardinality': (
                series.nunique() >
                self.high_card_thresh)
        }

    def _make_recommendation(
            self,
            profile: ColumnProfile) -> str:
        """Generate preprocessing recommendation."""
        if profile.inferred_type == 'constant':
            return "DROP — constant column, no info"

        if profile.inferred_type == 'id':
            return "DROP — ID column, no predictive value"

        if profile.pct_missing > 50:
            return (f"DROP — {profile.pct_missing:.0f}% "
                    f"missing, too sparse")

        if profile.inferred_type == 'numeric':
            recs = []
            if profile.has_missing:
                recs.append("Impute with median")
            if profile.high_skew:
                recs.append("Log transform (high skew)")
            if profile.has_outliers:
                recs.append("Consider robust scaling")
            recs.append("StandardScaler")
            return " → ".join(recs)

        if profile.inferred_type == 'categorical':
            if profile.high_cardinality:
                return ("High cardinality → "
                        "Target encode or drop")
            if profile.has_missing:
                return ("Impute mode → "
                        "OneHotEncode")
            return "OneHotEncode"

        if profile.inferred_type == 'datetime':
            return "Extract year/month/day/weekday"

        if profile.inferred_type == 'text':
            return "TF-IDF vectorize or drop"

        return "Review manually"

    def profile_column(
            self,
            series: pd.Series) -> ColumnProfile:
        """Profile a single column."""
        n = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique())

        inferred = self._infer_type(series)

        profile = ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            inferred_type=inferred,
            n_missing=n_missing,
            pct_missing=round(n_missing / n * 100, 2),
            n_unique=n_unique,
            pct_unique=round(n_unique / n * 100, 2),
            is_constant=(inferred == 'constant'),
            is_id_column=(inferred == 'id'),
            has_missing=(n_missing > 0))

        if inferred == 'numeric':
            stats = self._profile_numeric(series)
            for k, v in stats.items():
                if hasattr(profile, k):
                    setattr(profile, k, v)

        elif inferred == 'categorical':
            stats = self._profile_categorical(series)
            for k, v in stats.items():
                if hasattr(profile, k):
                    setattr(profile, k, v)

        profile.recommendation = (
            self._make_recommendation(profile))

        return profile

    def _suggest_target(
            self,
            df: pd.DataFrame) -> Optional[str]:
        """Suggest most likely target column."""
        candidates = []

        target_hints = [
            'target', 'label', 'output',
            'result', 'class', 'category',
            'price', 'salary', 'revenue',
            'churn', 'fraud', 'default',
            'survival', 'outcome', 'y']

        for col in df.columns:
            col_lower = col.lower()
            for hint in target_hints:
                if hint in col_lower:
                    candidates.append(col)
                    break

        if candidates:
            return candidates[0]

        # Last numeric column as fallback
        num_cols = df.select_dtypes(
            include='number').columns
        if len(num_cols) > 0:
            return num_cols[-1]

        return None

    def _calculate_quality_score(
            self,
            profile: DataProfile) -> float:
        """
        Calculate overall data quality score 0-100.
        """
        score = 100.0

        # Penalize missing values
        avg_missing = np.mean([
            p.pct_missing
            for p in profile.column_profiles.values()])
        score -= avg_missing * 0.5

        # Penalize duplicates
        score -= profile.pct_duplicates * 0.3

        # Penalize constant/id columns
        n_useless = sum(
            1 for p in
            profile.column_profiles.values()
            if p.is_constant or p.is_id_column)
        score -= n_useless * 2

        # Penalize high outlier columns
        n_outlier_cols = sum(
            1 for p in
            profile.column_profiles.values()
            if p.has_outliers)
        score -= n_outlier_cols * 1

        return max(0.0, min(100.0, round(score, 1)))

    def profile(self,
                df: pd.DataFrame) -> DataProfile:
        """
        Profile entire DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Complete DataProfile object
        """
        n_duplicates = int(df.duplicated().sum())
        memory_mb = round(
            df.memory_usage(deep=True).sum() /
            1024 / 1024, 2)

        data_profile = DataProfile(
            n_rows=len(df),
            n_cols=len(df.columns),
            n_duplicates=n_duplicates,
            pct_duplicates=round(
                n_duplicates / len(df) * 100, 2),
            memory_mb=memory_mb)

        for col in df.columns:
            col_profile = self.profile_column(df[col])
            data_profile.column_profiles[col] = (
                col_profile)

        data_profile.target_suggestion = (
            self._suggest_target(df))

        data_profile.quality_score = (
            self._calculate_quality_score(data_profile))

        # Collect issues
        issues = []
        if data_profile.pct_duplicates > 5:
            issues.append(
                f"⚠️  {data_profile.pct_duplicates:.1f}% "
                f"duplicate rows detected")

        for col, prof in (
                data_profile.column_profiles.items()):
            if prof.pct_missing > 30:
                issues.append(
                    f"⚠️  '{col}': "
                    f"{prof.pct_missing:.0f}% missing")
            if prof.is_constant:
                issues.append(
                    f"⚠️  '{col}': constant — "
                    f"no predictive value")
            if prof.high_skew:
                issues.append(
                    f"📊 '{col}': high skew "
                    f"({prof.skewness:.2f}) — "
                    f"consider log transform")

        data_profile.issues = issues

        return data_profile

    def print_report(
            self,
            profile: DataProfile) -> None:
        """Print formatted profile report."""
        print("=" * 60)
        print("  AUTO DATA PROFILE REPORT")
        print("=" * 60)

        print(f"\n📊 Dataset Overview:")
        print(f"  Rows:        {profile.n_rows:,}")
        print(f"  Columns:     {profile.n_cols}")
        print(f"  Duplicates:  {profile.n_duplicates} "
              f"({profile.pct_duplicates:.1f}%)")
        print(f"  Memory:      {profile.memory_mb} MB")
        print(f"  Quality:     "
              f"{profile.quality_score:.0f}/100")

        if profile.target_suggestion:
            print(f"\n🎯 Suggested Target: "
                  f"'{profile.target_suggestion}'")

        print(f"\n📋 Column Analysis:")
        print(f"  {'Column':<20} | "
              f"{'Type':<12} | "
              f"{'Missing':>8} | "
              f"{'Unique':>8} | "
              f"{'Recommendation'}")
        print(f"  {'-'*80}")

        for col, prof in (
                profile.column_profiles.items()):
            flag = ""
            if prof.is_constant:
                flag = " 🚫"
            elif prof.is_id_column:
                flag = " 🔑"
            elif prof.pct_missing > 20:
                flag = " ⚠️"

            print(f"  {col[:18]:<20} | "
                  f"{prof.inferred_type:<12} | "
                  f"{prof.pct_missing:>7.1f}% | "
                  f"{prof.n_unique:>8} | "
                  f"{prof.recommendation[:35]}"
                  f"{flag}")

        if profile.issues:
            print(f"\n⚠️  Issues Detected:")
            for issue in profile.issues:
                print(f"  {issue}")

        print(f"\n{'='*60}")


def create_sample_dataset() -> pd.DataFrame:
    """Create messy dataset for profiling demo."""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'employee_id': range(1, n + 1),
        'age': np.random.normal(35, 8, n).astype(int),
        'salary': np.random.lognormal(11, 0.5, n),
        'department': np.random.choice(
            ['Engineering', 'Sales', 'HR',
             'Marketing', 'Finance'], n),
        'years_exp': np.random.exponential(4, n),
        'performance_score': np.random.uniform(1, 5, n),
        'city': np.random.choice(
            ['Bangalore', 'Mumbai', 'Delhi',
             'Hyderabad', 'Pune',
             'Chennai', 'Kolkata',
             'Ahmedabad', 'Jaipur', 'Surat',
             'Lucknow', 'Kanpur', 'Nagpur',
             'Indore', 'Thane',
             'Bhopal', 'Visakhapatnam',
             'Patna', 'Vadodara', 'Ghaziabad'], n),
        'constant_col': 1,
        'churn': np.random.choice([0, 1], n,
                                   p=[0.85, 0.15])
    })

    # Add missing values
    df.loc[
        np.random.choice(n, 50, replace=False),
        'age'] = np.nan
    df.loc[
        np.random.choice(n, 80, replace=False),
        'years_exp'] = np.nan
    df.loc[
        np.random.choice(n, 30, replace=False),
        'department'] = np.nan

    # Add duplicates
    dupes = df.sample(20)
    df = pd.concat([df, dupes]).reset_index(
        drop=True)

    return df


if __name__ == "__main__":
    print("Creating sample dataset...\n")
    df = create_sample_dataset()

    profiler = AutoDataProfiler()
    profile = profiler.profile(df)
    profiler.print_report(profile)

    print(f"\n✅ Auto profiling complete!")
    print(f"   No human analysis needed!")
    print(f"   Next: auto build preprocessing pipeline!")
