"""
Day 39 — Pandas: Data Cleaning
Topic: Production Data Cleaning Pipeline
Date: 26 June 2026
Author: Bala Ravi

This is a reusable cleaning pipeline
for the Indian Job Market Analyzer project!
"""
import pandas as pd
import numpy as np
from typing import Optional


class DataCleaner:
    """
    Reusable data cleaning pipeline.
    Used in Indian Job Market Analyzer (Day 47+)!
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize cleaner with DataFrame.

        Args:
            df: Raw DataFrame to clean
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []

    def log(self, message: str) -> None:
        """Log a cleaning step."""
        self.cleaning_log.append(message)
        print(f"  ✅ {message}")

    def remove_duplicates(
            self,
            subset: Optional[list] = None) -> 'DataCleaner':
        """Remove duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates(
            subset=subset).reset_index(drop=True)
        removed = before - len(self.df)
        self.log(f"Removed {removed} duplicates")
        return self

    def fix_missing_numerical(
            self,
            columns: list,
            strategy: str = 'median') -> 'DataCleaner':
        """
        Fill missing numerical values.

        Args:
            columns: Column names to fix
            strategy: 'mean', 'median', or 'zero'
        """
        for col in columns:
            if col in self.df.columns:
                missing = self.df[col].isnull().sum()
                if missing > 0:
                    if strategy == 'median':
                        fill = self.df[col].median()
                    elif strategy == 'mean':
                        fill = self.df[col].mean()
                    else:
                        fill = 0
                    self.df[col].fillna(
                        fill, inplace=True)
                    self.log(
                        f"Filled {missing} missing "
                        f"in '{col}' with {strategy} "
                        f"({fill:.1f})")
        return self

    def fix_missing_categorical(
            self,
            columns: list,
            fill_value: str = 'Unknown') -> 'DataCleaner':
        """Fill missing categorical values."""
        for col in columns:
            if col in self.df.columns:
                missing = self.df[col].isnull().sum()
                if missing > 0:
                    self.df[col].fillna(
                        fill_value, inplace=True)
                    self.log(
                        f"Filled {missing} missing "
                        f"in '{col}' with '{fill_value}'")
        return self

    def clean_strings(
            self, columns: list) -> 'DataCleaner':
        """Strip whitespace and title case strings."""
        for col in columns:
            if col in self.df.columns:
                self.df[col] = (self.df[col]
                                .str.strip()
                                .str.title())
                self.log(f"Cleaned strings in '{col}'")
        return self

    def remove_outliers(
            self,
            column: str,
            method: str = 'iqr') -> 'DataCleaner':
        """Remove outliers using IQR method."""
        before = len(self.df)
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        self.df = self.df[
            (self.df[column] >= lower) &
            (self.df[column] <= upper)
        ].reset_index(drop=True)
        removed = before - len(self.df)
        self.log(
            f"Removed {removed} outliers from '{column}' "
            f"(range: {lower:.0f}-{upper:.0f})")
        return self

    def fix_dtypes(
            self, type_map: dict) -> 'DataCleaner':
        """
        Fix column data types.

        Args:
            type_map: {column: dtype} mapping
        """
        for col, dtype in type_map.items():
            if col in self.df.columns:
                try:
                    if dtype == 'numeric':
                        self.df[col] = pd.to_numeric(
                            self.df[col], errors='coerce')
                    elif dtype == 'datetime':
                        self.df[col] = pd.to_datetime(
                            self.df[col], errors='coerce')
                    else:
                        self.df[col] = self.df[col].astype(
                            dtype)
                    self.log(
                        f"Fixed dtype for '{col}' → {dtype}")
                except Exception as e:
                    print(f"  ⚠️ Could not fix '{col}': {e}")
        return self

    def get_clean_data(self) -> pd.DataFrame:
        """Return cleaned DataFrame."""
        return self.df

    def summary(self) -> None:
        """Print cleaning summary."""
        print("\n=== Cleaning Summary ===")
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape:    {self.df.shape}")
        print(f"Rows removed:   "
              f"{self.original_shape[0] - len(self.df)}")
        print(f"Missing remaining: "
              f"{self.df.isnull().sum().sum()}")
        print(f"\nSteps completed: "
              f"{len(self.cleaning_log)}")
        for step in self.cleaning_log:
            print(f"  • {step}")


if __name__ == "__main__":
    print("=== Production Data Cleaning Pipeline ===\n")

    np.random.seed(42)
    n = 30

    raw_data = pd.DataFrame({
        'job_title': np.random.choice(
            ['Data Scientist', 'ML Engineer',
             'Data Analyst', None], n),
        'company': np.random.choice(
            ['TCS', 'Infosys', '  wipro  ',
             'AMAZON', None], n),
        'location': np.random.choice(
            ['bangalore', 'MUMBAI', 'delhi',
             'Hyderabad', None], n),
        'salary_lpa': (list(np.random.normal(20, 8, n-3)) +
                       [200, -10, None]),
        'experience': ([None, 3, 2, 1] * 7 + [2]),
        'salary_str': ['₹' + str(int(s)) + ' LPA'
                       if not np.isnan(s) else None
                       for s in
                       (list(np.random.normal(
                           20, 8, n-3)) + [200, -10, 0])]
    })

    # Add duplicates
    raw_data = pd.concat([
        raw_data,
        raw_data.iloc[:3]
    ], ignore_index=True)

    print(f"Raw data shape: {raw_data.shape}")
    print(f"Missing values: "
          f"{raw_data.isnull().sum().sum()}")

    print("\nRunning cleaning pipeline:")
    cleaner = DataCleaner(raw_data)

    df_clean = (cleaner
                .remove_duplicates()
                .fix_dtypes({'salary_lpa': 'numeric'})
                .fix_missing_numerical(
                    ['salary_lpa', 'experience'])
                .fix_missing_categorical(
                    ['job_title', 'company', 'location'])
                .clean_strings(
                    ['company', 'location'])
                .remove_outliers('salary_lpa')
                .get_clean_data())

    cleaner.summary()

    print(f"\nClean data sample:")
    print(df_clean.head())
    print("\n✅ Data ready for analysis and ML!")
