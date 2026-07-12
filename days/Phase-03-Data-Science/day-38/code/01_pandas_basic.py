"""
Day 38 — Pandas: DataFrames
Topic: Series and DataFrame Basics
Date: 25 June 2026
Author: Bala Ravi

Pandas = Python Data Analysis Library
Built on top of NumPy — adds labels to arrays!

Real World Connection:
    Every Kaggle competition uses Pandas!
    Every data scientist uses Pandas daily!
    Foundation of all data science work!
"""
import pandas as pd
import numpy as np


def series_demo() -> None:
    """Demonstrate Pandas Series — 1D labeled data."""
    print("=== Pandas Series ===\n")

    # Create Series
    scores = pd.Series(
        [85, 92, 78, 95, 88, 73, 96],
        index=['Bala', 'Ravi', 'Kumar',
               'Priya', 'Hari', 'Arjun', 'Meera']
    )

    print(f"Student Scores:\n{scores}")
    print(f"\nTop scorer: {scores.idxmax()} "
          f"({scores.max()})")
    print(f"Average: {scores.mean():.1f}")
    print(f"Passed (>=80): {(scores >= 80).sum()}")

    # Series operations
    print(f"\nStudents who passed:")
    print(scores[scores >= 80])

    # Arithmetic
    curved = scores + 5
    print(f"\nAfter 5-point curve:")
    print(curved)


def dataframe_creation_demo() -> None:
    """Show multiple ways to create DataFrames."""
    print("\n=== DataFrame Creation ===\n")

    # From dictionary
    data = {
        'name': ['Bala', 'Ravi', 'Kumar',
                 'Priya', 'Hari'],
        'age': [22, 23, 21, 22, 24],
        'marks': [85, 92, 78, 95, 88],
        'course': ['AI', 'ML', 'DS', 'AI', 'ML'],
        'city': ['Bangalore', 'Chennai',
                 'Mumbai', 'Delhi', 'Hyderabad']
    }
    df = pd.DataFrame(data)

    print("Student DataFrame:")
    print(df)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")

    # From NumPy array
    np.random.seed(42)
    arr = np.random.randint(50, 100, (4, 3))
    df_from_numpy = pd.DataFrame(
        arr,
        columns=['Math', 'Science', 'English'],
        index=['Student1', 'Student2',
               'Student3', 'Student4']
    )
    print(f"\nFrom NumPy array:\n{df_from_numpy}")

    return df


def dataframe_selection_demo(df: pd.DataFrame) -> None:
    """Show data selection methods."""
    print("\n=== Data Selection ===\n")

    # Column selection
    print("Single column (Series):")
    print(df['marks'])

    print("\nMultiple columns (DataFrame):")
    print(df[['name', 'marks', 'course']])

    # Row selection with iloc
    print("\nFirst row (iloc[0]):")
    print(df.iloc[0])

    print("\nFirst 3 rows (iloc[0:3]):")
    print(df.iloc[0:3])

    # Row selection with loc
    print("\nRow with index 2 (loc[2]):")
    print(df.loc[2])

    # Boolean filtering
    print("\nStudents scoring > 85:")
    print(df[df['marks'] > 85])

    print("\nAI course students:")
    print(df[df['course'] == 'AI'])

    print("\nYoung high-scorers (age<23 AND marks>80):")
    print(df[(df['age'] < 23) & (df['marks'] > 80)])


def basic_statistics_demo(df: pd.DataFrame) -> None:
    """Show basic statistical operations."""
    print("\n=== Basic Statistics ===\n")

    print("describe():")
    print(df.describe())

    print(f"\nMean marks: {df['marks'].mean():.2f}")
    print(f"Std marks:  {df['marks'].std():.2f}")
    print(f"Max marks:  {df['marks'].max()}")
    print(f"Min marks:  {df['marks'].min()}")

    print(f"\nValue counts for course:")
    print(df['course'].value_counts())

    print(f"\nCorrelation (age vs marks):")
    print(f"{df['age'].corr(df['marks']):.3f}")


if __name__ == "__main__":
    series_demo()
    df = dataframe_creation_demo()
    dataframe_selection_demo(df)
    basic_statistics_demo(df)
