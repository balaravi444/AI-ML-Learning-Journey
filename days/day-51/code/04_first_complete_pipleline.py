"""
Day 51 — ML Fundamentals + Scikit-learn
Topic: First Complete ML Pipeline
Date: 08 July 2026
Author: Bala Ravi

Everything together — preview of what Phase 4
will build in detail!
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV)
from sklearn.metrics import (
    r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')


def create_student_dataset(
        n: int = 1000) -> pd.DataFrame:
    """
    Create student performance dataset.
    Preview of Day 59 project!
    """
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai',
              'Delhi', 'Hyderabad', 'Pune']
    schools = ['Government', 'Private', 'CBSE']

    study_hours = np.random.uniform(1, 10, n)
    attendance = np.random.uniform(40, 100, n)
    prev_score = np.random.uniform(30, 95, n)
    assignments = np.random.randint(0, 10, n)
    sleep_hours = np.random.uniform(4, 9, n)
    city = np.random.choice(cities, n)
    school = np.random.choice(schools, n)

    # City effect
    city_effect = {'Bangalore': 3,
                   'Mumbai': 2,
                   'Delhi': 1,
                   'Hyderabad': 0,
                   'Pune': -1}
    school_effect = {'Government': -2,
                     'Private': 3,
                     'CBSE': 1}

    score = (
        study_hours * 4.5 +
        attendance * 0.25 +
        prev_score * 0.35 +
        assignments * 1.0 +
        sleep_hours * 1.2 +
        np.array([city_effect[c] for c in city]) +
        np.array([school_effect[s]
                  for s in school]) +
        np.random.normal(0, 5, n))
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        'study_hours': study_hours.round(1),
        'attendance_pct': attendance.round(1),
        'prev_score': prev_score.round(1),
        'assignments_done': assignments,
        'sleep_hours': sleep_hours.round(1),
        'city': city,
        'school_type': school,
        'final_score': score.round(1)
    })

    # Add missing values (real world!)
    missing_idx = np.random.choice(
        n, int(n * 0.05), replace=False)
    df.loc[missing_idx[:25],
           'study_hours'] = np.nan
    df.loc[missing_idx[25:],
           'attendance_pct'] = np.nan

    return df


def build_and_evaluate_pipeline(
        df: pd.DataFrame) -> None:
    """
    Build and evaluate complete ML pipeline.
    """
    print("=== Complete ML Pipeline ===\n")

    num_features = [
        'study_hours', 'attendance_pct',
        'prev_score', 'assignments_done',
        'sleep_hours']
    cat_features = ['city', 'school_type']

    X = df[num_features + cat_features]
    y = df['final_score']

    print(f"Features: {len(num_features)} numerical "
          f"+ {len(cat_features)} categorical")
    print(f"Missing values: {X.isnull().sum().sum()}")

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42))

    # Build pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(
            strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(
            strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Compare models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42)
    }

    print(f"\n{'Model':<22} | {'R²':>8} | "
          f"{'MAE':>8} | {'CV R²':>8}")
    print("-" * 55)

    best_model = None
    best_score = -np.inf

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cv = cross_val_score(
            Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)]),
            X, y, cv=5,
            scoring='r2').mean()

        print(f"{name:<22} | {r2:>8.4f} | "
              f"{mae:>8.2f} | {cv:>8.4f}")

        if cv > best_score:
            best_score = cv
            best_model = name

    print(f"\n✅ Best model: {best_model} "
          f"(CV R²={best_score:.4f})")

    # Sample predictions
    print("\n📊 Sample Predictions:")
    sample = X_test.head(4)
    actual = y_test.head(4)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(sample)

    print(f"  {'Actual':>8} | "
          f"{'Predicted':>10} | "
          f"{'Error':>8}")
    print("  " + "-" * 32)
    for act, pred in zip(
            actual.values, preds):
        print(f"  {act:>8.1f} | "
              f"{pred:>10.1f} | "
              f"{abs(act-pred):>8.1f}")

    print("\n🚀 This is the Student Performance")
    print("   Predictor preview! (Day 59!) 🎯")


if __name__ == "__main__":
    df = create_student_dataset()
    print(f"Student dataset: {df.shape}")
    print(f"Missing values: "
          f"{df.isnull().sum().sum()}\n")

    build_and_evaluate_pipeline(df)

    print("\n" + "=" * 50)
    print("  Phase 4 — What's Coming!")
    print("=" * 50)
    print("  Day 52 → Linear Regression (math!)")
    print("  Day 53 → Logistic Regression")
    print("  Day 54 → Decision Trees")
    print("  Day 55 → Random Forest")
    print("  Day 56 → SVM + KNN")
    print("  Day 57 → Model Evaluation")
    print("  Day 58 → Hyperparameter Tuning")
    print("  Day 59 → Student Performance Predictor!")
    print("  Day 63 → NLP Basics")
    print("  Day 69 → AI Hiring Assistant!")
