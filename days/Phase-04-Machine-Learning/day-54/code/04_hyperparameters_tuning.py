"""
Day 54 — Decision Trees
Topic: Hyperparameter Tuning + Complete Pipeline
Date: 11 July 2026
Author: Bala Ravi

Finding the best tree parameters!
Preview of GridSearchCV (Day 58)!
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV)
from sklearn.metrics import (
    f1_score, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


def create_mixed_dataset(
        n: int = 1000) -> pd.DataFrame:
    """Create student dataset with categorical features."""
    np.random.seed(42)

    cities = ['Bangalore', 'Mumbai',
              'Delhi', 'Hyderabad', 'Pune']
    schools = ['Government', 'Private', 'CBSE']

    study = np.random.uniform(1, 10, n)
    attend = np.random.uniform(40, 100, n)
    prev = np.random.uniform(30, 95, n)
    assign = np.random.randint(0, 10, n)
    sleep = np.random.uniform(4, 9, n)
    city = np.random.choice(cities, n)
    school = np.random.choice(schools, n)

    # Add missing values
    study[np.random.choice(n, 30,
                            replace=False)] = np.nan

    city_effect = {
        'Bangalore': 3, 'Mumbai': 2,
        'Delhi': 1, 'Hyderabad': 0, 'Pune': -1}
    school_effect = {
        'Government': -2, 'Private': 3, 'CBSE': 1}

    score = (study * 5 + attend * 0.3 +
              prev * 0.4 + assign * 1.5 +
              sleep * 1.0 +
              np.array([city_effect[c] for c in city]) +
              np.array([school_effect[s] for s in school]) +
              np.random.normal(0, 5, n))

    score = np.where(np.isnan(score), 60, score)
    passed = (score > 70).astype(int)

    return pd.DataFrame({
        'study_hours': study,
        'attendance_pct': attend.round(1),
        'prev_score': prev.round(1),
        'assignments_done': assign,
        'sleep_hours': sleep.round(1),
        'city': city,
        'school_type': school,
        'passed': passed
    })


def manual_hyperparameter_search(
        df: pd.DataFrame) -> None:
    """Search hyperparameters manually."""
    print("=== Manual Hyperparameter Search ===\n")

    num_features = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    cat_features = ['city', 'school_type']

    X = df[num_features + cat_features]
    y = df['passed']

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])

    print(f"{'Depth':>6} | {'Min Leaf':>9} | "
          f"{'Train F1':>9} | {'Test F1':>8} | "
          f"{'CV F1':>7} | {'Status':>15}")
    print("-" * 65)

    best_cv = 0
    best_params = {}

    for depth in [3, 5, 7, 10, None]:
        for min_leaf in [5, 10, 20]:
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_leaf=min_leaf,
                    random_state=42))
            ])
            pipe.fit(X_train, y_train)

            train_f1 = f1_score(
                y_train, pipe.predict(X_train))
            test_f1 = f1_score(
                y_test, pipe.predict(X_test))
            cv_f1 = cross_val_score(
                pipe, X, y, cv=5,
                scoring='f1').mean()

            gap = train_f1 - test_f1
            if gap > 0.1:
                status = "⚠️ Overfit"
            elif test_f1 > 0.85:
                status = "✅ Great"
            elif test_f1 > 0.75:
                status = "⚡ Good"
            else:
                status = "❌ Underfit"

            depth_str = str(depth) if depth else "None"
            print(f"{depth_str:>6} | {min_leaf:>9} | "
                  f"{train_f1:>9.4f} | {test_f1:>8.4f} | "
                  f"{cv_f1:>7.4f} | {status:>15}")

            if cv_f1 > best_cv:
                best_cv = cv_f1
                best_params = {
                    'max_depth': depth,
                    'min_samples_leaf': min_leaf}

    print(f"\n✅ Best params: {best_params}")
    print(f"   Best CV F1: {best_cv:.4f}")


def grid_search_demo(
        df: pd.DataFrame) -> None:
    """Use GridSearchCV for automated tuning."""
    print("\n=== GridSearchCV (Preview of Day 58!) ===\n")

    num_features = ['study_hours', 'attendance_pct',
                     'prev_score', 'assignments_done',
                     'sleep_hours']
    cat_features = ['city', 'school_type']

    X = df[num_features + cat_features]
    y = df['passed']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', DecisionTreeClassifier(
            random_state=42))
    ])

    param_grid = {
        'model__max_depth': [3, 5, 7, 10],
        'model__min_samples_leaf': [5, 10, 20],
        'model__criterion': ['gini', 'entropy']
    }

    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=5, scoring='f1',
        n_jobs=-1)

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.2,
                         random_state=42,
                         stratify=y))

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters:")
    for param, value in (
            grid_search.best_params_.items()):
        clean_name = param.replace('model__', '')
        print(f"  {clean_name}: {value}")

    print(f"\nBest CV F1: "
          f"{grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    print(f"Test F1:    {test_f1:.4f}")

    print(f"\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Fail', 'Pass']))

    print(f"\n💡 GridSearchCV automatically tries ALL")
    print(f"   {len(param_grid['model__max_depth']) * len(param_grid['model__min_samples_leaf']) * len(param_grid['model__criterion'])} parameter combinations!")
    print(f"   We'll use this properly on Day 58! 🔥")


if __name__ == "__main__":
    df = create_mixed_dataset()
    print(f"Dataset: {df.shape}\n")

    manual_hyperparameter_search(df)
    grid_search_demo(df)
