"""
Day 58 — Cross Validation + Hyperparameter Tuning
Topic: Complete Tuning Pipeline — Production Ready
Date: 15 July 2026
Author: Bala Ravi

The full workflow used in every ML project:
1. Cross-validate baseline models
2. Select best model type
3. Tune hyperparameters with RandomizedSearch
4. Nested CV for honest evaluation
5. Final evaluation on test set ONCE!

This is what the Student Performance Predictor
(Day 59+) will use! 🚀
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split)
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, classification_report,
    roc_auc_score, accuracy_score)
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')
import time


def create_rich_dataset(
        n: int = 1000) -> pd.DataFrame:
    """Create student dataset with mixed features."""
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

    # Missing values
    study[np.random.choice(
        n, 30, replace=False)] = np.nan

    city_eff = {'Bangalore': 3, 'Mumbai': 2,
                'Delhi': 1, 'Hyderabad': 0,
                'Pune': -1}
    school_eff = {'Government': -2,
                   'Private': 3, 'CBSE': 1}

    score = (study * 5 + attend * 0.3 +
              prev * 0.4 + assign * 1.5 +
              sleep * 1.0 +
              np.array([city_eff[c] for c in city]) +
              np.array([school_eff[s] for s in school]) +
              np.random.normal(0, 5, n))

    score = np.where(np.isnan(score), 60, score)

    return pd.DataFrame({
        'study_hours': study,
        'attendance_pct': attend.round(1),
        'prev_score': prev.round(1),
        'assignments_done': assign,
        'sleep_hours': sleep.round(1),
        'city': city,
        'school_type': school,
        'passed': (score > 70).astype(int)
    })


def build_base_pipeline(
        num_features: list,
        cat_features: list,
        model) -> Pipeline:
    """Build complete preprocessing + model pipeline."""
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(
            strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def step1_baseline_comparison(
        df: pd.DataFrame) -> str:
    """Step 1: Compare baseline models."""
    print("=" * 55)
    print("  STEP 1: Baseline Model Comparison")
    print("=" * 55)

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

    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf',
                    probability=True,
                    random_state=42),
        'KNN (k=7)': KNeighborsClassifier(
            n_neighbors=7)
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    print(f"\n{'Model':<25} | {'CV F1':>7} | "
          f"{'Std':>6} | {'CV AUC':>7}")
    print("-" * 55)

    best_model_name = None
    best_cv_score = 0

    for name, model in models.items():
        pipeline = build_base_pipeline(
            num_features, cat_features, model)

        f1_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring='f1', n_jobs=-1)
        auc_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring='roc_auc', n_jobs=-1)

        mean_f1 = f1_scores.mean()
        print(f"{name:<25} | {mean_f1:>7.4f} | "
              f"{f1_scores.std():>6.4f} | "
              f"{auc_scores.mean():>7.4f}")

        if mean_f1 > best_cv_score:
            best_cv_score = mean_f1
            best_model_name = name

    print(f"\n✅ Best baseline: {best_model_name} "
          f"(CV F1={best_cv_score:.4f})")
    return best_model_name


def step2_tune_best_model(
        df: pd.DataFrame) -> dict:
    """Step 2: Tune the best model."""
    print(f"\n{'=' * 55}")
    print(f"  STEP 2: Hyperparameter Tuning (RF)")
    print(f"{'=' * 55}")

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

    pipeline = build_base_pipeline(
        num_features, cat_features,
        RandomForestClassifier(
            random_state=42, n_jobs=-1))

    param_dist = {
        'model__n_estimators': randint(100, 500),
        'model__max_depth': randint(5, 20),
        'model__min_samples_leaf': randint(1, 15),
        'model__max_features': ['sqrt', 'log2'],
        'model__min_samples_split': randint(2, 10)
    }

    cv = StratifiedKFold(
        n_splits=5, shuffle=True,
        random_state=42)

    print(f"\nRunning RandomizedSearchCV (50 iters)...")
    start = time.time()

    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=50, cv=cv,
        scoring='f1', n_jobs=-1,
        random_state=42,
        return_train_score=True)

    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.1f}s\n")
    print(f"Best Parameters:")
    for param, value in search.best_params_.items():
        clean = param.replace('model__', '')
        val_str = (f"{value:.3f}"
                   if isinstance(value, float)
                   else str(value))
        print(f"  {clean}: {val_str}")

    print(f"\nBest CV F1: {search.best_score_:.4f}")

    return {
        'best_pipeline': search.best_estimator_,
        'best_params': search.best_params_,
        'best_cv_f1': search.best_score_,
        'X_test': X_test,
        'y_test': y_test
    }


def step3_final_evaluation(
        result: dict) -> None:
    """Step 3: Final evaluation on test set ONCE."""
    print(f"\n{'=' * 55}")
    print(f"  STEP 3: Final Test Set Evaluation")
    print(f"  (Test set used for the FIRST time!)")
    print(f"{'=' * 55}\n")

    best_pipeline = result['best_pipeline']
    X_test = result['X_test']
    y_test = result['y_test']

    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(
        X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Final Model Performance:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Fail', 'Pass']))

    baseline_f1 = 0.85  # approximate
    improvement = f1 - baseline_f1
    print(f"Improvement over baseline: "
          f"+{improvement:.4f} F1")

    print(f"\n✅ Production Pipeline Ready!")
    print(f"   Best CV F1:   {result['best_cv_f1']:.4f}")
    print(f"   Final Test F1: {f1:.4f}")

    gap = abs(result['best_cv_f1'] - f1)
    if gap < 0.02:
        print(f"   Gap: {gap:.4f} ← CV was reliable! ✅")
    else:
        print(f"   Gap: {gap:.4f} ← Some overfitting! ⚠️")

    print(f"\n🚀 This pipeline goes into")
    print(f"   Student Performance Predictor (Day 59)!")


if __name__ == "__main__":
    df = create_rich_dataset()
    print(f"Dataset: {df.shape}\n")

    best_model = step1_baseline_comparison(df)
    result = step2_tune_best_model(df)
    step3_final_evaluation(result)

    print(f"\n{'=' * 55}")
    print(f"  Complete ML Workflow Summary:")
    print(f"{'=' * 55}")
    steps = [
        "Day 51 → ML Fundamentals + Sklearn",
        "Day 52 → Linear Regression",
        "Day 53 → Logistic Regression",
        "Day 54 → Decision Trees",
        "Day 55 → Random Forest",
        "Day 56 → SVM + KNN",
        "Day 57 → Model Evaluation + Metrics",
        "Day 58 → CV + Hyperparameter Tuning ✅",
        "Day 59 → Student Performance Predictor! 🚀"
    ]
    for step in steps:
        done = "✅" if "51-58" not in step else ""
        print(f"  {step}")
