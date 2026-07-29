"""
Day 68 — Autonomous Data Scientist
Topic: AutoML Engine — Auto Model Selection + Tuning
Date: 25 July 2026
Author: Bala Ravi

Tries 6 models automatically, tunes top 3,
picks the best. Zero human decisions needed.
"""
import numpy as np
import pandas as pd
import joblib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, KFold,
    cross_val_score, train_test_split)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error,
    mean_squared_error, classification_report)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform, loguniform
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """Result for a single model evaluation."""
    model_name: str
    task_type: str
    baseline_score: float
    tuned_score: float
    test_score: float
    train_time: float
    best_params: Dict = field(default_factory=dict)
    model: Any = None
    metrics: Dict = field(default_factory=dict)


@dataclass
class AutoMLReport:
    """Complete AutoML training report."""
    task_type: str
    n_rows: int
    n_features: int
    n_classes: Optional[int]
    scoring_metric: str
    all_results: List[ModelResult] = (
        field(default_factory=list))
    best_model_name: str = ""
    best_score: float = 0.0
    best_model: Any = None
    total_time: float = 0.0


# Model candidates per task type
CLASSIFICATION_MODELS = {
    'Logistic Regression': {
        'model': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42),
        'params': {
            'C': loguniform(0.01, 100),
            'solver': ['lbfgs', 'liblinear'],
            'penalty': ['l2']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(
            random_state=42, n_jobs=-1,
            class_weight='balanced'),
        'params': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2']
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            random_state=42),
        'params': {
            'n_estimators': randint(50, 200),
            'max_depth': randint(2, 8),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4)
        }
    },
    'Extra Trees': {
        'model': ExtraTreesClassifier(
            random_state=42, n_jobs=-1,
            class_weight='balanced'),
        'params': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 20),
            'min_samples_leaf': randint(1, 10)
        }
    },
    'SVM': {
        'model': SVC(
            probability=True,
            class_weight='balanced',
            random_state=42),
        'params': {
            'C': loguniform(0.1, 100),
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': randint(3, 20),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}

REGRESSION_MODELS = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Ridge': {
        'model': Ridge(random_state=42),
        'params': {
            'alpha': loguniform(0.01, 100)
        }
    },
    'Random Forest': {
        'model': RandomForestRegressor(
            random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 20),
            'min_samples_leaf': randint(1, 10)
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(
            random_state=42),
        'params': {
            'n_estimators': randint(50, 200),
            'max_depth': randint(2, 8),
            'learning_rate': loguniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4)
        }
    },
    'Extra Trees': {
        'model': ExtraTreesRegressor(
            random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 20),
            'min_samples_leaf': randint(1, 10)
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': loguniform(0.1, 100),
            'gamma': ['scale', 'auto'],
            'epsilon': uniform(0.01, 0.5)
        }
    }
}


class AutoMLEngine:
    """
    Autonomous ML engine.

    Given preprocessed X, y:
    1. Detects task type
    2. Runs baseline for all models
    3. Tunes top 3 models
    4. Returns best model + full report
    """

    def __init__(self,
                 n_iter: int = 20,
                 cv_folds: int = 5,
                 top_n_to_tune: int = 3,
                 random_state: int = 42
                 ) -> None:
        """
        Initialize AutoML engine.

        Args:
            n_iter: RandomizedSearch iterations
            cv_folds: CV folds for evaluation
            top_n_to_tune: Top N models to tune
            random_state: Random seed
        """
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.top_n = top_n_to_tune
        self.random_state = random_state
        self.report = None

    def _get_models(
            self,
            task_type: str) -> dict:
        """Get model candidates for task type."""
        if task_type == 'classification':
            return CLASSIFICATION_MODELS
        return REGRESSION_MODELS

    def _get_scoring(
            self,
            task_type: str) -> str:
        """Get scoring metric for task type."""
        if task_type == 'classification':
            return 'f1_weighted'
        return 'r2'

    def _get_cv(
            self,
            task_type: str,
            y: np.ndarray):
        """Get CV strategy for task type."""
        if task_type == 'classification':
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state)
        return KFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state)

    def _evaluate_classification(
            self,
            model,
            X_test: np.ndarray,
            y_test: np.ndarray,
            label_encoder: Optional[LabelEncoder]
            ) -> dict:
        """Full classification metrics."""
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': float(
                accuracy_score(y_test, y_pred)),
            'f1_weighted': float(
                f1_score(y_test, y_pred,
                         average='weighted',
                         zero_division=0)),
            'f1_macro': float(
                f1_score(y_test, y_pred,
                         average='macro',
                         zero_division=0))
        }
        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                metrics['roc_auc'] = float(
                    roc_auc_score(
                        y_test, y_prob[:, 1]))
            else:
                metrics['roc_auc'] = float(
                    roc_auc_score(
                        y_test, y_prob,
                        multi_class='ovr',
                        average='weighted'))
        except Exception:
            pass
        return metrics

    def _evaluate_regression(
            self,
            model,
            X_test: np.ndarray,
            y_test: np.ndarray) -> dict:
        """Full regression metrics."""
        y_pred = model.predict(X_test)
        return {
            'r2': float(r2_score(y_test, y_pred)),
            'mae': float(
                mean_absolute_error(y_test, y_pred)),
            'rmse': float(
                np.sqrt(mean_squared_error(
                    y_test, y_pred))),
            'mape': float(
                np.mean(np.abs(
                    (y_test - y_pred) /
                    (np.abs(y_test) + 1e-10)
                )) * 100)
        }

    def _run_baseline(
            self,
            models: dict,
            X_train: np.ndarray,
            y_train: np.ndarray,
            task_type: str) -> List[tuple]:
        """
        Run baseline (no tuning) for all models.

        Returns:
            List of (name, score) sorted by score
        """
        print("\n  📊 Baseline Evaluation:")
        print(f"  {'Model':<25} | "
              f"{'CV Score':>10} | "
              f"{'Time':>8}")
        print(f"  {'-'*48}")

        scoring = self._get_scoring(task_type)
        cv = self._get_cv(task_type, y_train)

        baseline_scores = []
        for name, config in models.items():
            start = time.time()
            try:
                scores = cross_val_score(
                    config['model'],
                    X_train, y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1)
                score = scores.mean()
                elapsed = time.time() - start
                print(f"  {name:<25} | "
                      f"{score:>10.4f} | "
                      f"{elapsed:>7.1f}s")
                baseline_scores.append(
                    (name, score))
            except Exception as e:
                print(f"  {name:<25} | "
                      f"{'FAILED':>10} | "
                      f"{str(e)[:20]}")

        return sorted(
            baseline_scores,
            key=lambda x: x[1],
            reverse=True)

    def _tune_model(
            self,
            name: str,
            config: dict,
            X_train: np.ndarray,
            y_train: np.ndarray,
            task_type: str) -> tuple:
        """
        Tune a single model with RandomizedSearch.

        Returns:
            (best_estimator, best_score, best_params)
        """
        scoring = self._get_scoring(task_type)
        cv = self._get_cv(task_type, y_train)

        params = config['params']
        model = config['model']

        if not params:
            # No params to tune — just fit
            model.fit(X_train, y_train)
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=scoring,
                n_jobs=-1)
            return model, scores.mean(), {}

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=self.n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.random_state,
            refit=True)

        search.fit(X_train, y_train)
        return (search.best_estimator_,
                search.best_score_,
                search.best_params_)

    def fit(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_test: np.ndarray,
             y_test: np.ndarray,
             task_type: str,
             label_encoder: Optional[
                 LabelEncoder] = None
             ) -> AutoMLReport:
        """
        Run complete AutoML pipeline.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            task_type: 'classification' or 'regression'
            label_encoder: For classification labels

        Returns:
            AutoMLReport with all results
        """
        total_start = time.time()
        models = self._get_models(task_type)

        n_classes = (
            len(np.unique(y_train))
            if task_type == 'classification'
            else None)

        self.report = AutoMLReport(
            task_type=task_type,
            n_rows=len(X_train) + len(X_test),
            n_features=X_train.shape[1],
            n_classes=n_classes,
            scoring_metric=self._get_scoring(
                task_type))

        print(f"\n{'='*55}")
        print(f"  AutoML Engine Starting")
        print(f"  Task:     {task_type}")
        print(f"  Train:    {X_train.shape}")
        print(f"  Test:     {X_test.shape}")
        print(f"  Models:   {len(models)}")
        print(f"  Metric:   "
              f"{self.report.scoring_metric}")
        print(f"{'='*55}")

        # Step 1: Baseline
        print(f"\n  Step 1: Baseline Evaluation")
        ranked = self._run_baseline(
            models, X_train, y_train, task_type)

        # Step 2: Tune top N
        print(f"\n  Step 2: Tuning Top "
              f"{self.top_n} Models")

        tuned_results = []
        for rank, (name, baseline_score) in (
                enumerate(ranked[:self.top_n], 1)):
            print(f"\n  [{rank}/{self.top_n}] "
                  f"Tuning: {name}...")

            start = time.time()
            config = models[name]
            tuned_model, tuned_score, best_params = (
                self._tune_model(
                    name, config,
                    X_train, y_train, task_type))
            train_time = time.time() - start

            # Evaluate on test set
            if task_type == 'classification':
                metrics = self._evaluate_classification(
                    tuned_model, X_test, y_test,
                    label_encoder)
                test_score = metrics['f1_weighted']
            else:
                metrics = self._evaluate_regression(
                    tuned_model, X_test, y_test)
                test_score = metrics['r2']

            result = ModelResult(
                model_name=name,
                task_type=task_type,
                baseline_score=baseline_score,
                tuned_score=tuned_score,
                test_score=test_score,
                train_time=train_time,
                best_params=best_params,
                model=tuned_model,
                metrics=metrics)

            tuned_results.append(result)
            self.report.all_results.append(result)

            improvement = tuned_score - baseline_score
            print(f"    Baseline: {baseline_score:.4f}")
            print(f"    Tuned:    {tuned_score:.4f} "
                  f"(+{improvement:.4f})")
            print(f"    Test:     {test_score:.4f}")
            print(f"    Time:     {train_time:.1f}s")

        # Step 3: Pick best
        best_result = max(
            tuned_results,
            key=lambda r: r.tuned_score)

        self.report.best_model_name = (
            best_result.model_name)
        self.report.best_score = (
            best_result.tuned_score)
        self.report.best_model = (
            best_result.model)
        self.report.total_time = (
            time.time() - total_start)

        return self.report

    def print_report(
            self,
            report: AutoMLReport) -> None:
        """Print formatted AutoML report."""
        print(f"\n{'='*55}")
        print(f"  AutoML Report")
        print(f"{'='*55}")
        print(f"\n  Task:      {report.task_type}")
        print(f"  Rows:      {report.n_rows:,}")
        print(f"  Features:  {report.n_features}")
        print(f"  Metric:    {report.scoring_metric}")
        print(f"  Duration:  "
              f"{report.total_time:.1f}s")

        print(f"\n  Model Comparison:")
        print(f"  {'Model':<25} | "
              f"{'Baseline':>9} | "
              f"{'Tuned':>7} | "
              f"{'Test':>7}")
        print(f"  {'-'*55}")

        for r in sorted(
                report.all_results,
                key=lambda x: x.tuned_score,
                reverse=True):
            marker = (
                " ← BEST"
                if r.model_name ==
                report.best_model_name else "")
            print(f"  {r.model_name:<25} | "
                  f"{r.baseline_score:>9.4f} | "
                  f"{r.tuned_score:>7.4f} | "
                  f"{r.test_score:>7.4f}"
                  f"{marker}")

        best = next(
            r for r in report.all_results
            if r.model_name ==
            report.best_model_name)

        print(f"\n  🏆 Best Model: "
              f"{report.best_model_name}")
        print(f"  Best Params:")
        for k, v in best.best_params.items():
            val_str = (
                f"{v:.4f}"
                if isinstance(v, float)
                else str(v))
            print(f"    {k}: {val_str}")

        print(f"\n  Final Metrics:")
        for k, v in best.metrics.items():
            print(f"    {k}: {v:.4f}")

        print(f"\n{'='*55}")

    def save_best_model(
            self,
            save_dir: str) -> str:
        """Save best model to disk."""
        if self.report is None:
            raise ValueError(
                "Run fit() first!")

        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir, 'best_model.pkl')

        save_data = {
            'model': self.report.best_model,
            'model_name': self.report.best_model_name,
            'task_type': self.report.task_type,
            'metrics': next(
                r.metrics for r in
                self.report.all_results
                if r.model_name ==
                self.report.best_model_name),
            'n_features': self.report.n_features
        }

        joblib.dump(save_data, path)
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"\n✅ Best model saved: {path}")
        print(f"   Model: {self.report.best_model_name}")
        print(f"   Size:  {size:.1f} MB")
        return path


def demo_automl() -> None:
    """Full AutoML demo on employee churn."""
    from sklearn.datasets import make_classification

    print("=== AutoML Engine Demo ===\n")

    # Synthetic classification dataset
    X, y = make_classification(
        n_samples=800,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        random_state=42,
        class_sep=0.8)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X, y, test_size=0.2,
            random_state=42, stratify=y))

    engine = AutoMLEngine(
        n_iter=15,
        cv_folds=5,
        top_n_to_tune=3)

    report = engine.fit(
        X_train, y_train,
        X_test, y_test,
        task_type='classification')

    engine.print_report(report)

    # Save
    engine.save_best_model(
        "projects/autonomous_data_scientist/ml")


if __name__ == "__main__":
    demo_automl()
