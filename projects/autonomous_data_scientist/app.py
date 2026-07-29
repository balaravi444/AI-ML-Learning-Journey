"""
Autonomous Data Scientist
FastAPI Application — Main Entry Point
Author: Bala Ravi
GitHub: github.com/balaravi444

Upload any CSV → Auto profile → Auto preprocess
→ AutoML → SHAP explanations → Live prediction API
"""
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from fastapi import (
    FastAPI, HTTPException,
    UploadFile, File, BackgroundTasks)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="Autonomous Data Scientist",
    description=(
        "Upload any CSV → Get predictions + explanations. "
        "AutoML + SHAP. Built by Bala Ravi — "
        "Day 69 of 90-day AI/ML learning journey."),
    version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])

# ─── Global State ─────────────────────────────────
STATE = {
    'model': None,
    'preprocessor': None,
    'metadata': None,
    'shap_explainer': None,
    'shap_values': None,
    'X_train': None,
    'feature_names': None,
    'training_status': 'idle',
    'training_report': None
}

MODEL_DIR = os.path.join(
    os.path.dirname(__file__), 'ml')


def load_artifacts() -> None:
    """Load saved model artifacts at startup."""
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    prep_path = os.path.join(
        MODEL_DIR, 'preprocessor.pkl')
    meta_path = os.path.join(MODEL_DIR, 'metadata.pkl')

    if all(os.path.exists(p) for p in [
            model_path, prep_path, meta_path]):
        model_data = joblib.load(model_path)
        STATE['model'] = model_data['model']
        STATE['metadata'] = joblib.load(meta_path)
        STATE['preprocessor'] = joblib.load(prep_path)
        STATE['training_status'] = 'ready'
        print("✅ Artifacts loaded!")
    else:
        print("⚠️  No model found — upload CSV to train")


@app.on_event("startup")
async def startup():
    load_artifacts()


# ─── Schemas ──────────────────────────────────────
class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        example={
            "age": 32,
            "salary": 450000,
            "department": "Engineering",
            "years_experience": 4.5,
            "performance_score": 3.8,
            "satisfaction_score": 2.1,
            "overtime": 1})


class TrainRequest(BaseModel):
    target_col: str = Field(
        ..., example="churn")


# ─── Endpoints ────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve dashboard."""
    html_path = os.path.join(
        os.path.dirname(__file__),
        'templates', 'index.html')
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return HTMLResponse(
        "<h1>Autonomous Data Scientist</h1>"
        "<p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/api/health")
async def health():
    """Health check."""
    meta = STATE.get('metadata', {})
    return {
        "status": "ok",
        "model_loaded": STATE['model'] is not None,
        "training_status": STATE['training_status'],
        "model_type": meta.get('best_model', 'none'),
        "task_type": meta.get('task_type', 'none')
    }


@app.post("/api/train")
async def train_model(
        background_tasks: BackgroundTasks,
        target_col: str = "churn",
        file: UploadFile = File(...)):
    """
    Upload CSV and train AutoDS pipeline.
    Runs in background — poll /api/status.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files supported")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(
        contents.decode('utf-8')))

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{target_col}' not found. "
                   f"Available: {list(df.columns)}")

    STATE['training_status'] = 'training'

    background_tasks.add_task(
        run_training_pipeline,
        df, target_col)

    return {
        "message": "Training started!",
        "rows": len(df),
        "columns": len(df.columns),
        "target": target_col,
        "status": "training",
        "poll": "/api/status"
    }


def run_training_pipeline(
        df: pd.DataFrame,
        target_col: str) -> None:
    """Background training task."""
    try:
        import sys
        sys.path.insert(
            0, os.path.dirname(__file__))

        from days_day_67_code_01_auto_data_profiler \
            import AutoDataProfiler
        from days_day_67_code_02_auto_preprocessor \
            import AutoPreprocessor
        from days_day_68_code_01_automl_engine \
            import AutoMLEngine
        from sklearn.model_selection import (
            train_test_split)

        # Profile
        profiler = AutoDataProfiler()
        profile = profiler.profile(df)

        # Preprocess
        preprocessor = AutoPreprocessor()
        pipeline, y = preprocessor.build_pipeline(
            df, target_col=target_col)

        drop_cols = preprocessor.drop_cols.copy()
        if target_col in df.columns:
            drop_cols.append(target_col)
        X_raw = df.drop(
            columns=[c for c in drop_cols
                     if c in df.columns],
            errors='ignore')
        X = pipeline.transform(X_raw)

        # Split
        stratify = (
            y if preprocessor.task_type ==
            'classification' else None)
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X, y, test_size=0.2,
                random_state=42,
                stratify=stratify))

        # AutoML
        engine = AutoMLEngine(
            n_iter=15, cv_folds=5,
            top_n_to_tune=3)
        report = engine.fit(
            X_train, y_train,
            X_test, y_test,
            task_type=preprocessor.task_type,
            label_encoder=preprocessor.label_encoder)

        # Save
        os.makedirs(MODEL_DIR, exist_ok=True)
        engine.save_best_model(MODEL_DIR)
        preprocessor.save(
            os.path.join(MODEL_DIR, 'preprocessor.pkl'))

        best = next(
            r for r in report.all_results
            if r.model_name == report.best_model_name)

        metadata = {
            'target_col': target_col,
            'task_type': preprocessor.task_type,
            'best_model': report.best_model_name,
            'metrics': best.metrics,
            'n_features_in': X.shape[1],
            'quality_score': profile.quality_score,
            'label_classes': (
                list(preprocessor.label_encoder.classes_)
                if preprocessor.label_encoder
                else None),
            'feature_columns': list(X_raw.columns)
        }
        joblib.dump(
            metadata,
            os.path.join(MODEL_DIR, 'metadata.pkl'))

        # Update state
        STATE['model'] = report.best_model
        STATE['preprocessor'] = preprocessor
        STATE['metadata'] = metadata
        STATE['X_train'] = X_train
        STATE['training_status'] = 'ready'
        STATE['training_report'] = {
            'best_model': report.best_model_name,
            'best_score': report.best_score,
            'task_type': preprocessor.task_type,
            'total_time': report.total_time,
            'all_results': [
                {
                    'model': r.model_name,
                    'baseline': r.baseline_score,
                    'tuned': r.tuned_score,
                    'test': r.test_score
                }
                for r in report.all_results
            ]
        }

    except Exception as e:
        STATE['training_status'] = f'error: {str(e)}'
        raise


@app.get("/api/status")
async def training_status():
    """Get training status."""
    return {
        "status": STATE['training_status'],
        "model_ready": STATE['model'] is not None,
        "report": STATE.get('training_report')
    }


@app.post("/api/predict")
async def predict(request: PredictRequest):
    """
    Predict for new data row.
    Returns prediction + SHAP explanation.
    """
    if STATE['model'] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Upload CSV first.")

    meta = STATE['metadata']
    preprocessor = STATE['preprocessor']
    model = STATE['model']

    try:
        # Build input row
        input_df = pd.DataFrame([request.features])

        # Handle missing columns
        if meta.get('feature_columns'):
            for col in meta['feature_columns']:
                if col not in input_df.columns:
                    input_df[col] = 0

        X = preprocessor.transform(input_df)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Feature processing error: {str(e)}")

    # Predict
    task_type = meta.get('task_type', 'classification')

    if task_type == 'classification':
        pred_idx = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        classes = meta.get('label_classes')
        if classes:
            prediction = classes[pred_idx]
            probabilities = {
                cls: float(round(p, 4))
                for cls, p in zip(classes, proba)}
        else:
            prediction = int(pred_idx)
            probabilities = {
                str(i): float(round(p, 4))
                for i, p in enumerate(proba)}

        confidence = float(proba.max())

    else:
        pred_val = float(model.predict(X)[0])
        prediction = pred_val
        probabilities = {}
        confidence = None

    # SHAP explanation
    explanation = _get_shap_explanation(X, model)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "task_type": task_type,
        "explanation": explanation,
        "model_used": meta.get('best_model', 'unknown')
    }


def _get_shap_explanation(
        X: np.ndarray,
        model) -> List[dict]:
    """Get SHAP explanation for single row."""
    try:
        import shap

        if STATE['X_train'] is not None:
            explainer = shap.TreeExplainer(model)
        else:
            return []

        shap_vals = explainer.shap_values(X)

        if isinstance(shap_vals, list):
            shap_use = shap_vals[1][0]
        else:
            shap_use = shap_vals[0]

        meta = STATE['metadata']
        n_features = len(shap_use)

        feature_labels = [
            f"feature_{i}"
            for i in range(n_features)]

        if (meta and
                meta.get('feature_columns')):
            from days_day_67_code_02_auto_preprocessor \
                import AutoPreprocessor
            preprocessor = STATE['preprocessor']
            if preprocessor:
                feature_labels = (
                    feature_labels[:n_features])

        explanation = []
        for i, (val, feat) in enumerate(
                zip(shap_use, feature_labels)):
            explanation.append({
                'feature': feat,
                'shap_value': float(round(val, 4)),
                'direction': (
                    'increases prediction'
                    if val > 0.01 else
                    'decreases prediction'
                    if val < -0.01 else
                    'neutral')
            })

        return sorted(
            explanation,
            key=lambda x: abs(x['shap_value']),
            reverse=True)[:10]

    except Exception:
        return []


@app.get("/api/report")
async def get_report():
    """Get AutoML training report."""
    if STATE['training_report'] is None:
        raise HTTPException(
            status_code=404,
            detail="No training report available")
    return STATE['training_report']


@app.get("/api/explain/global")
async def global_explanation():
    """Get global feature importance."""
    if STATE['model'] is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded")

    try:
        importances = (
            STATE['model'].feature_importances_)
        n = len(importances)
        features = [
            {'feature': f'feature_{i}',
             'importance': float(round(imp, 4))}
            for i, imp in enumerate(importances)]
        features.sort(
            key=lambda x: x['importance'],
            reverse=True)
        return {
            "global_importance": features[:15],
            "model": STATE['metadata'].get(
                'best_model', 'unknown')
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e))


@app.get("/api/stats")
async def model_stats():
    """Model performance statistics."""
    meta = STATE.get('metadata', {})
    return {
        "model": meta.get('best_model', 'none'),
        "task_type": meta.get('task_type', 'none'),
        "metrics": meta.get('metrics', {}),
        "quality_score": meta.get(
            'quality_score', 0),
        "author": "Bala Ravi",
        "day": "Day 69 of 90",
        "github": (
            "github.com/balaravi444/"
            "AI-ML-Learning-Journey")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8003,
        reload=True)
