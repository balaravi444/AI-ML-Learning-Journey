"""
Skin Disease Detector
FastAPI Application — Main Entry Point
Author: Bala Ravi
GitHub: github.com/balaravi444

Upload skin image → AI diagnosis in seconds.
7 disease classes including Melanoma detection.

⚠️ DISCLAIMER: Educational AI tool only.
   NOT a replacement for medical diagnosis.
   Always consult a qualified dermatologist.
"""
import os
import io
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from fastapi import (
    FastAPI, HTTPException,
    UploadFile, File, Form)
from fastapi.responses import (
    HTMLResponse, JSONResponse)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

app = FastAPI(
    title="Skin Disease Detector",
    description=(
        "AI-powered skin disease detection. "
        "7 disease classes including Melanoma. "
        "Built by Bala Ravi — Day 79 of 90-day "
        "AI/ML learning journey.\n\n"
        "⚠️ DISCLAIMER: Educational tool only. "
        "Not a replacement for medical diagnosis."),
    version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])

# ─── Constants ────────────────────────────────────
CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Benign Keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic Nevi',
    'Vascular Lesion'
]

MELANOMA_IDX = 4
MELANOMA_THRESHOLD = 0.3  # Clinical threshold!

SEVERITY_MAP = {
    'Actinic Keratosis': 'MEDIUM',
    'Basal Cell Carcinoma': 'HIGH',
    'Benign Keratosis': 'LOW',
    'Dermatofibroma': 'LOW',
    'Melanoma': 'CRITICAL',
    'Melanocytic Nevi': 'LOW',
    'Vascular Lesion': 'LOW'
}

RECOMMENDATIONS = {
    'Actinic Keratosis': (
        "Schedule a dermatologist appointment "
        "within 2-4 weeks. This is a precancerous "
        "lesion that requires treatment to prevent "
        "progression to skin cancer."),
    'Basal Cell Carcinoma': (
        "Seek dermatologist consultation within "
        "1-2 weeks. Basal cell carcinoma is the "
        "most common skin cancer but highly treatable "
        "when caught early."),
    'Benign Keratosis': (
        "Routine monitoring recommended. This appears "
        "to be a benign lesion. Annual skin check "
        "with a dermatologist is advisable."),
    'Dermatofibroma': (
        "No urgent action required. Dermatofibromas "
        "are benign growths. Monitor for any changes "
        "in size, color, or shape."),
    'Melanoma': (
        "⚠️ SEEK IMMEDIATE MEDICAL ATTENTION. "
        "This lesion shows characteristics consistent "
        "with melanoma, the most serious form of skin "
        "cancer. Contact a dermatologist TODAY. "
        "Early detection dramatically improves outcomes."),
    'Melanocytic Nevi': (
        "Monitor regularly. This appears to be a "
        "common mole. Watch for ABCDE changes: "
        "Asymmetry, Border irregularity, Color change, "
        "Diameter increase, Evolution."),
    'Vascular Lesion': (
        "Generally benign. Vascular lesions are "
        "common and usually harmless. "
        "Consult a dermatologist if concerned.")
}

ALERT_LEVELS = {
    'LOW': 'GREEN',
    'MEDIUM': 'YELLOW',
    'HIGH': 'ORANGE',
    'CRITICAL': 'RED'
}

# ─── Global Model State ───────────────────────────
MODEL_STATE = {
    'model': None,
    'metadata': None,
    'loaded': False,
    'error': None
}

IMG_SIZE = 96  # 224 for production


def load_model_artifacts() -> None:
    """Load model at startup."""
    if not TF_AVAILABLE:
        MODEL_STATE['error'] = (
            "TensorFlow not installed")
        return

    model_dir = os.path.join(
        os.path.dirname(__file__), 'models')
    model_path = os.path.join(
        model_dir, 'final_model.h5')
    meta_path = os.path.join(
        model_dir, 'final_metadata.pkl')

    if os.path.exists(model_path):
        try:
            MODEL_STATE['model'] = (
                keras.models.load_model(model_path))
            MODEL_STATE['loaded'] = True
            print("✅ Model loaded successfully!")
        except Exception as e:
            MODEL_STATE['error'] = str(e)
            print(f"⚠️  Model load error: {e}")
    else:
        MODEL_STATE['error'] = (
            "Model not found. Run training first.")
        print("⚠️  No trained model found.")
        print("    Run: python days/day-77/code/"
              "03_train_phase1.py")

    if os.path.exists(meta_path):
        MODEL_STATE['metadata'] = (
            joblib.load(meta_path))


@app.on_event("startup")
async def startup():
    load_model_artifacts()


# ─── Image Preprocessing ──────────────────────────
def preprocess_image(
        image_bytes: bytes,
        img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Preprocess uploaded image for model input.

    MUST match training preprocessing exactly!
    MobileNetV2: resize + scale to -1 to 1

    Args:
        image_bytes: Raw image bytes
        img_size: Target size

    Returns:
        Preprocessed numpy array (1, H, W, 3)
    """
    if not PIL_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Pillow not installed. "
                   "Run: pip install pillow")

    try:
        img = Image.open(
            io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((img_size, img_size))
        img_array = np.array(img, dtype=np.float32)

        # MobileNetV2 preprocessing: -1 to 1
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Image processing error: {str(e)}. "
                   f"Ensure file is a valid image.")


def predict_disease(
        img_array: np.ndarray) -> dict:
    """
    Run model prediction on preprocessed image.

    Applies clinical threshold for Melanoma:
    If P(Melanoma) > 0.3 → predict Melanoma
    regardless of other class probabilities.

    Args:
        img_array: Preprocessed image array

    Returns:
        Prediction result dictionary
    """
    model = MODEL_STATE['model']

    if model is None:
        # Demo mode - return simulated prediction
        probs = np.random.dirichlet(
            np.ones(7) * 0.5)
        probs = probs.tolist()
    else:
        probs = model.predict(
            img_array, verbose=0)[0].tolist()

    # Apply clinical threshold for Melanoma
    mel_prob = probs[MELANOMA_IDX]
    if mel_prob >= MELANOMA_THRESHOLD:
        pred_idx = MELANOMA_IDX
    else:
        pred_idx = int(np.argmax(probs))

    diagnosis = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    severity = SEVERITY_MAP[diagnosis]
    recommendation = RECOMMENDATIONS[diagnosis]
    alert_level = ALERT_LEVELS[severity]

    all_probs = {
        cls: float(round(p, 4))
        for cls, p in zip(CLASS_NAMES, probs)}

    return {
        'diagnosis': diagnosis,
        'confidence': round(confidence, 4),
        'confidence_pct': f"{confidence*100:.1f}%",
        'severity': severity,
        'alert_level': alert_level,
        'recommendation': recommendation,
        'all_probabilities': all_probs,
        'melanoma_probability': round(mel_prob, 4),
        'melanoma_threshold_used': MELANOMA_THRESHOLD,
        'model_loaded': MODEL_STATE['loaded'],
        'disclaimer': (
            "⚠️ This AI tool is for educational "
            "purposes only and does NOT replace "
            "professional medical diagnosis. "
            "Always consult a qualified dermatologist.")
    }


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
        "<h1>Skin Disease Detector</h1>"
        "<p>Visit <a href='/docs'>/docs</a></p>")


@app.get("/api/health")
async def health():
    """Health check + model status."""
    meta = MODEL_STATE.get('metadata', {}) or {}
    return {
        "status": "ok",
        "model_loaded": MODEL_STATE['loaded'],
        "model_error": MODEL_STATE.get('error'),
        "tf_available": TF_AVAILABLE,
        "pil_available": PIL_AVAILABLE,
        "n_classes": len(CLASS_NAMES),
        "phase2_accuracy": meta.get(
            'phase2_acc', 'N/A'),
        "melanoma_threshold": MELANOMA_THRESHOLD
    }


@app.post("/api/predict")
async def predict(
        file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image.

    Accepts: JPEG, PNG, WebP
    Returns: Diagnosis + confidence + clinical info

    ⚠️ NOT for medical diagnosis decisions.
    """
    # Validate file type
    allowed = ['image/jpeg', 'image/png',
                'image/webp', 'image/jpg']
    if (file.content_type and
            file.content_type not in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: "
                   f"{file.content_type}. "
                   f"Use JPEG, PNG, or WebP.")

    # Read image
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded.")

    if len(contents) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=400,
            detail="File too large. Max 10MB.")

    # Preprocess + predict
    img_array = preprocess_image(
        contents, img_size=IMG_SIZE)
    result = predict_disease(img_array)
    result['filename'] = file.filename

    return JSONResponse(content=result)


@app.post("/api/predict/batch")
async def predict_batch(
        files: List[UploadFile] = File(...)):
    """
    Predict skin disease for multiple images.

    Max 10 images per batch.
    Returns list of predictions.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Max 10 images per batch.")

    results = []
    for file in files:
        try:
            contents = await file.read()
            img_array = preprocess_image(
                contents, img_size=IMG_SIZE)
            result = predict_disease(img_array)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)})

    # Summary
    diagnoses = [
        r.get('diagnosis', 'error')
        for r in results
        if 'diagnosis' in r]
    critical = [
        r for r in results
        if r.get('severity') == 'CRITICAL']

    return {
        "total": len(files),
        "successful": len(diagnoses),
        "results": results,
        "summary": {
            cls: diagnoses.count(cls)
            for cls in set(diagnoses)},
        "critical_cases": len(critical),
        "disclaimer": (
            "AI tool only. Consult a dermatologist.")
    }


@app.get("/api/classes")
async def get_classes():
    """Get all disease classes with severity info."""
    classes = []
    for idx, cls_name in enumerate(CLASS_NAMES):
        severity = SEVERITY_MAP[cls_name]
        classes.append({
            'index': idx,
            'name': cls_name,
            'severity': severity,
            'alert_level': ALERT_LEVELS[severity],
            'recommendation': RECOMMENDATIONS[cls_name]
        })
    return {
        "n_classes": len(CLASS_NAMES),
        "classes": classes,
        "melanoma_idx": MELANOMA_IDX,
        "melanoma_threshold": MELANOMA_THRESHOLD
    }


@app.get("/api/stats")
async def model_stats():
    """Model performance statistics."""
    meta = MODEL_STATE.get('metadata', {}) or {}
    clinical = meta.get('clinical_metrics', {})
    mel = clinical.get('Melanoma', {})

    return {
        "model": "MobileNetV2 + Transfer Learning",
        "dataset": "HAM10000 (synthetic for demo)",
        "img_size": f"{IMG_SIZE}×{IMG_SIZE}",
        "n_classes": len(CLASS_NAMES),
        "phase1_accuracy": meta.get(
            'phase1_acc', 'N/A'),
        "phase2_accuracy": meta.get(
            'phase2_acc', 'N/A'),
        "melanoma_recall": mel.get(
            'recall', 'N/A'),
        "melanoma_auc": mel.get(
            'auc', 'N/A'),
        "melanoma_threshold": MELANOMA_THRESHOLD,
        "tech_stack": [
            "Python", "TensorFlow", "Keras",
            "MobileNetV2", "FastAPI",
            "Transfer Learning"],
        "author": "Bala Ravi",
        "day": "Day 79 of 90",
        "github": (
            "github.com/balaravi444/"
            "AI-ML-Learning-Journey"),
        "disclaimer": (
            "Educational purposes only.")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8004,
        reload=True)
