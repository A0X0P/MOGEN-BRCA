"""Inference package: load trained models and run predictions on Patient records."""

from src.inference.predict import (
    InferenceArtifacts,
    PredictionResult,
    TaskPrediction,
    load_model,
    predict,
    predict_batch,
)

__all__ = [
    "InferenceArtifacts",
    "PredictionResult",
    "TaskPrediction",
    "load_model",
    "predict",
    "predict_batch",
]
