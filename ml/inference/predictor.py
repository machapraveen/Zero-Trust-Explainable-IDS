"""
Model Predictor
===============
Real-time prediction interface unified with the 10-Model Zero-Trust Deep Learning
& SHAP Explainability Engine.
"""

from datetime import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ml.zero_trust_pipeline import MULTILEVEL_FEATURES, get_zero_trust_engine

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Real-time model predictor backed by the 10-Model Zero-Trust Deep Ensemble
    and SHAP Interpretability Layer.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/")
        self.config = config or {}
        self.threshold = float(self.config.get("threshold", 0.5))
        self.batch_size = int(self.config.get("batch_size", 32))
        self.engine = get_zero_trust_engine()
        self.feature_names = MULTILEVEL_FEATURES
        self.num_features = len(self.feature_names)
        self.metadata: Dict[str, Any] = {
            "initialized_at": self.engine.initialized_at or datetime.utcnow().isoformat(),
            "version": "2.0.0-zero-trust",
            "status": "loaded" if self.engine.initialized else "initializing",
        }

    def is_loaded(self) -> bool:
        """Check whether the 10-model ensemble + SHAP explainer are initialized."""
        return bool(self.engine and self.engine.initialized)

    def predict(
        self,
        features: Union[np.ndarray, List[float], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Make real-time 10-model + SHAP + Zero-Trust prediction."""
        if isinstance(features, dict):
            flow = features
        else:
            arr = np.asarray(features, dtype=float).flatten()
            flow = {
                name: float(arr[idx]) if idx < len(arr) else 0.0
                for idx, name in enumerate(self.feature_names)
            }

        analysis = self.engine.analyze_flow(flow, include_shap=True)
        prob = float(analysis["threat_probability"])
        label = "malicious" if prob >= self.threshold else "benign"
        return {
            "probability": prob,
            "label": label,
            "attack_type": analysis["attack_type"],
            "threshold": self.threshold,
            "confidence": analysis["confidence"],
            "model_used": analysis["model_used"],
            "ensemble_votes": analysis["ensemble_votes"],
            "shap_explanation": analysis["shap_explanation"],
            "zero_trust_decision": analysis["zero_trust_decision"],
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": self.metadata["version"],
        }

    def predict_batch(
        self,
        features_batch: Union[np.ndarray, List[Any]],
    ) -> List[Dict[str, Any]]:
        return [self.predict(item) for item in features_batch]

    def get_feature_importance(self) -> Dict[str, float]:
        xgb_model = self.engine.ml_models.get("xgboost")
        if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
            return {
                name: round(float(val), 5)
                for name, val in zip(self.feature_names, xgb_model.feature_importances_)
            }
        return {name: round(1.0 / self.num_features, 5) for name in self.feature_names}

    def get_model_info(self) -> Dict[str, Any]:
        return self.engine.get_status_summary()

    def update_threshold(self, threshold: float) -> None:
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")

    def health_check(self) -> Dict[str, Any]:
        summary = self.engine.get_status_summary()
        return {
            "healthy": summary["initialized"] and summary["shap_explainer_active"],
            "status": "loaded" if summary["initialized"] else "uninitialized",
            "models_loaded": summary["initialized"],
            "n_models": summary["n_models"],
            "shap_explainer_active": summary["shap_explainer_active"],
            "predictions_count": summary["predictions_count"],
        }
