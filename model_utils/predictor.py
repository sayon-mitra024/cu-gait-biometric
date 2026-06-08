"""
predictor.py
Mirrors the exact GEI pipeline from the CU Gait Biometric Colab notebook.
"""

import os
import pickle
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class GaitPredictor:
    def __init__(self, model_path: str, db_path: str):
        self.model_path = model_path
        self.db_path = db_path
        self.model = None
        self.feature_model = None
        self.database: dict = {}
        self._load_model()
        self._load_database()

    # ── Model Loading ────────────────────────────────────────────────────────
    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Model not found at {self.model_path}. Predictions will fail.")
            return
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras import models as keras_models

            print(f"[INFO] Loading model from {self.model_path} ...")
            base_model = load_model(self.model_path)

            # Extract feature layer — same as Colab
            self.feature_model = keras_models.Model(
                inputs=base_model.inputs,
                outputs=base_model.get_layer('biometric_feature_layer').output
            )
            self.model = base_model
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")

    # ── Database Loading ─────────────────────────────────────────────────────
    def _load_database(self):
        if not os.path.exists(self.db_path):
            print("[INFO] No database found. Starting with empty database.")
            self.database = {}
            return
        try:
            with open(self.db_path, "rb") as f:
                self.database = pickle.load(f)
            print(f"[INFO] Database loaded: {len(self.database)} subjects enrolled.")
        except Exception as e:
            print(f"[ERROR] Database load failed: {e}")
            self.database = {}

    def save_database(self):
        """Persist the in-memory database back to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)
        return len(self.database)

    # ── GEI Pipeline (exact copy from Colab) ────────────────────────────────
    def video_to_gei(self, video_path: str, num_frames: int = 60):
        """
        Convert a walking video into a Gait Energy Image (GEI).
        Returns a (1, 128, 128, 3) numpy array, or the string 'NOT_GAIT'.
        """
        cap = cv2.VideoCapture(video_path)
        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False
        )
        silhouettes = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= num_frames:
                break

            fgmask = fgbg.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 800:
                    x, y, w, h = cv2.boundingRect(largest)
                    if (h / float(w)) > 1.2:
                        human = cv2.resize(fgmask[y:y+h, x:x+w], (128, 128))
                        silhouettes.append(human)
            frame_count += 1

        cap.release()

        if len(silhouettes) < 10:
            return "NOT_GAIT"

        gei = np.mean(silhouettes, axis=0) / 255.0
        gei_3ch = np.repeat(gei[:, :, np.newaxis], 3, axis=-1)
        return gei_3ch.reshape(1, 128, 128, 3)

    def extract_features(self, video_path: str):
        """Extract deep biometric feature vector from a video."""
        gei = self.video_to_gei(video_path)
        if isinstance(gei, str) and gei == "NOT_GAIT":
            return "NOT_GAIT"
        if gei is None:
            return None
        if self.feature_model is None:
            raise RuntimeError("Model is not loaded.")
        return self.feature_model.predict(gei, verbose=0)

    # ── Register ─────────────────────────────────────────────────────────────
    def register(self, video_path: str, subject_name: str) -> dict:
        feat = self.extract_features(video_path)

        if isinstance(feat, str) and feat == "NOT_GAIT":
            return {
                "status": "error",
                "message": "SECURITY REJECT: Subject is not walking. Facial/Static media denied."
            }
        if feat is None:
            return {"status": "error", "message": "Failed to read video file."}

        if subject_name not in self.database:
            self.database[subject_name] = []
        self.database[subject_name].append(feat)

        return {
            "status": "success",
            "message": f"Successfully enrolled '{subject_name}'. "
                       f"(Total profiles: {len(self.database[subject_name])})"
        }

    # ── Recognize ────────────────────────────────────────────────────────────
    def recognize(self, video_path: str) -> dict:
        if not self.database:
            return {"status": "error", "message": "Database is empty. Please enroll someone first."}

        feat = self.extract_features(video_path)

        if isinstance(feat, str) and feat == "NOT_GAIT":
            return {
                "status": "error",
                "message": "SECURITY REJECT: Subject is not walking. Facial/Static media denied."
            }
        if feat is None:
            return {"status": "error", "message": "Failed to read video file."}

        best_match, best_sim = None, -1.0
        for db_name, saved_feats in self.database.items():
            for s_feat in saved_feats:
                sim = np.dot(feat.flatten(), s_feat.flatten()) / (
                    np.linalg.norm(feat) * np.linalg.norm(s_feat)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_match = db_name

        conf = max(0.0, min(100.0, float(best_sim) * 100))

        if conf >= 80.0:
            return {
                "status": "success",
                "match": best_match,
                "confidence": round(conf, 2)
            }
        else:
            return {
                "status": "error",
                "message": f"ACCESS DENIED: Unknown Biometric Signature. (Confidence: {round(conf, 2)}%)"
            }

    # ── Utility ──────────────────────────────────────────────────────────────
    def db_status(self) -> dict:
        return {
            "enrolled": len(self.database),
            "subjects": list(self.database.keys())
        }
