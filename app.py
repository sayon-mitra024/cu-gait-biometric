"""
app.py  —  CU Gait Biometric System  |  Flask Backend
Deploy on Render.com (free tier or paid).
"""

import os
import uuid
import traceback
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from model_utils.predictor import GaitPredictor

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024   # 200 MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model + database once at startup ───────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model",
                 "Chandigarh_University_Gait_Biometric_Prototype.keras")
)
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(__file__), "model", "CU_gait_database.pkl")
)

predictor = GaitPredictor(MODEL_PATH, DB_PATH)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file) -> str:
    """Save uploaded file and return its path."""
    ext = file.filename.rsplit(".", 1)[1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(fname))
    file.save(path)
    return path


# ── Pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── API ──────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    db = predictor.db_status()
    return jsonify({
        "model_loaded": predictor.model is not None,
        "enrolled_subjects": db["enrolled"],
        "subjects": db["subjects"]
    })


@app.route("/api/register", methods=["POST"])
def api_register():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided."}), 400

    file = request.files["video"]
    subject_name = request.form.get("subject_name", "").strip()

    if not subject_name:
        return jsonify({"status": "error", "message": "Subject name is required."}), 400
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Unsupported file type. Use MP4, AVI, or MOV."}), 400

    path = save_upload(file)
    try:
        result = predictor.register(path, subject_name)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided."}), 400

    file = request.files["video"]
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "Unsupported file type. Use MP4, AVI, or MOV."}), 400

    path = save_upload(file)
    try:
        result = predictor.recognize(path)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route("/api/save_database", methods=["POST"])
def api_save_database():
    try:
        count = predictor.save_database()
        return jsonify({
            "status": "success",
            "message": f"Database saved. {count} subject(s) persisted to disk."
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Save failed: {str(e)}"}), 500


@app.route("/api/subjects")
def api_subjects():
    db = predictor.db_status()
    return jsonify(db)


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
