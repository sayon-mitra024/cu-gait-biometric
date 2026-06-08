---
title: CU Gait Biometric
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---
# CU Gait Biometric System — Render Deployment Guide

## Project Structure

```
gait-deploy/
├── app.py                          ← Flask server (all API routes)
├── model_utils/
│   ├── __init__.py
│   └── predictor.py                ← GEI pipeline + model logic (mirrors Colab)
├── templates/
│   └── index.html                  ← Full web UI (Register + Authenticate)
├── model/
│   ├── Chandigarh_University_Gait_Biometric_Prototype.keras  ← PUT YOUR MODEL HERE
│   └── CU_gait_database.pkl        ← PUT YOUR DATABASE HERE (optional)
├── uploads/                        ← Temp video storage (auto-created)
├── requirements.txt
├── Procfile
└── render.yaml
```

---

## Step 1 — Download Your Model Files from Google Drive

1. Open your Google Drive folder:
   `https://drive.google.com/drive/folders/1ZQ8qRxKKBl92EN2znmdDYqJfl4lD12Ca`
2. Download these two files:
   - `Chandigarh_University_Gait_Biometric_Prototype.keras`
   - `CU_gait_database.pkl`
3. Place **both files** inside the `model/` folder of this project.

---

## Step 2 — Push to GitHub

```bash
# Inside the gait-deploy folder:
git init
git add .
git commit -m "Initial CU Gait Biometric deployment"

# Create a NEW repo on github.com (name: cu-gait-biometric)
# Then:
git remote add origin https://github.com/YOUR_USERNAME/cu-gait-biometric.git
git branch -M main
git push -u origin main
```

> ⚠️ If your `.keras` model file is larger than 100 MB, you need Git LFS:
> ```bash
> git lfs install
> git lfs track "model/*.keras"
> git lfs track "model/*.pkl"
> git add .gitattributes
> git add .
> git commit -m "Add model via LFS"
> git push
> ```

---

## Step 3 — Deploy on Render

1. Go to **https://render.com** and sign up / log in (free).
2. Click **"New +"** → **"Web Service"**.
3. Connect your **GitHub** account and select the `cu-gait-biometric` repo.
4. Fill in the settings:

   | Setting | Value |
   |---------|-------|
   | **Name** | `cu-gait-biometric` |
   | **Region** | Singapore (closest to India) |
   | **Branch** | `main` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 2` |
   | **Plan** | Free |

5. Click **"Create Web Service"**.
6. Wait ~5–10 minutes for the build to finish.
7. Your live URL will be: `https://cu-gait-biometric.onrender.com`

---

## Step 4 — Verify Deployment

Visit your URL and check:
- `https://cu-gait-biometric.onrender.com/` → Web UI loads
- `https://cu-gait-biometric.onrender.com/health` → Returns `{"model_loaded": true}`
- `https://cu-gait-biometric.onrender.com/api/status` → Shows enrolled subjects

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/api/status` | Model + DB status |
| POST | `/api/register` | Enroll a subject (form: `video` + `subject_name`) |
| POST | `/api/recognize` | Authenticate (form: `video`) |
| POST | `/api/save_database` | Persist DB to disk |
| GET | `/api/subjects` | List enrolled subjects |

---

## How the Pipeline Works (Same as Colab)

```
Video (.mp4/.avi)
    │
    ▼
Background Subtraction (MOG2)
    │
    ▼
Silhouette Extraction (largest human contour, h/w > 1.2)
    │
    ▼
Resize each silhouette → 128×128
    │
    ▼
Average all silhouettes → Gait Energy Image (GEI)
    │
    ▼
Normalize + Convert to 3-channel (128, 128, 3)
    │
    ▼
feature_model.predict()  ← biometric_feature_layer output
    │
    ▼
Cosine Similarity vs database entries
    │
    ▼
Score ≥ 80% → ACCESS GRANTED + identity
Score < 80% → ACCESS DENIED
```

---

## Important Notes

### Free Render Tier Limitations
- Server **sleeps after 15 min** of inactivity — first request takes ~30 seconds to wake up.
- **512 MB RAM** — TensorFlow CPU is large; if build fails, upgrade to Starter ($7/month).
- **No persistent disk** on free tier — `save_database` saves to memory only. To persist:
  - Upgrade to Render's paid tier (persistent disk), OR
  - Use Render's environment variable to pre-load a base database.

### Running Locally First (Recommended to Test)
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

### If Model File > 100 MB (LFS Issue)
Alternative: host the model on Google Drive and download it at startup.
Add this to the top of `app.py`:

```python
import gdown
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
```

Add `gdown` to `requirements.txt`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not found` | Check model files are in `model/` folder and pushed to GitHub |
| `Memory error` | Upgrade Render plan or use `tensorflow-cpu` (already set) |
| `Timeout` | Increase `--timeout` in Procfile (already 300s) |
| `NOT_GAIT error` | Video too short or subject not walking; use 3–10 sec video |
| `Build fails` | Check Python version is 3.11 in render.yaml |
