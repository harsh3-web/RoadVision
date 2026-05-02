# Road Damage Detector — Full-Stack App

YOLOv8 + ByteTrack model that detects and counts **potholes, cracks, and manholes**
in road videos, served as a full-stack web app.

- **Backend:** FastAPI (`backend/`) — loads `best.pt` once at startup, runs `model.track()` on uploaded videos, returns per-class unique counts and a playable annotated MP4.
- **Frontend:** React + Vite (`frontend/`) — drag-and-drop upload, live progress, video playback, downloadable result.
- **Deployment:** Backend on Render, frontend on Vercel — both auto-deploy on `git push`.

```
.
├── backend/
│   ├── main.py            # FastAPI app + lifespan model load
│   ├── api/routes.py      # /predict, /health, /results/{id}/video
│   ├── services/detector.py  # YOLO + ByteTrack wrapper
│   ├── schemas/           # Pydantic models
│   ├── model/             # place best.pt here (gitignored)
│   ├── config.py
│   ├── requirements.txt
│   └── runtime.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── components/
│   │   └── styles.css
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── vercel.json
├── render.yaml            # Render blueprint (backend)
└── .gitignore
```

---

## 1. Local development

### 1a. Backend

```bash
cd backend

# 1. Drop your trained weights here
#    (the BTP model from /content/drive/MyDrive/BTP/runs/final_model_safe_v2/weights/best.pt)
cp /path/to/best.pt model/best.pt

# 2. Install Python deps
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional) install ffmpeg for in-browser-playable MP4 output
#    macOS:  brew install ffmpeg
#    Ubuntu: sudo apt-get install ffmpeg
#    Windows: choco install ffmpeg

# 4. Configure
cp .env.example .env

# 5. Run
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

Open <http://localhost:10000/docs> to try the Swagger UI.

### 1b. Frontend

```bash
cd frontend
cp .env.example .env           # VITE_BACKEND_URL=http://localhost:10000
npm install
npm run dev                    # http://localhost:5173
```

---

## 2. API

| Method | Path                          | Purpose                                       |
|-------:|-------------------------------|-----------------------------------------------|
| GET    | `/health`                     | Liveness + model-loaded flag + class names    |
| POST   | `/predict`                    | `multipart/form-data` field `file` = video    |
| GET    | `/results/{job_id}/video`     | Streams the annotated MP4                     |
| DELETE | `/results/{job_id}`           | Removes the cached result from server         |

`POST /predict` response:

```json
{
  "job_id": "9f...",
  "video_url": "http://localhost:10000/results/9f.../video",
  "counts": [
    { "class_name": "pothole", "count": 12 },
    { "class_name": "crack",   "count": 4  },
    { "class_name": "manhole", "count": 1  }
  ],
  "total_unique_objects": 17,
  "frames_processed": 540,
  "inference_seconds": 23.4
}
```

---

## 3. Deploy backend → Render

1. Push this repo to GitHub.
2. In Render, click **New → Blueprint**, point it at the repo. It will pick up `render.yaml`.
3. Render auto-installs `ffmpeg` + system libs in the build command.
4. **Upload `best.pt`** so Render can find it. Two options:
   - **Easiest:** create a Render Disk (1 GB), mount it at `/var/data`, set `MODEL_PATH=/var/data/best.pt`, then `scp` the file in.
   - **Or:** host `best.pt` somewhere (S3, GitHub Release, Drive) and add a `curl -L -o model/best.pt <url>` line at the top of `buildCommand`. Set `MODEL_PATH=model/best.pt`.
5. Hit **Deploy**. Once green, note the public URL — e.g. `https://pothole-detection-api.onrender.com`.

> The free tier sleeps after 15 min of inactivity and has only CPU — first request after sleep takes ~30 s. Use the Standard tier (or higher) for real demos.

Auto-deploy: every push to the connected branch redeploys (`autoDeploy: true` in `render.yaml`).

---

## 4. Deploy frontend → Vercel

1. In Vercel, **Add New → Project**, import the same GitHub repo.
2. Set **Root Directory** to `frontend`. Framework auto-detects as Vite (Vercel reads `vercel.json`).
3. **Environment Variables** → add:
   - `VITE_BACKEND_URL` = `https://pothole-detection-api.onrender.com` (from step 3 above, no trailing slash)
4. Deploy. Every push to main triggers a fresh deploy.

---

## 5. Connecting frontend ↔ backend

- The frontend reads `VITE_BACKEND_URL` at build time (Vite inlines `import.meta.env.*`).
- The backend's CORS list is controlled by `CORS_ORIGINS` (default `*`). For production, set it to your Vercel URL, e.g. `https://your-app.vercel.app`.
- The annotated MP4 is served from the backend, so the `<video>` tag in the UI hits Render directly using the URL returned by `/predict`.

---

## 6. Adapting the original Colab notebook

The notebook (`Untitled0.ipynb`) does training, evaluation, and tracking. The web app
only does **inference** (Section 12-15 of the notebook). What was reorganised:

| Notebook                             | App                                            |
|--------------------------------------|------------------------------------------------|
| `YOLO("/content/best.pt")`           | Loaded once in `services/detector.py` lifespan |
| `model.track(source=...)`            | `Detector.predict_video(...)`                  |
| `unique_ids` + `defaultdict(set)`    | Same — returned as `counts` in response        |
| `ffmpeg ... libx264`                 | `Detector._transcode_to_mp4`                   |
| `files.upload()` / `files.download()`| React drag-and-drop + `<a download>`           |

Training cells were dropped — production never re-trains during a request.

---

## 7. Troubleshooting

- **`Model failed to load`** in logs → `best.pt` missing. Check `MODEL_PATH` and the Render disk.
- **Browser can't play video** → ffmpeg wasn't installed; `render.yaml` installs it but if you deploy elsewhere add it manually.
- **CORS error** → set `CORS_ORIGINS` to your Vercel domain.
- **`413 File too large`** → bump `MAX_UPLOAD_SIZE` (bytes).
- **Cold-start timeout on free Render** → upgrade plan or warm with a cron pinger.
