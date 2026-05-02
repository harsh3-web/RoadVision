"""Colab GPU inference server for the Pothole Detection backend.

Open a new Colab notebook (Runtime -> Change runtime type -> T4 GPU) and paste
each cell below into its own Colab cell, in order.

Required before you start:
- An ngrok account (free): https://dashboard.ngrok.com/get-started/your-authtoken
- A claimed free static domain at https://dashboard.ngrok.com/domains
  (e.g. pothole-yourname.ngrok-free.app). Using a static domain means your
  Render env var never changes between Colab sessions.
- The GCS URL of your model weights (same MODEL_WEIGHTS_URL Render uses).
"""

# %% [cell 1] install deps
# !pip install -q fastapi "uvicorn[standard]" python-multipart ultralytics opencv-python-headless pyngrok nest-asyncio


# %% [cell 2] download the model weights
# Replace MODEL_WEIGHTS_URL with your GCS URL (the same one your Render service uses).
# import os
# os.environ["MODEL_WEIGHTS_URL"] = "https://storage.googleapis.com/your-bucket/best.pt"
# !curl -fL --retry 3 -o best.pt "$MODEL_WEIGHTS_URL"
# !ls -lh best.pt


# %% [cell 3] FastAPI app
import asyncio
import shutil
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO

CLASS_NAMES = ["pothole", "crack", "manhole"]
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
TRACKER = "bytetrack.yaml"
MODEL_PATH = "best.pt"

WORK_DIR = Path("/tmp/inference")
WORK_DIR.mkdir(exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"
print("Device:", device, "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

model = YOLO(MODEL_PATH)
app = FastAPI(title="Pothole GPU Inference")

# Updated by the inference loop, read by GET /progress.
progress_state: dict[str, int] = {}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "classes": CLASS_NAMES,
    }


@app.get("/progress/{job_id}")
def progress(job_id: str):
    return {"frames_processed": progress_state.get(job_id, 0)}


def _run_inference(src_path: Path, job_id: str, run_dir: Path) -> dict:
    started = time.perf_counter()
    progress_state[job_id] = 0

    results = model.track(
        source=str(src_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        tracker=TRACKER,
        persist=True,
        save=True,
        stream=True,
        device=device,
        project=str(WORK_DIR),
        name=job_id,
        exist_ok=True,
        verbose=False,
    )

    unique_ids = defaultdict(set)
    frames = 0
    for r in results:
        frames += 1
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for obj_id, cls in zip(ids, classes):
                unique_ids[int(cls)].add(int(obj_id))
        progress_state[job_id] = frames

    annotated = None
    for ext in (".mp4", ".avi"):
        for p in run_dir.glob(f"*{ext}"):
            if p.name != src_path.name:
                annotated = p
                break
        if annotated:
            break
    if annotated is None:
        raise HTTPException(500, "No annotated video produced")

    playable = run_dir / "annotated.mp4"
    if annotated.suffix == ".mp4":
        shutil.copyfile(annotated, playable)
    else:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(annotated),
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(playable),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    elapsed = time.perf_counter() - started
    counts = [
        {"class_name": CLASS_NAMES[c], "count": len(ids)}
        for c, ids in sorted(unique_ids.items())
    ]
    for n in CLASS_NAMES:
        if not any(c["class_name"] == n for c in counts):
            counts.append({"class_name": n, "count": 0})

    src_path.unlink(missing_ok=True)

    return {
        "job_id": job_id,
        "counts": counts,
        "total_unique_objects": sum(len(ids) for ids in unique_ids.values()),
        "frames_processed": frames,
        "inference_seconds": round(elapsed, 2),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), job_id: str = Form(default=None)):
    rid = job_id or uuid.uuid4().hex
    run_dir = WORK_DIR / rid
    run_dir.mkdir(exist_ok=True)
    src_path = run_dir / (file.filename or "video.mp4")

    with src_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    # Run the blocking inference loop in a threadpool so /progress can be
    # served concurrently on the event loop.
    try:
        return await asyncio.to_thread(_run_inference, src_path, rid, run_dir)
    finally:
        progress_state.pop(rid, None)


@app.get("/video/{job_id}")
def video(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(400, "Invalid job id")
    p = WORK_DIR / job_id / "annotated.mp4"
    if not p.exists():
        raise HTTPException(404, "Result not found")
    return FileResponse(p, media_type="video/mp4")


# %% [cell 4] launch ngrok tunnel + uvicorn
# Replace these two values with your own:
NGROK_AUTHTOKEN = "PASTE_YOUR_NGROK_TOKEN_HERE"
NGROK_STATIC_DOMAIN = "pothole-yourname.ngrok-free.app"  # claimed at dashboard.ngrok.com/domains

import asyncio

import nest_asyncio
import uvicorn
from pyngrok import ngrok

ngrok.set_auth_token(NGROK_AUTHTOKEN)
ngrok.kill()
public_url = ngrok.connect(8000, hostname=NGROK_STATIC_DOMAIN).public_url
print("Public URL:", public_url)
print("Set GPU_INFERENCE_URL on Render to this exact value, then restart the Render service.")

nest_asyncio.apply()
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)
asyncio.get_event_loop().run_until_complete(server.serve())
