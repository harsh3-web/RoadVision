# Pothole Detection — Architecture & Flow

End-to-end map of how a video upload becomes an annotated result on the screen.

---

## 1. The three services

```
            ┌─────────────────────────┐
            │   Browser (Vercel)      │
            │   React + Vite frontend │
            │   • UploadForm          │
            │   • ProgressCard (ETA)  │
            │   • ResultDisplay       │
            └───────────┬─────────────┘
                        │ HTTPS
                        ▼
            ┌─────────────────────────┐
            │  Render (free CPU)      │
            │  FastAPI backend        │
            │  • Receives upload      │
            │  • Stores job state     │
            │  • Forwards to GPU      │
            │  • Polls GPU progress   │
            │  • Serves result video  │
            └───────────┬─────────────┘
                        │ HTTPS via ngrok static domain
                        ▼
            ┌─────────────────────────┐
            │  Colab notebook (T4)    │
            │  FastAPI + uvicorn      │
            │  • YOLO + ByteTrack     │
            │  • Runs on GPU          │
            │  • Tracks progress dict │
            │  • Returns annotated mp4│
            └─────────────────────────┘

            (separate process)
            ┌─────────────────────────┐
            │  scripts/keepalive.mjs  │
            │  Pings Render /health   │
            │  every 180s so the free │
            │  tier doesn't sleep     │
            └─────────────────────────┘
```

---

## 2. Full request lifecycle

```
USER picks video and clicks Upload
        │
        ▼
[1] FRONTEND  POST /predict  (multipart, video file)
        │      api.js → startPrediction()
        ▼
[2] RENDER  api/routes.py @predict
        │   • generates job_id
        │   • streams file to disk (storage/uploads/<job_id>.mp4)
        │   • count_video_frames(upload)  ← cv2 reads metadata for total_frames
        │   • jobs.create(job_id, total_frames)   status=processing
        │   • spawns thread → _run_job(job_id, upload_path)
        │   • returns 202 { job_id, status_url, total_frames }
        ▼
[3] FRONTEND   App.jsx receives 202 → starts pollJob()
        │    (polls /jobs/<id> at 1.5s, 1.5s, 1.5s, then 4s, then 8s)
        │
        ▼
[4] RENDER  _run_job thread
        │   detector.predict_video(upload_path, job_id, progress_cb)
        │     ├─ spawns POLL THREAD ──────────────────┐
        │     │     polls Colab /progress/<job_id>     │
        │     │     every 2s then 5s                   │
        │     │     calls progress_cb(frames)          │
        │     │     → jobs.set_progress(job_id, N)     │
        │     │                                        │
        │     └─ POSTs file + job_id to Colab /predict │
        │                                              │
        ▼                                              │
[5] COLAB  predict() endpoint                          │
        │   • streams upload to /tmp/inference/<job_id>│
        │   • asyncio.to_thread(_run_inference)        │
        │     so /progress can be served concurrently  │
        │   • _run_inference loops over model.track()  │
        │     for each frame:                          │
        │       progress_state[job_id] = frames ───────┼──┐
        │   • transcodes annotated video to MP4        │  │
        │   • returns JSON {counts, frames, …}         │  │
        ▼                                              │  │
[6] COLAB  /progress/<job_id>  ◄───────────────────────┘  │
        │   reads progress_state[job_id]                  │
        │   returns {"frames_processed": N}               │
        │   (served on event loop while inference runs)   │
        ▼                                                 │
[7] RENDER  poll thread updates JobManager every 5s ◄─────┘
        │
        ▼
[8] FRONTEND   each poll tick reads /jobs/<id>
        │   sees frames_processed climb
        │   ProgressCard updates:
        │     • % bar (frames / total_frames)
        │     • ETA   ((total - frames) / fps)
        │     • Speed (frames / elapsed)
        ▼
[9] RENDER  POST /predict to Colab returns
        │   • downloads /video/<remote_job_id> from Colab
        │   • saves to storage/results/<job_id>/annotated.mp4
        │   • final progress_cb(total_frames)
        │   • jobs.mark_done(job_id, result)
        ▼
[10] FRONTEND  next poll sees status="done"
        │      pollJob() returns final status
        │      ResultDisplay renders the video and counts
        │      <video src="<render>/results/<job_id>/video">
        ▼
[11] RENDER  GET /results/<job_id>/video
        │   FileResponse streams annotated.mp4 to browser
        ▼
[12] BROWSER  user watches the annotated video,
              optionally clicks Download
```

---

## 3. File-by-file responsibilities

### Frontend (`frontend/src/`)
| File | Role |
|---|---|
| `main.jsx` | React entrypoint |
| `App.jsx` | Top-level state machine: idle / uploading / processing / done. Wires upload → poll → result. |
| `api.js` | `startPrediction`, `fetchJob`, `pollJob` (with backoff: 1.5s × 3 → 4s × 7 → 8s), `checkHealth` |
| `components/UploadForm.jsx` | File picker + submit |
| `components/ProgressCard.jsx` | Live %, frame count, ETA, fps |
| `components/ResultDisplay.jsx` | Annotated video + class counts + download button |
| `components/StatusBar.jsx` | Backend health indicator |

### Render backend (`backend/`)
| File | Role |
|---|---|
| `main.py` | FastAPI app, CORS, request logging, calls `detector.load()` on startup |
| `config.py` | Pydantic settings: `GPU_INFERENCE_URL`, `CORS_ORIGINS`, `CLASS_NAMES`, paths |
| `api/routes.py` | `/health`, `/predict`, `/jobs/{id}`, `/results/{id}/video`, `/results/{id}/download`, DELETE `/results/{id}` |
| `schemas/prediction.py` | Pydantic response models (`JobAccepted`, `JobStatus`, `PredictionResult`) |
| `services/jobs.py` | Thread-safe in-memory `JobManager` — tracks status, progress, ETA |
| `services/detector.py` | `RemoteDetector` — forwards uploads to Colab, side-thread polls `/progress` every 5s |

### Colab GPU server (`colab/gpu_server.py`)
| Cell | Role |
|---|---|
| Cell 1 | `pip install` deps |
| Cell 2 | `curl` model weights from your GCS URL |
| Cell 3 | FastAPI app: `/health`, `/predict`, `/progress/{id}`, `/video/{id}`. Inference runs in `asyncio.to_thread` so the event loop stays responsive. `progress_state` dict updated per frame. |
| Cell 4 | Starts ngrok tunnel + uvicorn server, prints public URL |

### Ops
| File | Role |
|---|---|
| `render.yaml` | Render service config — Python 3.11.9, env vars (`GPU_INFERENCE_URL`, `CORS_ORIGINS`), apt packages (ffmpeg, libgl1) |
| `frontend/vercel.json` | Vercel config for the React app |
| `scripts/keepalive.mjs` | Node script — pings `/health` every 180s to prevent Render free-tier sleep |

---

## 4. Concurrency model — why nothing blocks

```
RENDER (single uvicorn worker)
  ┌───────────────────────────────────────────────────────────┐
  │ event loop                                                │
  │   • serves all incoming HTTP requests (uploads, polls)    │
  │ thread #1: _run_job(job_id_A)                             │
  │   ├─ thread #1a: poll Colab /progress every 5s            │
  │   └─ blocks on requests.post → Colab /predict             │
  │ thread #2: _run_job(job_id_B)  (if a 2nd user uploads)    │
  └───────────────────────────────────────────────────────────┘

COLAB (single uvicorn worker)
  ┌───────────────────────────────────────────────────────────┐
  │ event loop                                                │
  │   • serves /health, /progress, /video                     │
  │   • upload streaming for /predict                         │
  │ threadpool slot: _run_inference (the YOLO + tracker loop) │
  │   • writes progress_state[job_id] = N each frame          │
  │   • does NOT block the event loop                         │
  └───────────────────────────────────────────────────────────┘
```

The trick on Colab is `await asyncio.to_thread(_run_inference, …)` — without it, the inference loop would freeze the event loop and `/progress` requests would queue up until predict finished, which defeats the whole purpose.

---

## 5. Data persistence

Everything is **ephemeral** — fine for a demo, not for production:

| Where | What | Lifetime |
|---|---|---|
| Render disk `storage/uploads/` | Uploaded video | Deleted in `_run_job` finally block |
| Render disk `storage/results/<job_id>/` | Annotated MP4 | Until DELETE `/results/<job_id>` or container restart |
| Render memory `JobManager` | Job status dict | Lost on container restart |
| Colab `/tmp/inference/<job_id>/` | Source + annotated | Until Colab runtime resets (~12 hr) |
| Colab `progress_state` dict | Live frame counter | Cleared in `predict()` finally block |

---

## 6. Daily operation (10-day demo)

```
1. Open Colab notebook
2. Runtime → Run all  (cells 1 → 4 in order)
3. Verify the printed Public URL matches your static ngrok domain
4. (Optional) In a terminal:
     BACKEND_URL=https://your-render.onrender.com node scripts/keepalive.mjs
5. Open the Vercel frontend → upload a video → watch it work
6. When Colab disconnects (~90 min idle / ~12 hr max), re-run cells 3 & 4
   (the static ngrok domain stays the same so Render needs no changes)
```

---

## 7. Failure modes & where they show up

| Symptom | Likely cause | Where to look |
|---|---|---|
| Frontend says "Model not loaded" | Render started before Colab; restart Render or wait | Render logs |
| `MissingSchema` URL error | `GPU_INFERENCE_URL` missing `https://` prefix | Render env vars |
| `progress poll got HTTP 404` | Colab uvicorn is running old code without `/progress` | Restart cells 3 & 4 in Colab |
| Progress stuck at 0%, finishes correctly | Poll thread can't reach Colab; check ngrok is alive | Render logs for "progress poll error" |
| Colab disconnects mid-job | Free tier idle/runtime limit | Re-run cells 3 & 4 |
| Render slow to respond after idle | Free tier slept | `scripts/keepalive.mjs` prevents this |
