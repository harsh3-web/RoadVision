"""Remote GPU inference client.

Forwards the upload to a Colab-hosted FastAPI server (exposed via ngrok) which
runs YOLO + ByteTrack on a GPU and returns the annotated MP4 + counts.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import requests

from config import settings

logger = logging.getLogger(__name__)


class DetectorNotReadyError(RuntimeError):
    pass


def count_video_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return max(0, total)


class RemoteDetector:
    def __init__(self, base_url: str) -> None:
        url = base_url.strip().rstrip("/")
        if url and not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        self._base_url = url

    def load(self) -> None:
        if not self._base_url:
            logger.warning("GPU_INFERENCE_URL is empty — predict calls will fail.")
            return
        try:
            r = requests.get(f"{self._base_url}/health", timeout=10)
            r.raise_for_status()
            logger.info("Connected to remote GPU at %s", self._base_url)
        except requests.RequestException as exc:
            logger.warning("Remote GPU not reachable yet at %s: %s", self._base_url, exc)

    @property
    def is_ready(self) -> bool:
        return bool(self._base_url)

    def _poll_progress(
        self,
        job_id: str,
        progress_cb: Optional[Callable[[int], None]],
        stop_event: threading.Event,
    ) -> None:
        if progress_cb is None:
            return
        url = f"{self._base_url}/progress/{job_id}"
        interval = 2.0
        while not stop_event.wait(interval):
            interval = 5.0
            try:
                r = requests.get(url, timeout=10)
                if not r.ok:
                    logger.warning("[job %s] progress poll got HTTP %d", job_id, r.status_code)
                    continue
                frames = int(r.json().get("frames_processed", 0))
                logger.info("[job %s] remote progress: %d frames", job_id, frames)
                if frames > 0:
                    progress_cb(frames)
            except (requests.RequestException, ValueError) as exc:
                logger.warning("[job %s] progress poll error: %s", job_id, exc)

    def predict_video(
        self,
        source_path: Path,
        job_id: str,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> dict:
        if not self._base_url:
            raise DetectorNotReadyError("GPU_INFERENCE_URL not configured.")

        run_dir = settings.RESULT_DIR / job_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[job %s] forwarding to remote GPU at %s", job_id, self._base_url)
        started = time.perf_counter()

        stop_event = threading.Event()
        poll_thread = threading.Thread(
            target=self._poll_progress,
            args=(job_id, progress_cb, stop_event),
            daemon=True,
        )
        poll_thread.start()

        try:
            with source_path.open("rb") as fh:
                files = {"file": (source_path.name, fh, "video/mp4")}
                data = {"job_id": job_id}
                resp = requests.post(
                    f"{self._base_url}/predict",
                    files=files,
                    data=data,
                    timeout=settings.GPU_INFERENCE_TIMEOUT,
                )
            resp.raise_for_status()
            meta = resp.json()
            remote_job_id = meta["job_id"]
        finally:
            stop_event.set()
            poll_thread.join(timeout=2)

        playable = run_dir / "annotated.mp4"
        with requests.get(
            f"{self._base_url}/video/{remote_job_id}",
            stream=True,
            timeout=settings.GPU_INFERENCE_TIMEOUT,
        ) as vr:
            vr.raise_for_status()
            with playable.open("wb") as out:
                for chunk in vr.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)

        if progress_cb is not None:
            progress_cb(meta.get("frames_processed", 0))

        elapsed = time.perf_counter() - started
        logger.info("[job %s] remote inference done in %.2fs", job_id, elapsed)

        return {
            "video_path": playable,
            "counts": meta["counts"],
            "total_unique_objects": meta["total_unique_objects"],
            "frames_processed": meta["frames_processed"],
            "inference_seconds": meta.get("inference_seconds", round(elapsed, 2)),
        }


detector = RemoteDetector((settings.GPU_INFERENCE_URL or "").strip())
