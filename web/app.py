"""SentinEL web UI — upload calibration/interview videos and view results."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
DATA = ROOT / "web_data"
JOBS = DATA / "jobs"

ALLOWED_EXT = {".mp4", ".webm", ".mkv", ".mov", ".wav", ".m4a"}
MAX_POLL_LOGS = 300

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    logs: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self, *, lite: bool = True) -> dict[str, Any]:
        """lite=True keeps poll payloads small (avoids browser Failed to fetch)."""
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "created_at": self.created_at,
            "log_count": len(self.logs),
        }
        if lite:
            payload["logs"] = self.logs[-MAX_POLL_LOGS:]
            if self.status == "done" and self.result is not None:
                payload["result_ready"] = True
                payload["result_summary"] = _result_summary(self.result, self.kind)
        else:
            payload["logs"] = self.logs
            payload["result"] = self.result
        return payload


def _result_summary(result: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "analyze":
        if result.get("summary") and not result.get("answers"):
            return {**result["summary"], "results_path": result.get("results_path")}
        answers = result.get("answers") or []
        alerts = sum(1 for a in answers if a.get("status") == "PROBABLE_SCRIPT_READING")
        return {
            "answers": len(answers),
            "alerts": alerts,
            "contrastive_engine": result.get("contrastive_engine"),
            "results_path": result.get("results_path"),
        }
    return {
        "profile_path": result.get("profile_path"),
        "calibration_windows": result.get("profile", {}).get("calibration_windows"),
    }


jobs: dict[str, Job] = {}
_jobs_lock = asyncio.Lock()

app = FastAPI(title="SentinEL", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.on_event("startup")
async def _warmup_models() -> None:
    """Preload Whisper in background so first calibration job is faster."""
    import config as cfg

    if not cfg.PRELOAD_MODELS_ON_STARTUP:
        return

    async def _load() -> None:
        import asyncio

        await asyncio.to_thread(_preload_sync)

    import asyncio

    asyncio.create_task(_load())


def _preload_sync() -> None:
    try:
        import config as cfg

        if (
            cfg.KAGGLE_OFFLOAD
            and cfg.KAGGLE_OFFLOAD_TRANSCRIPTION
            and cfg.SKIP_LOCAL_WHISPER_WHEN_KAGGLE
            and cfg.KAGGLE_GPU_URL
        ):
            logging.getLogger(__name__).info(
                "Skipping local Whisper preload — Kaggle GPU offload enabled"
            )
            return
        from processors.transcript_processor import preload_models

        preload_models(calibration_only=cfg.PRELOAD_CALIBRATION_MODEL_ONLY)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("Model preload skipped: %s", exc)


def _job_dir(job_id: str) -> Path:
    d = JOBS / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _check_ext(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
        )
    return ext


def _progress_cb(job: Job):
    def cb(percent: int, message: str, log_entry: dict[str, Any] | None) -> None:
        job.progress = percent
        job.message = message
        if log_entry:
            job.logs.append(log_entry)
            if len(job.logs) > MAX_POLL_LOGS * 4:
                job.logs = job.logs[-MAX_POLL_LOGS * 2 :]

    return cb


async def _run_calibrate_job(job: Job, video_path: Path) -> None:
    job.status = "running"
    out_profile = _job_dir(job.id) / "calibration_profile.json"

    def _work() -> dict[str, Any]:
        from services.pipeline import run_calibrate

        return run_calibrate(
            video_path,
            output_path=out_profile,
            progress=_progress_cb(job),
        )

    try:
        result = await asyncio.to_thread(_work)
        job.result = {
            "profile": result["profile"],
            "profile_path": str(out_profile),
            "elapsed_sec": result["elapsed_sec"],
        }
        job.status = "done"
        job.progress = 100
        job.message = "Calibration complete"
    except Exception as exc:
        logger.error("Calibrate job failed:\n%s", traceback.format_exc())
        job.status = "error"
        job.error = str(exc)
        job.message = f"Failed: {exc}"


async def _run_analyze_job(
    job: Job,
    video_path: Path,
    calibration: dict[str, Any],
) -> None:
    job.status = "running"
    out_results = _job_dir(job.id) / "results.json"

    def _work() -> dict[str, Any]:
        from services.pipeline import run_analyze

        return run_analyze(
            video_path,
            calibration,
            output_path=out_results,
            progress=_progress_cb(job),
        )

    try:
        result = await asyncio.to_thread(_work)
        answers = result.get("answers") or []
        alerts = sum(1 for a in answers if a.get("status") == "PROBABLE_SCRIPT_READING")
        job.result = {
            "results_path": str(out_results),
            "summary": {
                "answers": len(answers),
                "alerts": alerts,
                "contrastive_engine": result.get("contrastive_engine"),
                "elapsed_sec": result.get("elapsed_sec"),
            },
            "_full_results_file": str(out_results),
        }
        job.status = "done"
        job.progress = 100
        job.message = f"Analysis complete ({len(answers)} answers, {alerts} alerts)"
    except Exception as exc:
        logger.error("Analyze job failed:\n%s", traceback.format_exc())
        job.status = "error"
        job.error = str(exc)
        job.message = f"Failed: {exc}"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


def _check_dependencies() -> str | None:
    """Return error message if ML stack is missing in this Python process."""
    try:
        import whisperx  # noqa: F401
        return None
    except ImportError as exc:
        return (
            f"Missing dependency: {exc}. "
            "Stop the server and start with .\\run_web.ps1 (not plain 'python -m uvicorn'). "
            "If that fails, run: pip install torch torchaudio --index-url "
            "https://download.pytorch.org/whl/cpu && pip install whisperx && pip install -r requirements.txt"
        )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    dep_err = _check_dependencies()
    return {
        "status": "ok" if not dep_err else "missing_dependencies",
        "python_deps_ok": dep_err is None,
        "message": dep_err or "Ready",
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, full: bool = False) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.to_dict(lite=not full)


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict[str, Any]:
    """Full analysis/calibration payload (loaded from disk for analyze jobs)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, "Job not finished yet")

    if job.kind == "analyze":
        path = (job.result or {}).get("results_path") or (job.result or {}).get(
            "_full_results_file"
        )
        if path and Path(path).is_file():
            return json.loads(Path(path).read_text(encoding="utf-8"))
        if job.result and job.result.get("answers"):
            return job.result
        raise HTTPException(404, "Results file not found")

    if job.result:
        return job.result
    raise HTTPException(404, "No result available")


@app.post("/api/calibrate")
async def api_calibrate(video: UploadFile = File(...)) -> dict[str, str]:
    dep_err = _check_dependencies()
    if dep_err:
        raise HTTPException(503, dep_err)

    job_id = str(uuid.uuid4())
    ext = _check_ext(video.filename or "video.mp4")
    dest = _job_dir(job_id) / f"calibration{ext}"
    _save_upload(video, dest)

    job = Job(id=job_id, kind="calibrate")
    async with _jobs_lock:
        jobs[job_id] = job

    asyncio.create_task(_run_calibrate_job(job, dest))
    return {"job_id": job_id}


@app.post("/api/analyze")
async def api_analyze(
    interview: UploadFile = File(...),
    calibration_job_id: str | None = Form(None),
    calibration_file: UploadFile | None = File(None),
) -> dict[str, str]:
    dep_err = _check_dependencies()
    if dep_err:
        raise HTTPException(503, dep_err)

    if not calibration_job_id and not calibration_file:
        raise HTTPException(
            400,
            "Provide calibration_job_id (from a calibrate run) or upload calibration_file (.json)",
        )

    profile: dict[str, Any] | None = None
    if calibration_job_id:
        cal_job = jobs.get(calibration_job_id)
        if not cal_job or cal_job.kind != "calibrate":
            raise HTTPException(400, "Invalid calibration_job_id")
        if cal_job.status != "done" or not cal_job.result:
            raise HTTPException(400, "Calibration job not finished yet")
        profile = cal_job.result.get("profile")
        if not profile:
            path = cal_job.result.get("profile_path")
            if path and Path(path).is_file():
                profile = json.loads(Path(path).read_text(encoding="utf-8"))
        if not profile:
            raise HTTPException(400, "Calibration job has no profile")

    job_id = str(uuid.uuid4())
    ext = _check_ext(interview.filename or "interview.mp4")
    interview_dest = _job_dir(job_id) / f"interview{ext}"
    _save_upload(interview, interview_dest)

    if calibration_file and calibration_file.filename:
        if not profile:
            cal_ext = Path(calibration_file.filename).suffix.lower()
            if cal_ext != ".json":
                raise HTTPException(400, "Calibration file must be .json")
            cal_dest = _job_dir(job_id) / "calibration_profile.json"
            _save_upload(calibration_file, cal_dest)
            profile = json.loads(cal_dest.read_text(encoding="utf-8"))

    if not profile:
        raise HTTPException(400, "Could not load calibration profile")

    job = Job(id=job_id, kind="analyze")
    async with _jobs_lock:
        jobs[job_id] = job

    asyncio.create_task(_run_analyze_job(job, interview_dest, profile))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}/export")
async def export_results(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or job.status != "done" or not job.result:
        raise HTTPException(404, "Results not ready")
    path = _job_dir(job_id) / "results.json"
    if not path.is_file():
        path = _job_dir(job_id) / "export.json"
        path.write_text(json.dumps(job.result, indent=2), encoding="utf-8")
    return FileResponse(path, media_type="application/json", filename="results.json")


def main() -> None:
    import uvicorn

    DATA.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
