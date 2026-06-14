"""Shared calibrate/analyze pipeline for CLI and web UI."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import config
from engine.analysis_engine import AnalysisEngine
from engine.contrastive_engine import ContrastiveEngine
from engine.fused_scorer import FusedScorer
from engine.linguistic_analyzer import LinguisticAnalyzer
from engine.profile_memory import BehavioralProfile
from gpu_client import KaggleGPUClient
from processors.audio_processor import AudioProcessor
from processors.transcript_processor import (
    TranscriptProcessor,
    load_whisper_calibration_model,
    load_whisper_model,
    preload_models,
)
from scoring.baseline import load_baseline_profile, save_baseline_profile

ProgressCallback = Callable[[int, str, dict[str, Any] | None], None]


def _answer_technical_density(transcript: dict[str, Any]) -> float:
    text = str(transcript.get("transcript", "")).lower()
    if not text.strip():
        return 0.0
    from engine.feature_extraction import TECHNICAL_RE

    tokens = text.split()
    hits = len(TECHNICAL_RE.findall(text))
    return hits / max(len(tokens), 1)


@dataclass
class PipelineLogger:
    """Structured decision / metric log for UI and debugging."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def log(
        self,
        step: str,
        message: str,
        *,
        phase: str = "pipeline",
        metrics: dict[str, Any] | None = None,
        decision: str | None = None,
        level: str = "info",
    ) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "step": step,
            "level": level,
            "message": message,
            "metrics": metrics or {},
            "decision": decision,
        }
        self.entries.append(entry)
        return entry

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.entries)


def _tag_message(message: str, backend: str | None) -> str:
    """Prefix progress text with a visible runtime tag for the web UI."""
    if backend == "kaggle":
        return f"[Kaggle GPU] {message}"
    if backend == "local":
        return f"[Local CPU] {message}"
    if backend == "hybrid":
        return f"[Local + Kaggle] {message}"
    return message


def _emit(
    cb: ProgressCallback | None,
    percent: int,
    message: str,
    log_entry: dict[str, Any] | None = None,
    *,
    backend: str | None = None,
) -> None:
    if cb:
        cb(percent, _tag_message(message, backend), log_entry)


def _startup_models(
    log: PipelineLogger,
    *,
    calibration: bool = False,
) -> KaggleGPUClient:
    gpu_client = KaggleGPUClient(
        base_url=config.KAGGLE_GPU_URL,
        secret=config.SENTINEL_SECRET,
        timeout=config.KAGGLE_GPU_TIMEOUT_SEC,
    )
    use_kaggle_asr = (
        gpu_client.offload_active
        and config.SKIP_LOCAL_WHISPER_WHEN_KAGGLE
        and not calibration
    )
    if use_kaggle_asr:
        log.log(
            "startup",
            _tag_message(
                "Kaggle GPU offload active — skipping local Whisper load for interview",
                "kaggle",
            ),
            phase="system",
            metrics={"kaggle_url": config.KAGGLE_GPU_URL, "runtime": "kaggle"},
        )
        if config.KAGGLE_FALLBACK_LOCAL_ASR:
            log.log(
                "startup",
                f"Preloading local Whisper ({config.WHISPER_MODEL_SIZE}) as Kaggle ASR fallback",
                phase="system",
            )
            load_whisper_model()
        return gpu_client

    if calibration and config.FAST_CALIBRATION:
        log.log(
            "startup",
            (
                f"Loading models (calibration Whisper "
                f"{config.WHISPER_CALIBRATION_MODEL_SIZE}, interview {config.WHISPER_MODEL_SIZE})"
            ),
            phase="system",
        )
        load_whisper_calibration_model(skip_filler_check=True)
    else:
        log.log(
            "startup",
            f"Loading models (Whisper {config.WHISPER_MODEL_SIZE}, {config.WHISPER_COMPUTE_TYPE})",
            phase="system",
        )
        load_whisper_model()
    return gpu_client


def _kaggle_prefetch_answers(
    gpu_client: KaggleGPUClient,
    answers: list[dict[str, Any]],
    gpu_reading_profile: dict[str, Any] | None,
    log: PipelineLogger,
    *,
    transcript_proc: TranscriptProcessor | None = None,
    progress: ProgressCallback | None = None,
) -> list[tuple[dict[str, Any] | None, float | None]]:
    """Kaggle transcription with per-answer timeout and local Whisper fallback."""
    import concurrent.futures as cf

    n = len(answers)
    out: list[tuple[dict[str, Any] | None, float | None]] = [(None, None)] * n
    workers = min(config.KAGGLE_PARALLEL_ANSWERS, max(n, 1))
    timeout_sec = int(config.KAGGLE_TRANSCRIBE_TIMEOUT_SEC)
    fallback_local = bool(config.KAGGLE_FALLBACK_LOCAL_ASR and transcript_proc is not None)
    kaggle_ok = 0
    fallback_ok = 0
    failed = 0

    def _one(idx: int, answer: dict[str, Any]) -> tuple[int, dict[str, Any] | None, float | None]:
        duration = float(answer["end_sec"]) - float(answer["start_sec"])
        audio = answer.get("audio_bytes", b"")
        transcript, gpu = gpu_client.process_answer(
            audio,
            answer_id=int(answer["answer_id"]),
            duration_sec=duration,
            gpu_reading_profile=gpu_reading_profile,
        )
        return idx, transcript, gpu

    log.log(
        "kaggle_prefetch",
        _tag_message(
            f"Offloading {n} answer(s) to Kaggle GPU (workers={workers}, timeout={timeout_sec}s)",
            "kaggle",
        ),
        phase="analyze",
        metrics={
            "parallel_workers": workers,
            "has_gpu_profile": bool(gpu_reading_profile),
            "fallback_local_asr": fallback_local,
            "runtime": "kaggle",
        },
    )

    def _local_fallback(idx: int, answer: dict[str, Any]) -> dict[str, Any] | None:
        if not fallback_local or transcript_proc is None:
            return None
        rows = transcript_proc.transcribe_answers([answer])
        return rows[0] if rows else None

    def _store(
        idx: int,
        transcript: dict[str, Any] | None,
        gpu: float | None,
        *,
        source: str,
    ) -> None:
        nonlocal kaggle_ok, fallback_ok, failed
        out[idx] = (transcript, gpu)
        if transcript and str(transcript.get("transcript", "")).strip():
            if source == "kaggle":
                kaggle_ok += 1
            else:
                fallback_ok += 1
        else:
            failed += 1

    if workers <= 1:
        for i, answer in enumerate(answers):
            pct = 28 + int(20 * (i / max(n, 1)))
            _emit(
                progress,
                pct,
                f"Kaggle ASR answer {i + 1}/{n}...",
                backend="kaggle",
            )
            transcript: dict[str, Any] | None = None
            gpu_score: float | None = None
            try:
                with cf.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(_one, i, answer)
                    _idx, transcript, gpu_score = fut.result(timeout=timeout_sec)
            except cf.TimeoutError:
                log.log(
                    "kaggle_prefetch",
                    f"Kaggle ASR timeout on answer {i} after {timeout_sec}s",
                    phase="analyze",
                    level="warning",
                    metrics={"answer_id": answer.get("answer_id"), "timeout_sec": timeout_sec},
                )
            except Exception as exc:
                log.log(
                    "kaggle_prefetch",
                    f"Kaggle ASR error on answer {i}: {exc}",
                    phase="analyze",
                    level="warning",
                )
            if not transcript or not str(transcript.get("transcript", "")).strip():
                local_row = _local_fallback(i, answer)
                if local_row:
                    log.log(
                        "kaggle_prefetch",
                        f"Answer {i}: local Whisper fallback",
                        phase="analyze",
                        level="warning",
                        metrics={"answer_id": answer.get("answer_id")},
                        decision="local_asr_fallback",
                    )
                    _store(i, local_row, None, source="local")
                else:
                    _store(i, transcript, gpu_score, source="kaggle")
            else:
                _store(i, transcript, gpu_score, source="kaggle")
    else:
        completed = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, i, a): i for i, a in enumerate(answers)}
            for fut in futures:
                idx = futures[fut]
                try:
                    i, transcript, gpu = fut.result(timeout=timeout_sec)
                except cf.TimeoutError:
                    log.log(
                        "kaggle_prefetch",
                        f"Kaggle ASR timeout on answer {idx}",
                        phase="analyze",
                        level="warning",
                    )
                    local_row = _local_fallback(idx, answers[idx])
                    _store(idx, local_row, None, source="local")
                    completed += 1
                    continue
                except Exception as exc:
                    log.log(
                        "kaggle_prefetch",
                        f"Kaggle ASR error on answer {idx}: {exc}",
                        phase="analyze",
                        level="warning",
                    )
                    local_row = _local_fallback(idx, answers[idx])
                    _store(idx, local_row, None, source="local")
                    completed += 1
                    continue
                if not transcript or not str(transcript.get("transcript", "")).strip():
                    local_row = _local_fallback(i, answers[i])
                    _store(i, local_row or transcript, gpu, source="local" if local_row else "kaggle")
                else:
                    _store(i, transcript, gpu, source="kaggle")
                completed += 1
                _emit(
                    progress,
                    28 + int(20 * (completed / max(n, 1))),
                    f"Kaggle ASR {completed}/{n}...",
                    backend="kaggle",
                )

    log.log(
        "kaggle_prefetch",
        f"ASR complete — kaggle={kaggle_ok}, local_fallback={fallback_ok}, empty={failed}",
        phase="analyze",
        metrics={
            "kaggle_ok": kaggle_ok,
            "local_fallback": fallback_ok,
            "empty_or_failed": failed,
            "total": n,
        },
    )
    _emit(progress, 48, "Transcription complete", backend="kaggle" if kaggle_ok else "local")
    return out


def _gpu_score(
    gpu_client: KaggleGPUClient,
    audio_bytes: bytes,
    reading_profile: dict,
    duration_sec: float,
) -> float | None:
    result = gpu_client.analyze(audio_bytes, reading_profile, duration_sec)
    if result is None:
        return None
    if isinstance(result, dict):
        raw = result.get("score", result.get("gpu_score"))
        if raw is not None:
            return min(max(float(raw), 0.0), 1.0)
    return None


def run_calibrate(
    video_path: str | Path,
    *,
    output_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build calibration profile (SCRIPT reading fingerprint) from video."""
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    log = PipelineLogger()
    t0 = time.perf_counter()
    out_path = Path(output_path or "calibration_profile.json")

    _emit(progress, 2, "Starting calibration...", log.log("start", f"Calibration: {video.name}", phase="calibrate"))

    gpu_client = _startup_models(log, calibration=True)
    _emit(progress, 5, "Models loaded", None)

    audio = AudioProcessor()
    engine = AnalysisEngine()
    transcript_proc = TranscriptProcessor()
    linguistic_analyzer = LinguisticAnalyzer()

    fast = config.FAST_CALIBRATION
    video_fps = config.VIDEO_CALIBRATION_FPS if fast else config.VIDEO_TIMELINE_FPS

    _emit(
        progress,
        10,
        "Audio features"
        + (" (fast: no diarization)" if fast and config.SKIP_DIARIZATION_CALIBRATION else ""),
    )

    def _run_audio() -> list[dict[str, Any]]:
        return audio.process_calibration(str(video))

    def _run_video() -> dict[str, Any]:
        # Video gaze/lip scanning removed (performance + robustness).
        return {"timeline_path": "", "timeline": [], "native_fps": 0.0}

    t_audio = time.perf_counter()
    answers = _run_audio()
    video_result = _run_video()
    audio_sec = round(time.perf_counter() - t_audio, 2)

    log.log(
        "audio",
        f"Calibration audio: {len(answers)} answer segment(s) ({audio_sec}s)",
        phase="calibrate",
        metrics={
            "answers": len(answers),
            "fast_mode": fast,
            "skip_diarization": config.SKIP_DIARIZATION_CALIBRATION,
            "window_parallel_workers": config.CALIBRATION_WINDOW_PARALLEL_WORKERS,
            "elapsed_sec": audio_sec,
        },
    )
    gpu_reading_profile: dict | None = None
    if gpu_client.enabled and config.KAGGLE_OFFLOAD_CALIBRATION:
        _emit(progress, 12, "GPU calibration (Kaggle)...", None)
        t_gpu = time.perf_counter()
        best = max(
            answers,
            key=lambda a: len(a.get("audio_bytes") or b""),
            default=None,
        )
        if best and best.get("audio_bytes"):
            gpu_reading_profile = gpu_client.calibrate(best["audio_bytes"])
        gpu_sec = round(time.perf_counter() - t_gpu, 2)
        if gpu_reading_profile:
            log.log(
                "gpu_calibrate",
                f"GPU reading profile from Kaggle /calibrate ({gpu_sec}s)",
                phase="calibrate",
                metrics={"has_parselmouth": "parselmouth_baseline" in gpu_reading_profile},
            )
        else:
            log.log(
                "gpu_calibrate",
                "Kaggle GPU calibrate skipped or failed (local profile still valid)",
                phase="calibrate",
                level="warning",
                metrics={"elapsed_sec": gpu_sec},
            )
    elif gpu_client.enabled:
        log.log(
            "gpu_calibrate",
            "Kaggle /calibrate skipped (KAGGLE_OFFLOAD_CALIBRATION=false) — local CPU profile only",
            phase="calibrate",
        )

    windows = AudioProcessor.collect_windows(answers)
    t_acoustic = time.perf_counter()
    reading_profile = engine.calibrate(windows)
    acoustic_sec = round(time.perf_counter() - t_acoustic, 2)
    log.log(
        "acoustic_profile",
        "Built acoustic reading profile from openSMILE windows",
        phase="calibrate",
        metrics={"windows": len(windows), "profile_keys": len(reading_profile), "elapsed_sec": acoustic_sec},
    )
    _emit(progress, 35, f"Acoustic profile from {len(windows)} windows ({acoustic_sec}s)", None)

    _emit(
        progress,
        40,
        f"Transcribing calibration (Whisper {config.WHISPER_CALIBRATION_MODEL_SIZE if fast else config.WHISPER_MODEL_SIZE})...",
    )
    t_asr = time.perf_counter()
    cal_transcripts = transcript_proc.transcribe_answers(answers, calibration_fast=fast)
    asr_sec = round(time.perf_counter() - t_asr, 2)
    log.log(
        "asr",
        f"Calibration transcription ({asr_sec}s)",
        phase="calibrate",
        metrics={"backend": config.WHISPER_CALIBRATION_MODEL_SIZE if fast else config.WHISPER_MODEL_SIZE},
    )
    linguistic_calibration = linguistic_analyzer.calibrate(cal_transcripts)
    log.log(
        "linguistic",
        "Linguistic calibration complete",
        phase="calibrate",
        metrics={
            "transcripts": len(cal_transcripts),
            "asr_elapsed_sec": asr_sec,
            "skip_align": fast and config.WHISPER_SKIP_ALIGN_CALIBRATION,
        },
    )
    _emit(progress, 65, f"Transcription complete ({asr_sec}s)", None)

    timeline: list[dict[str, Any]] = video_result.get("timeline") or []
    log.log(
        "video",
        "Video scanning disabled (no gaze/lip timeline)",
        phase="calibrate",
        metrics={"frames": 0, "timeline_fps": 0},
    )
    _emit(progress, 85, "Building SCRIPT behavioral profile...")

    contrastive = ContrastiveEngine()
    script_profile, script_diag = contrastive.build_script_profile_from_calibration(
        answers,
        transcripts=cal_transcripts,
        timeline=timeline,
    )
    natural_seed = contrastive.build_natural_seed_from_calibration(
        answers,
        transcripts=cal_transcripts,
        timeline=timeline,
    )

    log.log(
        "script_profile_build",
        "SCRIPT profile feature extraction",
        phase="calibrate",
        metrics=script_diag,
    )

    if len(windows) < config.MIN_CALIBRATION_WINDOWS:
        msg = (
            f"Calibration produced only {len(windows)} acoustic windows "
            f"(minimum {config.MIN_CALIBRATION_WINDOWS}). "
            "Use a longer calibration clip with clear speech."
        )
        if config.CALIBRATION_FAIL_SOFT:
            log.log("script_profile", msg, phase="calibrate", level="warning")
        else:
            raise ValueError(msg)

    if script_profile.sample_count < config.MIN_SCRIPT_PROFILE_SAMPLES:
        msg = (
            f"SCRIPT behavioral profile has {script_profile.sample_count} samples "
            f"(minimum {config.MIN_SCRIPT_PROFILE_SAMPLES}). "
            f"Diagnostics: {script_diag}. "
            "Check calibration transcription and audio quality."
        )
        if config.CALIBRATION_FAIL_SOFT:
            log.log("script_profile", msg, phase="calibrate", level="warning")
            script_profile = contrastive.build_population_script_fallback(reading_profile)
            log.log(
                "script_profile",
                "Using population acoustic prior fallback for SCRIPT profile",
                phase="calibrate",
                level="warning",
                metrics={"samples": script_profile.sample_count},
            )
        else:
            raise ValueError(msg)

    log.log(
        "script_profile",
        "SCRIPT profile built from intentional reading windows",
        phase="calibrate",
        metrics={
            "samples": script_profile.sample_count,
            "natural_seed_samples": natural_seed.sample_count,
            **script_diag,
        },
        decision="This profile represents how the user sounds while reading.",
    )

    personal_baseline = None
    if config.ENABLE_INTRA_INDIVIDUAL:
        from engine.intra_individual import build_calibration_personal_baseline

        personal_baseline = build_calibration_personal_baseline(
            answers,
            transcripts=cal_transcripts,
            timeline=timeline,
        )
        log.log(
            "personal_baseline",
            "Personal speaking baseline seeded from calibration",
            phase="calibrate",
            metrics=personal_baseline.summary(),
        )

    profile = {
        "version": 5,
        "source_video": str(video),
        "timeline_path": video_result.get("timeline_path", ""),
        "acoustic_reading_profile": reading_profile,
        "gpu_reading_profile": gpu_reading_profile,
        "script_profile": script_profile.to_dict(),
        "natural_profile_seed": natural_seed.to_dict(),
        "script_build_diagnostics": script_diag,
        "linguistic_calibration": linguistic_calibration,
        "gaze_calibration": {},
        "lip_calibration": {},
        "calibration_answers": len(answers),
        "calibration_windows": len(windows),
        "timeline_frames": 0,
        "contrastive_engine": True,
        "personal_baseline": personal_baseline.to_dict() if personal_baseline else None,
        "intra_individual_modeling": config.ENABLE_INTRA_INDIVIDUAL,
    }
    save_baseline_profile(profile, out_path)

    elapsed = round(time.perf_counter() - t0, 2)
    log.log(
        "complete",
        f"Calibration saved ({elapsed}s)",
        phase="calibrate",
        metrics={"elapsed_sec": elapsed, "output": str(out_path)},
    )
    _emit(progress, 100, "Calibration complete", None)

    return {
        "profile": profile,
        "profile_path": str(out_path),
        "decision_log": log.to_list(),
        "elapsed_sec": elapsed,
    }


def run_analyze(
    video_path: str | Path,
    calibration: str | Path | dict[str, Any],
    *,
    output_path: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Analyze interview video against calibration profile."""
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    if isinstance(calibration, dict):
        profile = calibration
        cal_label = "uploaded profile"
    else:
        cal_path = Path(calibration)
        if not cal_path.is_file():
            raise FileNotFoundError(f"Calibration not found: {cal_path}")
        profile = load_baseline_profile(cal_path)
        cal_label = str(cal_path)

    log = PipelineLogger()
    t0 = time.perf_counter()
    out_path = Path(output_path) if output_path else None

    _emit(
        progress,
        2,
        "Starting analysis...",
        log.log("start", f"Interview: {video.name}", phase="analyze"),
    )

    reading_profile = profile.get("acoustic_reading_profile")
    if not reading_profile:
        raise ValueError("Calibration profile missing acoustic_reading_profile")
    gpu_reading_profile = profile.get("gpu_reading_profile")

    linguistic_calibration = profile.get("linguistic_calibration", {})

    gpu_client = _startup_models(log)
    use_kaggle_asr_plan = gpu_client.offload_active
    asr_plan = "Kaggle GPU" if use_kaggle_asr_plan else f"Local CPU ({config.WHISPER_MODEL_SIZE})"
    _emit(
        progress,
        5,
        f"Models loaded — transcription: {asr_plan}, diarization: Local CPU",
        None,
        backend="local",
    )

    audio = AudioProcessor()
    transcript_proc = TranscriptProcessor()
    engine = AnalysisEngine()
    linguistic_analyzer = LinguisticAnalyzer()
    scorer = FusedScorer()
    scorer.reset_ewma()

    contrastive = ContrastiveEngine()
    use_contrastive = config.ENABLE_CONTRASTIVE_ENGINE
    script_data = profile.get("script_profile")
    if script_data:
        contrastive.script_profile = BehavioralProfile.from_dict(script_data)
        log.log(
            "script_profile",
            "Loaded SCRIPT profile from calibration",
            phase="analyze",
            metrics={"samples": contrastive.script_profile.sample_count},
        )
    elif use_contrastive:
        use_contrastive = False
        log.log(
            "script_profile",
            "No script_profile in calibration — contrastive disabled",
            phase="analyze",
            level="warning",
            decision="Re-run calibrate for v5 profile.",
        )
    if use_contrastive:
        contrastive.reset_interview()
        seed_samples = contrastive.seed_natural_profile(profile.get("natural_profile_seed"))
        if seed_samples > 0:
            log.log(
                "natural_profile",
                f"NATURAL profile seeded from calibration voice anchor ({seed_samples} samples)",
                phase="analyze",
                metrics={"natural_profile_samples": seed_samples},
            )
        else:
            log.log(
                "natural_profile",
                "NATURAL profile not seeded — will learn from high-naturality interview windows",
                phase="analyze",
                level="warning",
                decision="Re-calibrate to store natural_profile_seed.",
            )

        if contrastive.script_profile.sample_count <= 0:
            contrastive.script_profile = contrastive.build_population_script_fallback(
                reading_profile
            )
            log.log(
                "script_profile",
                "SCRIPT profile empty — using population acoustic priors from reading profile",
                phase="analyze",
                level="warning",
                metrics={"samples": contrastive.script_profile.sample_count},
                decision="Re-run calibrate for a proper SCRIPT behavioral profile.",
            )

    from engine.cross_answer_content import SessionEvidenceAccumulator
    from engine.semantic_specificity import (
        apply_specificity_to_status,
        compute_semantic_specificity,
    )

    session_accumulator = SessionEvidenceAccumulator()
    null_video_channels = 0
    intra_session = None
    if config.ENABLE_INTRA_INDIVIDUAL:
        from engine.intra_individual import IntraIndividualSession

        intra_session = IntraIndividualSession.from_profile(profile)
        baseline_summary = intra_session.baseline.summary()
        if intra_session.baseline.is_unseeded():
            log.log(
                "personal_baseline",
                "Personal baseline empty — re-run calibrate to seed from calibration clip",
                phase="analyze",
                level="warning",
                metrics=baseline_summary,
                decision="Interview will bootstrap baseline from first answers (less reliable until seeded).",
            )
        else:
            log.log(
                "personal_baseline",
                "Intra-individual modeling active — deviation from personal baseline",
                phase="analyze",
                metrics=baseline_summary,
            )

    use_kaggle_segment = gpu_client.offload_segmentation_active
    if use_kaggle_segment:
            log.log(
                "kaggle_segment",
                _tag_message(
                    f"Kaggle segmentation ({config.KAGGLE_SEGMENT_MODE}) — local pyannote skipped",
                    "kaggle",
                ),
                phase="analyze",
            metrics={
                "kaggle_url": config.KAGGLE_GPU_URL,
                "segment_mode": config.KAGGLE_SEGMENT_MODE,
                "candidate_speaker": config.CANDIDATE_SPEAKER,
                "segment_timeout_sec": config.KAGGLE_SEGMENT_TIMEOUT_SEC,
                "skip_align": config.KAGGLE_SKIP_ALIGN_INTERVIEW,
                "transcribe_only": config.KAGGLE_TRANSCRIBE_ONLY,
            },
        )
    if use_kaggle_segment:
        av_msg = f"Segmentation ({config.KAGGLE_SEGMENT_MODE}) + local feature windows..."
        av_backend = "hybrid"
    else:
        av_msg = "Diarization + acoustic windows..."
        av_backend = "local"
    _emit(progress, 12, av_msg, backend=av_backend)

    segment_ctx: dict[str, Any] = {"payload": None, "wav_path": None}

    def _run_audio() -> list[dict[str, Any]]:
        if not use_kaggle_segment:
            log.log(
                "audio",
                _tag_message(
                    "Local pyannote diarization (recommended for AI vs candidate accuracy)",
                    "local",
                ),
                phase="analyze",
                metrics={
                    "candidate_speaker": config.CANDIDATE_SPEAKER,
                    "num_speakers": config.DIARIZATION_NUM_SPEAKERS,
                    "min_candidate_segment_sec": config.MIN_CANDIDATE_SEGMENT_SEC,
                },
            )
            return audio.process_interview(str(video))

        wav_path = audio.extract_audio(str(video))
        segment_ctx["wav_path"] = wav_path
        t_seg = time.perf_counter()
        _emit(
            progress,
            14,
            f"Kaggle GPU diarization ({config.KAGGLE_SEGMENT_MODE}) — uploading audio...",
            backend="kaggle",
        )
        segment_payload = gpu_client.segment_interview(str(video), wav_path=wav_path)
        segment_ctx["payload"] = segment_payload
        seg_sec = round(time.perf_counter() - t_seg, 2)
        answers_payload = (segment_payload or {}).get("answers") or []

        if not answers_payload:
            seg_err = (segment_payload or {}).get("error") if segment_payload else None
            if config.KAGGLE_SEGMENT_LOCAL_FALLBACK:
                log.log(
                    "kaggle_segment",
                    "Kaggle /segment_interview failed or empty — falling back to local pyannote",
                    phase="analyze",
                    level="warning",
                    metrics={
                        "elapsed_sec": seg_sec,
                        "kaggle_error": seg_err,
                        "kaggle_error_type": (segment_payload or {}).get("error_type"),
                    },
                    decision=str(seg_err)[:200] if seg_err else None,
                )
                segment_ctx["wav_path"] = None
                Path(wav_path).unlink(missing_ok=True)
                return audio.process_interview(str(video))
            log.log(
                "kaggle_segment",
                "Kaggle /segment_interview failed or empty — local fallback disabled",
                phase="analyze",
                level="error",
                metrics={
                    "elapsed_sec": seg_sec,
                    "kaggle_error": seg_err,
                    "kaggle_error_type": (segment_payload or {}).get("error_type"),
                },
                decision=str(seg_err)[:200] if seg_err else "empty_answers",
            )
            return []

        log.log(
            "kaggle_segment",
            f"Kaggle segment: {len(answers_payload)} answer(s) ({seg_sec}s)",
            phase="analyze",
            metrics={
                "answers": len(answers_payload),
                "elapsed_sec": seg_sec,
                "segmentation_backend": (segment_payload or {}).get(
                    "segmentation_backend"
                ),
                "speaker_selection": (segment_payload or {}).get("speaker_selection"),
                "alternate_answers": len(
                    (segment_payload or {}).get("alternate_answers") or []
                ),
            },
            decision=f"candidate_track={config.CANDIDATE_SPEAKER}",
        )
        return audio.process_interview_from_segmentation(
            str(video),
            answers_payload,
            speaker_selection=(segment_payload or {}).get("speaker_selection"),
            wav_path=wav_path,
        )

    def _run_video() -> dict[str, Any]:
        # Video gaze/lip scanning removed (performance + robustness).
        return {"timeline_path": "", "timeline": [], "native_fps": 0.0}

    if not use_kaggle_segment:
        _emit(
            progress,
            14,
            "Diarizing interview audio (pyannote — may take several minutes)...",
            backend="local",
        )

    t_av = time.perf_counter()
    interview_answers = _run_audio()
    video_result = _run_video()
    av_sec = round(time.perf_counter() - t_av, 2)
    seg_tag = "kaggle" if use_kaggle_segment else "local"
    _emit(
        progress,
        24,
        f"Audio ready: {len(interview_answers)} answer segment(s) ({av_sec}s)",
        backend=seg_tag,
    )

    speaker_sel = audio.last_speaker_selection or {}
    log.log(
        "audio",
        f"Interview: {len(interview_answers)} answer segment(s) ({av_sec}s)",
        phase="analyze",
        metrics={
            "answers": len(interview_answers),
            "candidate_speaker_strategy": config.CANDIDATE_SPEAKER,
            "speaker_selection": speaker_sel,
            "elapsed_sec": av_sec,
        },
        decision=f"candidate_track={speaker_sel.get('strategy', config.CANDIDATE_SPEAKER)}",
    )

    timeline: list[dict[str, Any]] = video_result.get("timeline") or []

    kaggle_cache: list[tuple[dict[str, Any] | None, float | None]] | None = None
    local_transcript_cache: list[dict[str, Any]] | None = None
    use_kaggle_asr = gpu_client.offload_active and bool(interview_answers)

    if use_kaggle_asr:
        _emit(
            progress,
            28,
            f"Transcription (parallel, {config.KAGGLE_PARALLEL_ANSWERS} workers)...",
            backend="kaggle",
        )
        kaggle_cache = _kaggle_prefetch_answers(
            gpu_client,
            interview_answers,
            gpu_reading_profile,
            log,
            transcript_proc=transcript_proc,
            progress=progress,
        )
    elif interview_answers:
        _emit(
            progress,
            28,
            f"Transcription (Whisper {config.WHISPER_MODEL_SIZE})...",
            backend="local",
        )
        t_asr = time.perf_counter()
        local_transcript_cache = transcript_proc.transcribe_answers(interview_answers)
        asr_sec = round(time.perf_counter() - t_asr, 2)
        log.log(
            "asr",
            _tag_message(f"Local interview transcription batch ({asr_sec}s)", "local"),
            phase="analyze",
            metrics={
                "answers": len(interview_answers),
                "elapsed_sec": asr_sec,
                "skip_align": config.WHISPER_SKIP_ALIGN_INTERVIEW,
                "model": config.WHISPER_MODEL_SIZE,
                "runtime": "local",
            },
        )

    if interview_answers and (kaggle_cache is not None or local_transcript_cache is not None):
        from engine.interviewer_segment_filter import (
            assess_kept_segment_quality,
            filter_segments_by_transcript,
            is_interviewer_transcript,
            is_off_topic_or_wrong_slice,
            needs_speaker_track_recovery,
        )

        def _transcripts_for_filter() -> list[dict[str, Any]]:
            if kaggle_cache is not None:
                return [
                    (
                        kaggle_cache[i][0]
                        if kaggle_cache[i][0] is not None
                        else {"transcript": ""}
                    )
                    for i in range(len(interview_answers))
                ]
            return local_transcript_cache or []

        def _apply_interviewer_filter() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            nonlocal interview_answers, kaggle_cache, local_transcript_cache
            transcripts_for_filter = _transcripts_for_filter()
            kept, excluded, keep_mask = filter_segments_by_transcript(
                interview_answers, transcripts_for_filter
            )
            interview_answers = kept
            if kaggle_cache is not None:
                kaggle_cache = [
                    row for row, keep in zip(kaggle_cache, keep_mask) if keep
                ]
            elif local_transcript_cache is not None:
                local_transcript_cache = [
                    row for row, keep in zip(local_transcript_cache, keep_mask) if keep
                ]
            return kept, excluded

        primary_total = len(interview_answers)
        kept, excluded = _apply_interviewer_filter()
        if excluded:
            log.log(
                "diarization_filter",
                f"Excluded {len(excluded)} interviewer segment(s) after ASR",
                phase="analyze",
                metrics={"excluded": excluded, "kept": len(kept)},
                decision="interviewer_prompt_removed",
            )

        if (
            config.DIARIZATION_SPEAKER_RECOVERY
            and needs_speaker_track_recovery(len(excluded), primary_total, len(kept))
        ):
            from engine.interviewer_segment_filter import infer_alternate_speaker

            payload = segment_ctx.get("payload") or {}
            speaker_sel = payload.get("speaker_selection") or {}
            alt_payload = payload.get("alternate_answers") or []
            wav_path = segment_ctx.get("wav_path")
            recovered = False
            recovery_max_asr = int(config.DIARIZATION_RECOVERY_MAX_ASR_SEGMENTS)
            allow_local_recovery = bool(config.DIARIZATION_RECOVERY_LOCAL_PYANNOTE)

            _emit(
                progress,
                30,
                "Speaker-track recovery: primary track mostly interviewer speech...",
                backend="hybrid" if use_kaggle_segment else "local",
            )

            def _evaluate_track(
                track_answers: list[dict[str, Any]],
                *,
                recovery_label: str,
                speaker_meta: dict[str, Any] | None = None,
            ) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], Any]:
                if not track_answers:
                    return 0, [], [], None
                if len(track_answers) > recovery_max_asr:
                    log.log(
                        "diarization_filter",
                        f"Skipping {recovery_label}: {len(track_answers)} segments "
                        f"exceeds recovery ASR cap ({recovery_max_asr})",
                        phase="analyze",
                        level="warning",
                        metrics={
                            "recovery_label": recovery_label,
                            "segment_count": len(track_answers),
                            "cap": recovery_max_asr,
                        },
                    )
                    return 0, [], [], None
                _emit(
                    progress,
                    32,
                    f"Recovery ASR: {len(track_answers)} segment(s) via {recovery_label}...",
                    backend="kaggle" if use_kaggle_asr else "local",
                )
                if use_kaggle_asr:
                    track_cache = _kaggle_prefetch_answers(
                        gpu_client,
                        track_answers,
                        gpu_reading_profile,
                        log,
                        transcript_proc=transcript_proc,
                        progress=progress,
                    )
                    track_transcripts = [
                        row[0] if row[0] is not None else {"transcript": ""}
                        for row in track_cache
                    ]
                else:
                    track_transcripts = transcript_proc.transcribe_answers(track_answers)
                    track_cache = None
                track_kept, track_excluded, track_mask = filter_segments_by_transcript(
                    track_answers, track_transcripts
                )
                if len(track_kept) > len(kept):
                    nonlocal interview_answers, kaggle_cache, local_transcript_cache, recovered
                    interview_answers = track_kept
                    if track_cache is not None:
                        kaggle_cache = [
                            row for row, keep in zip(track_cache, track_mask) if keep
                        ]
                        local_transcript_cache = None
                    else:
                        kaggle_cache = None
                        local_transcript_cache = [
                            row
                            for row, keep in zip(track_transcripts, track_mask)
                            if keep
                        ]
                    log.log(
                        "diarization_filter",
                        f"Recovered {len(track_kept)} candidate segment(s) via {recovery_label}",
                        phase="analyze",
                        level="warning",
                        metrics={
                            "recovery_label": recovery_label,
                            "track_excluded": track_excluded,
                            "track_kept": len(track_kept),
                            "speaker_selection": speaker_meta,
                        },
                        decision=recovery_label,
                    )
                    recovered = True
                return len(track_kept), track_kept, track_excluded, track_cache

            def _try_alternate_payload(
                boundaries: list[dict[str, Any]],
                *,
                recovery_label: str,
                speaker_meta: dict[str, Any] | None = None,
            ) -> None:
                nonlocal recovered
                if recovered or not boundaries or not wav_path or not Path(wav_path).is_file():
                    return
                log.log(
                    "diarization_filter",
                    f"Trying {recovery_label} ({len(boundaries)} segment(s))",
                    phase="analyze",
                    level="warning",
                    metrics={
                        "primary_kept": len(kept),
                        "primary_excluded": len(excluded),
                        "alternate_segments": len(boundaries),
                    },
                )
                track_answers = audio.process_interview_from_segmentation(
                    str(video),
                    boundaries,
                    speaker_selection={
                        **speaker_sel,
                        **(speaker_meta or {}),
                        "recovery_track": recovery_label,
                    },
                    wav_path=str(wav_path),
                )
                _evaluate_track(
                    track_answers,
                    recovery_label=recovery_label,
                    speaker_meta=speaker_meta,
                )

            # 1) Kaggle dual-track payload (when notebook returns alternate_answers)
            if alt_payload:
                _try_alternate_payload(
                    alt_payload,
                    recovery_label="alternate_speaker_track",
                )

            # 2) Kaggle re-segment forcing the other diarization label
            if (
                not recovered
                and use_kaggle_segment
                and wav_path
                and Path(wav_path).is_file()
            ):
                alt_speaker = infer_alternate_speaker(speaker_sel)
                if alt_speaker:
                    _emit(
                        progress,
                        31,
                        f"Kaggle re-segment with alternate speaker ({alt_speaker})...",
                        backend="kaggle",
                    )
                    retry_payload = gpu_client.segment_interview(
                        str(video),
                        wav_path=str(wav_path),
                        force_speaker=alt_speaker,
                    )
                    retry_answers = (retry_payload or {}).get("answers") or []
                    if retry_answers:
                        segment_ctx["payload"] = retry_payload
                        _try_alternate_payload(
                            retry_answers,
                            recovery_label="kaggle_force_speaker",
                            speaker_meta={
                                "forced_speaker": alt_speaker,
                                "speaker_selection": (retry_payload or {}).get(
                                    "speaker_selection"
                                ),
                            },
                        )

            # 3) Local pyannote dual-track (opt-in — very slow on CPU; skipped when Kaggle segment is on)
            if (
                not recovered
                and allow_local_recovery
                and wav_path
                and Path(wav_path).is_file()
            ):
                _emit(
                    progress,
                    33,
                    "Local pyannote dual-track recovery (may take several minutes)...",
                    backend="local",
                )
                log.log(
                    "diarization_filter",
                    "Building alternate speaker track locally (pyannote dual-track)",
                    phase="analyze",
                    level="warning",
                    metrics={
                        "primary_kept": len(kept),
                        "kaggle_alternate_count": len(alt_payload),
                    },
                )
                _primary_b, local_alt_b, local_sel = audio.segment_interview_dual_track(
                    str(wav_path)
                )
                _try_alternate_payload(
                    local_alt_b,
                    recovery_label="local_dual_track_alternate",
                    speaker_meta=local_sel,
                )
                if not recovered and _primary_b:
                    _try_alternate_payload(
                        _primary_b,
                        recovery_label="local_dual_track_primary",
                        speaker_meta=local_sel,
                    )
            elif not recovered and not allow_local_recovery:
                log.log(
                    "diarization_filter",
                    "Skipping local pyannote recovery (DIARIZATION_RECOVERY_LOCAL_PYANNOTE=false)",
                    phase="analyze",
                    level="warning",
                    metrics={"primary_kept": len(kept), "primary_excluded": len(excluded)},
                )

            # 4) Full local diarization when still insufficient (opt-in only)
            min_expected = max(3, int(round(primary_total * 0.35)))
            if (
                not recovered
                and allow_local_recovery
                and len(kept) < min_expected
            ):
                log.log(
                    "diarization_filter",
                    "Insufficient candidate segments — re-running local pyannote diarization",
                    phase="analyze",
                    level="warning",
                    metrics={"kept": len(kept), "min_expected": min_expected},
                )
                interview_answers = audio.process_interview(str(video))
                if use_kaggle_asr:
                    kaggle_cache = _kaggle_prefetch_answers(
                        gpu_client,
                        interview_answers,
                        gpu_reading_profile,
                        log,
                        transcript_proc=transcript_proc,
                        progress=progress,
                    )
                    local_transcript_cache = None
                else:
                    local_transcript_cache = transcript_proc.transcribe_answers(
                        interview_answers
                    )
                    kaggle_cache = None
                kept, excluded = _apply_interviewer_filter()
                if excluded:
                    log.log(
                        "diarization_filter",
                        f"After local diarization: excluded {len(excluded)} interviewer segment(s)",
                        phase="analyze",
                        metrics={"excluded": excluded, "kept": len(kept)},
                    )

        def _redo_kaggle_segmentation(
            reason: str, *, metrics: dict[str, Any] | None = None
        ) -> bool:
            """Re-segment on Kaggle (alternate track / force_speaker) when quality is still bad."""
            nonlocal interview_answers, kaggle_cache, local_transcript_cache
            from engine.interviewer_segment_filter import infer_alternate_speaker

            _emit(
                progress,
                26,
                "Re-segmenting on Kaggle GPU (speaker track quality too low)...",
                backend="kaggle",
            )
            log.log(
                "diarization_filter",
                f"Kaggle re-segmentation: {reason}",
                phase="analyze",
                level="warning",
                metrics=metrics or {},
                decision="kaggle_quality_fallback",
            )

            wav_path = segment_ctx.get("wav_path")
            if not wav_path or not Path(str(wav_path)).is_file():
                wav_path = audio.extract_audio(str(video))
                segment_ctx["wav_path"] = wav_path

            payload = segment_ctx.get("payload") or {}
            speaker_sel = payload.get("speaker_selection") or audio.last_speaker_selection or {}
            alt_payload = payload.get("alternate_answers") or []
            recovered = False

            def _apply_kaggle_track(
                boundaries: list[dict[str, Any]],
                *,
                recovery_label: str,
                speaker_meta: dict[str, Any] | None = None,
            ) -> bool:
                nonlocal interview_answers, kaggle_cache, local_transcript_cache, recovered
                if not boundaries:
                    return False
                track_answers = audio.process_interview_from_segmentation(
                    str(video),
                    boundaries,
                    speaker_selection={
                        **speaker_sel,
                        **(speaker_meta or {}),
                        "recovery_track": recovery_label,
                    },
                    wav_path=str(wav_path),
                )
                if not track_answers:
                    return False
                if use_kaggle_asr:
                    track_cache = _kaggle_prefetch_answers(
                        gpu_client,
                        track_answers,
                        gpu_reading_profile,
                        log,
                        transcript_proc=transcript_proc,
                        progress=progress,
                    )
                    track_transcripts = [
                        row[0] if row[0] is not None else {"transcript": ""}
                        for row in track_cache
                    ]
                else:
                    track_transcripts = transcript_proc.transcribe_answers(track_answers)
                    track_cache = None
                track_kept, track_excluded, track_mask = filter_segments_by_transcript(
                    track_answers, track_transcripts
                )
                if len(track_kept) <= len(interview_answers):
                    return False
                interview_answers = track_kept
                if track_cache is not None:
                    kaggle_cache = [
                        row for row, keep in zip(track_cache, track_mask) if keep
                    ]
                    local_transcript_cache = None
                else:
                    kaggle_cache = None
                    local_transcript_cache = [
                        row for row, keep in zip(track_transcripts, track_mask) if keep
                    ]
                log.log(
                    "diarization_filter",
                    f"Kaggle quality fallback recovered {len(track_kept)} segment(s) via {recovery_label}",
                    phase="analyze",
                    level="warning",
                    metrics={
                        "recovery_label": recovery_label,
                        "track_excluded": track_excluded,
                        "track_kept": len(track_kept),
                    },
                    decision=recovery_label,
                )
                recovered = True
                return True

            if alt_payload and _apply_kaggle_track(
                alt_payload, recovery_label="kaggle_alternate_quality_fallback"
            ):
                return True

            alt_speaker = infer_alternate_speaker(speaker_sel)
            if alt_speaker:
                retry_payload = gpu_client.segment_interview(
                    str(video),
                    wav_path=str(wav_path),
                    force_speaker=alt_speaker,
                )
                retry_answers = (retry_payload or {}).get("answers") or []
                if retry_answers:
                    segment_ctx["payload"] = retry_payload
                    if _apply_kaggle_track(
                        retry_answers,
                        recovery_label="kaggle_force_speaker_quality_fallback",
                        speaker_meta={
                            "forced_speaker": alt_speaker,
                            "speaker_selection": (retry_payload or {}).get(
                                "speaker_selection"
                            ),
                        },
                    ):
                        return True
            return recovered

        def _redo_local_diarization(reason: str, *, metrics: dict[str, Any] | None = None) -> None:
            nonlocal interview_answers, kaggle_cache, local_transcript_cache
            _emit(
                progress,
                26,
                "Re-segmenting with local pyannote (speaker track quality too low)...",
                backend="local",
            )
            log.log(
                "diarization_filter",
                f"Local pyannote re-diarization: {reason}",
                phase="analyze",
                level="warning",
                metrics=metrics or {},
                decision="local_pyannote_quality_fallback",
            )
            interview_answers = audio.process_interview(str(video))
            if use_kaggle_asr:
                kaggle_cache = _kaggle_prefetch_answers(
                    gpu_client,
                    interview_answers,
                    gpu_reading_profile,
                    log,
                    transcript_proc=transcript_proc,
                    progress=progress,
                )
                local_transcript_cache = None
            else:
                local_transcript_cache = transcript_proc.transcribe_answers(
                    interview_answers
                )
                kaggle_cache = None
            kept, excluded = _apply_interviewer_filter()
            if excluded:
                log.log(
                    "diarization_filter",
                    f"After local fallback: excluded {len(excluded)} bad segment(s)",
                    phase="analyze",
                    metrics={"excluded": excluded, "kept": len(kept)},
                )

        quality_ok, quality_meta = assess_kept_segment_quality(_transcripts_for_filter())
        if not quality_ok:
            if use_kaggle_segment and config.DIARIZATION_AUTO_KAGGLE_FALLBACK:
                _redo_kaggle_segmentation(
                    str(quality_meta.get("reason", "low_quality_segments")),
                    metrics=quality_meta,
                )
            elif config.DIARIZATION_AUTO_LOCAL_FALLBACK:
                _redo_local_diarization(
                    str(quality_meta.get("reason", "low_quality_segments")),
                    metrics=quality_meta,
                )
            quality_ok, quality_meta = assess_kept_segment_quality(_transcripts_for_filter())
            if not quality_ok:
                log.log(
                    "diarization_filter",
                    "Segment quality still low after segmentation fallback",
                    phase="analyze",
                    level="warning",
                    metrics=quality_meta,
                )

    wav_cleanup = segment_ctx.get("wav_path")
    if wav_cleanup:
        Path(wav_cleanup).unlink(missing_ok=True)
        segment_ctx["wav_path"] = None

    results_answers: list[dict] = []
    n_answers = max(len(interview_answers), 1)

    for idx, answer in enumerate(interview_answers):
        pct = 30 + int(55 * (idx / n_answers))
        aid = answer["answer_id"]
        score_backend = "hybrid" if kaggle_cache is not None else "local"
        _emit(
            progress,
            pct,
            f"Scoring answer {aid + 1}/{n_answers} (acoustic + contrastive local; ASR from "
            f"{'Kaggle GPU' if kaggle_cache is not None else 'local'})...",
            backend=score_backend,
        )

        duration = float(answer["end_sec"]) - float(answer["start_sec"])

        gpu: float | None = None
        transcript: dict[str, Any] | None = None
        if kaggle_cache is not None:
            transcript, gpu = kaggle_cache[idx]
            if transcript is None:
                log.log(
                    "kaggle_transcribe",
                    f"Kaggle failed for answer {aid} — falling back to local Whisper",
                    phase="analyze",
                    level="warning",
                )
                transcript = transcript_proc.transcribe_answer(
                    answer_id=answer["answer_id"],
                    audio_bytes=answer.get("audio_bytes", b""),
                    start_sec=answer["start_sec"],
                    end_sec=answer["end_sec"],
                )
        else:
            if local_transcript_cache is not None:
                transcript = local_transcript_cache[idx]
            else:
                transcript = transcript_proc.transcribe_answer(
                    answer_id=answer["answer_id"],
                    audio_bytes=answer.get("audio_bytes", b""),
                    start_sec=answer["start_sec"],
                    end_sec=answer["end_sec"],
                )
            gpu = _gpu_score(
                gpu_client,
                answer.get("audio_bytes", b""),
                gpu_reading_profile or {},
                duration,
            )

        tx_text = str((transcript or {}).get("transcript", ""))
        if is_interviewer_transcript(tx_text) or is_off_topic_or_wrong_slice(tx_text):
            log.log(
                "diarization_filter",
                f"Skipping answer {aid}: mis-segmented transcript after filter",
                phase="analyze",
                level="warning",
                metrics={
                    "start_sec": answer.get("start_sec"),
                    "end_sec": answer.get("end_sec"),
                    "transcript_preview": tx_text[:160],
                },
                decision="skip_missegmented_answer",
            )
            continue

        ac_score, ac_breakdown = engine.score_answer(answer["windows"], reading_profile)
        tech_density = _answer_technical_density(transcript)
        ac_score = AnalysisEngine.calibrate_channel_score(
            ac_score,
            duration_sec=duration,
            technical_density=tech_density,
        )
        t_window: list[dict[str, Any]] = []
        gaze_score, gaze_breakdown = None, {}
        lip_score, lip_breakdown = None, {}

        ling_score, ling_breakdown = linguistic_analyzer.analyze(
            transcript, linguistic_calibration
        )
        if config.ENABLE_COGNITIVE_SPONTANEITY:
            from engine.cognitive_spontaneity import dampen_linguistic_fluency_score

            ling_score = dampen_linguistic_fluency_score(
                ling_score, transcript, duration
            )

        semantic_spec: dict[str, Any] = {}
        specificity_score: float | None = None
        if config.ENABLE_SEMANTIC_SPECIFICITY:
            semantic_spec = compute_semantic_specificity(transcript)
            specificity_score = float(semantic_spec.get("generic_script_likelihood", 0.0))

        fused = scorer.score_answer(
            answer_id=answer["answer_id"],
            scores={
                "acoustic": ac_score,
                "linguistic": ling_score,
                "specificity": specificity_score,
                "gaze": gaze_score,
                "lip": lip_score,
                "gpu": gpu,
            },
            start_sec=answer["start_sec"],
            end_sec=answer["end_sec"],
        )
        fused_scorer_ewma = float(fused.get("smoothed_score", 0.0))
        null_video_channels += 1

        contrastive_summary: dict | None = None
        if use_contrastive:
            contrastive_summary = contrastive.process_answer(answer, transcript, timeline)
            fused["contrastive"] = contrastive_summary
            fused["confidence"] = contrastive_summary.get("confidence", "LOW")
            fused["status"] = contrastive_summary.get("status", fused["status"])
            contrastive_composite = float(
                contrastive_summary.get("composite_score")
                or contrastive_summary.get("ewma_score")
                or 0.0
            )
            fused["contrastive_composite"] = contrastive_composite
            fused["fused_scorer_ewma"] = fused_scorer_ewma
            fused["smoothed_score"] = (
                contrastive_composite
                if contrastive_composite > 0
                else fused_scorer_ewma
            )
            fused["ewma_score"] = fused["smoothed_score"]

            for w in contrastive_summary.get("windows", []):
                log.log(
                    "window",
                    (
                        f"Window {w.get('window_id')} "
                        f"[{w.get('start_sec', 0):.1f}-{w.get('end_sec', 0):.1f}s]"
                    ),
                    phase="analyze",
                    metrics={
                        "answer_id": aid,
                        "script_similarity": w.get("script_similarity"),
                        "natural_similarity": w.get("natural_similarity"),
                        "contrastive_score": w.get("contrastive_score"),
                        "naturality_score": w.get("naturality_score"),
                        "suspicious_flag": w.get("suspicious_flag"),
                        "suspicion_level": w.get("suspicion_level"),
                        "confidence_level": w.get("confidence_level"),
                        "ewma_after": w.get("ewma_after"),
                    },
                    decision=(
                        "SUSPICIOUS" if w.get("suspicious_flag") else "clear"
                    ),
                )

        if intra_session is not None:
            intra_block = intra_session.process_answer(
                answer,
                transcript,
                timeline,
                contrastive_summary=contrastive_summary,
            )
            fused = intra_session.finalize_answer(
                fused,
                intra_block,
                answer=answer,
                transcript=transcript,
                timeline=timeline,
                contrastive_summary=contrastive_summary,
            )
            log.log(
                "intra_individual",
                f"Answer {aid}: P(external)={intra_block.get('p_external_guidance', 0):.2f}",
                phase="analyze",
                metrics={
                    "answer_id": aid,
                    "p_external": intra_block.get("p_external_guidance"),
                    "rel_mean_deviation": (intra_block.get("person_relative") or {}).get(
                        "rel_mean_deviation"
                    ),
                    "intra_turbulence_suppression": (
                        intra_block.get("intra_answer_turbulence") or {}
                    ).get("intra_turbulence_suppression"),
                    "cognitive_cost_flatness": (intra_block.get("cognitive_cost") or {}).get(
                        "cognitive_cost_flatness"
                    ),
                    "intra_status": intra_block.get("intra_status"),
                },
                decision=fused.get("status"),
            )

        if config.ENABLE_SEMANTIC_SPECIFICITY and semantic_spec:
            beh = (contrastive_summary or {}).get("behavioral_synthesis") or {}
            ext_like = float(beh.get("external_sourcing_likelihood", 0.0) or 0.0)
            int_like = float(beh.get("internal_generation_likelihood", 0.0) or 0.0)
            session_snap = session_accumulator.after_answer(
                answer_id=aid,
                transcript=transcript,
                generic_script_likelihood=float(
                    semantic_spec.get("generic_script_likelihood", 0.0)
                ),
                contrastive_external=ext_like,
                contrastive_internal=int_like,
            )
            content_prof = session_snap.get("content_profile") or {}
            prior_status = str(fused.get("status", "CLEAR"))
            weighted_ev = float(
                (contrastive_summary or {}).get("weighted_evidence")
                or ((contrastive_summary or {}).get("composite_meta") or {}).get(
                    "weighted_evidence", 0.0
                )
                or 0.0
            )
            new_status, spec_reasons = apply_specificity_to_status(
                prior_status,
                semantic_spec,
                session_external_prior=float(
                    session_snap.get("session_external_prior", 0.5)
                ),
                content_uniformity=float(content_prof.get("content_uniformity", 0.0)),
                answer_index=idx,
                contrastive_external=ext_like,
                weighted_evidence=weighted_ev,
            )
            if new_status != prior_status:
                fused["status"] = new_status
                if contrastive_summary is not None:
                    contrastive_summary["status"] = new_status
                    existing = contrastive_summary.get("decision_explanation") or []
                    if not isinstance(existing, list):
                        existing = [str(existing)]
                    for r in spec_reasons:
                        if r not in existing:
                            existing.append(r)
                    contrastive_summary["decision_explanation"] = existing

            fused["semantic_specificity"] = semantic_spec
            fused["session_feedforward"] = session_snap
            log.log(
                "semantic_specificity",
                f"Answer {aid}: specificity={semantic_spec.get('specificity_score', 0):.2f} "
                f"generic={semantic_spec.get('generic_script_likelihood', 0):.2f}",
                phase="analyze",
                metrics={
                    "answer_id": aid,
                    **semantic_spec,
                    "session_external_prior": session_snap.get("session_external_prior"),
                    "content_uniformity": content_prof.get("content_uniformity"),
                },
                decision=fused.get("status"),
            )

        if null_video_channels >= 2 and idx == 1:
            log.log(
                "video",
                "Gaze/lip unavailable for all answers (video scanning disabled)",
                phase="analyze",
                level="warning",
                metrics={"null_gaze_lip_answers": null_video_channels},
                decision="Acoustic + linguistic + specificity channels only.",
            )

        log.log(
            "answer",
            f"Answer {aid}: {fused['status']}",
            phase="analyze",
            metrics={
                "answer_id": aid,
                "start_sec": answer["start_sec"],
                "end_sec": answer["end_sec"],
                "acoustic": ac_score,
                "linguistic": ling_score,
                "gaze": None,
                "lip": None,
                "fused_raw": fused.get("raw_score"),
                "fused_ewma": fused.get("smoothed_score"),
                "fused_scorer_ewma": fused.get("fused_scorer_ewma", fused_scorer_ewma),
                "specificity_score": (
                    semantic_spec.get("specificity_score") if semantic_spec else None
                ),
                "generic_script_likelihood": (
                    semantic_spec.get("generic_script_likelihood") if semantic_spec else None
                ),
                "confidence": fused.get("confidence"),
                "contrastive_ewma": (
                    contrastive_summary.get("composite_score")
                    or contrastive_summary.get("ewma_score")
                    if contrastive_summary
                    else None
                ),
            },
            decision=fused["status"],
        )

        results_answers.append(
            {
                **fused,
                "acoustic_breakdown": ac_breakdown,
                "gaze_breakdown": gaze_breakdown,
                "lip_breakdown": lip_breakdown,
                "linguistic_breakdown": ling_breakdown,
                "transcript": transcript.get("transcript", ""),
                "transcription_backend": transcript.get("transcription_backend", ""),
                "num_windows": len(answer.get("windows", [])),
                "timeline_frames": 0,
            }
        )

    session_sourcing: dict[str, Any] = {}
    if use_contrastive and config.ENABLE_COGNITIVE_SOURCING:
        from engine.cognitive_sourcing import finalize_interview_sourcing

        results_answers, session_sourcing = finalize_interview_sourcing(results_answers)

    # Final transcript-authority pass — session sourcing must not override personal answers
    if config.ENABLE_SEMANTIC_SPECIFICITY:
        content_prof = session_accumulator.content.session_profile()
        for ans in results_answers:
            spec = ans.get("semantic_specificity")
            if not spec:
                continue
            c = ans.get("contrastive") or {}
            beh = c.get("behavioral_synthesis") or {}
            weighted_ev = float(
                c.get("weighted_evidence")
                or (c.get("composite_meta") or {}).get("weighted_evidence", 0.0)
                or 0.0
            )
            new_status, spec_reasons = apply_specificity_to_status(
                str(ans.get("status", "CLEAR")),
                spec,
                session_external_prior=session_accumulator.session_external_prior,
                content_uniformity=float(content_prof.get("content_uniformity", 0.0)),
                answer_index=int(ans.get("answer_id", 0)),
                contrastive_external=float(beh.get("external_sourcing_likelihood", 0.0) or 0.0),
                weighted_evidence=weighted_ev,
            )
            if new_status != ans.get("status"):
                ans["status"] = new_status
                if c:
                    c["status"] = new_status
                    existing = c.get("decision_explanation") or []
                    if not isinstance(existing, list):
                        existing = [str(existing)]
                    for r in spec_reasons:
                        if r not in existing:
                            existing.append(r)
                    c["decision_explanation"] = existing

    session_intra: dict[str, Any] = {}
    if intra_session is not None:
        results_answers, session_intra = intra_session.finalize_session(results_answers)
        log.log(
            "intra_individual_session",
            f"Session P(external)={session_intra.get('session_probability', {}).get('p_external_final', 0):.2f}",
            phase="analyze",
            metrics=session_intra.get("cross_answer_drift"),
            decision=f"drift_uniformity={session_intra.get('cross_answer_drift', {}).get('cross_answer_uniformity')}",
        )

    if config.ENABLE_LLM_JUDGE:
        from engine.llm_judge import apply_llm_judge_to_answers

        results_answers = apply_llm_judge_to_answers(results_answers, log=log)

    payload: dict[str, Any] = {
        "version": 5 if use_contrastive else 4,
        "video": str(video),
        "calibration": cal_label,
        "timeline_path": video_result.get("timeline_path", ""),
        "contrastive_engine": use_contrastive,
        "candidate_speaker_strategy": config.CANDIDATE_SPEAKER,
        "speaker_selection": speaker_sel,
        "answers": results_answers,
        "decision_log": log.to_list(),
    }
    if use_contrastive:
        payload["window_logs"] = contrastive.export_window_logs()
        payload["profile_update_logs"] = contrastive.export_profile_update_logs()
        payload["profiles_end"] = contrastive.export_profiles()
        if config.ENABLE_COGNITIVE_SOURCING:
            payload["session_sourcing_inference"] = session_sourcing
        if session_intra:
            payload["session_intra_individual"] = session_intra
        payload["session_content_analysis"] = session_accumulator.content.session_profile()
        payload["session_feedforward_prior"] = round(
            session_accumulator.session_external_prior, 4
        )

    alerts = sum(1 for a in results_answers if a.get("status") == "PROBABLE_SCRIPT_READING")
    log.log(
        "complete",
        f"Analysis complete: {alerts}/{len(results_answers)} alerts",
        phase="analyze",
        metrics={
            "alerts": alerts,
            "answers": len(results_answers),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
    )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["results_path"] = str(out_path)

    _emit(progress, 100, "Analysis complete", None, backend="local")

    payload["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return payload
