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
from engine.gaze_analyzer import GazeAnalyzer
from engine.linguistic_analyzer import LinguisticAnalyzer
from engine.lip_analyzer import LipAnalyzer
from engine.profile_memory import BehavioralProfile
from gpu_client import KaggleGPUClient
from processors.audio_processor import AudioProcessor
from processors.transcript_processor import (
    TranscriptProcessor,
    load_whisper_calibration_model,
    load_whisper_model,
    preload_models,
)
from processors.video_processor import VideoProcessor
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


def _emit(
    cb: ProgressCallback | None,
    percent: int,
    message: str,
    log_entry: dict[str, Any] | None = None,
) -> None:
    if cb:
        cb(percent, message, log_entry)


def _startup_models(
    log: PipelineLogger,
    *,
    calibration: bool = False,
) -> KaggleGPUClient:
    if calibration and config.FAST_CALIBRATION:
        log.log(
            "startup",
            (
                f"Loading models (calibration Whisper "
                f"{config.WHISPER_CALIBRATION_MODEL_SIZE}, interview {config.WHISPER_MODEL_SIZE})"
            ),
            phase="system",
        )
        load_whisper_calibration_model()
    else:
        log.log(
            "startup",
            f"Loading models (Whisper {config.WHISPER_MODEL_SIZE}, {config.WHISPER_COMPUTE_TYPE})",
            phase="system",
        )
        load_whisper_model()
    return KaggleGPUClient(
        base_url=config.KAGGLE_GPU_URL,
        secret=config.SENTINEL_SECRET,
        timeout=config.KAGGLE_GPU_TIMEOUT_SEC,
    )


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
    video_proc = VideoProcessor()
    engine = AnalysisEngine()
    gaze_analyzer = GazeAnalyzer()
    lip_analyzer = LipAnalyzer()
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
        return video_proc.process_video(str(video), timeline_fps=video_fps)

    t_audio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        audio_future = pool.submit(_run_audio)
        video_future = pool.submit(_run_video)
        answers = audio_future.result()
        video_result = video_future.result()
    audio_sec = round(time.perf_counter() - t_audio, 2)

    log.log(
        "audio",
        f"Calibration audio: {len(answers)} answer segment(s) ({audio_sec}s, parallel w/ video)",
        phase="calibrate",
        metrics={
            "answers": len(answers),
            "fast_mode": fast,
            "skip_diarization": config.SKIP_DIARIZATION_CALIBRATION,
            "elapsed_sec": audio_sec,
        },
    )
    gpu_reading_profile: dict | None = None
    if gpu_client.enabled:
        _emit(progress, 12, "GPU calibration (Kaggle)...", None)
        for answer in answers:
            audio = answer.get("audio_bytes", b"")
            if not audio:
                continue
            gpu_result = gpu_client.calibrate(audio)
            if gpu_result:
                gpu_reading_profile = gpu_result
        if gpu_reading_profile:
            log.log(
                "gpu_calibrate",
                "GPU reading profile from Kaggle /calibrate",
                phase="calibrate",
                metrics={"has_parselmouth": "parselmouth_baseline" in gpu_reading_profile},
            )
        else:
            log.log(
                "gpu_calibrate",
                "Kaggle GPU calibrate returned no profile (check KAGGLE_GPU_URL / notebook)",
                phase="calibrate",
                level="warning",
            )

    windows = AudioProcessor.collect_windows(answers)
    reading_profile = engine.calibrate(windows)
    log.log(
        "acoustic_profile",
        "Built acoustic reading profile from openSMILE windows",
        phase="calibrate",
        metrics={"windows": len(windows), "profile_keys": len(reading_profile)},
    )
    _emit(progress, 35, f"Acoustic profile from {len(windows)} windows", None)

    _emit(
        progress,
        40,
        f"Transcribing calibration (Whisper {config.WHISPER_CALIBRATION_MODEL_SIZE if fast else config.WHISPER_MODEL_SIZE})...",
    )
    t_asr = time.perf_counter()
    cal_transcripts = transcript_proc.transcribe_answers(answers, calibration_fast=fast)
    asr_sec = round(time.perf_counter() - t_asr, 2)
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

    timeline = video_result["timeline"]
    gaze_calibration = gaze_analyzer.calibrate(timeline)
    lip_calibration = lip_analyzer.calibrate(timeline)
    log.log(
        "video",
        f"Video timeline: {len(timeline)} frames @ {video_fps} fps",
        phase="calibrate",
        metrics={"frames": len(timeline), "timeline_fps": video_fps},
    )
    _emit(progress, 85, "Building SCRIPT behavioral profile...")

    contrastive = ContrastiveEngine()
    script_profile = contrastive.build_script_profile_from_calibration(
        answers,
        transcripts=cal_transcripts,
        timeline=timeline,
    )
    log.log(
        "script_profile",
        "SCRIPT profile built from intentional reading windows",
        phase="calibrate",
        metrics={"samples": script_profile.sample_count},
        decision="This profile represents how the user sounds while reading.",
    )

    profile = {
        "version": 5,
        "source_video": str(video),
        "timeline_path": video_result["timeline_path"],
        "acoustic_reading_profile": reading_profile,
        "gpu_reading_profile": gpu_reading_profile,
        "script_profile": script_profile.to_dict(),
        "linguistic_calibration": linguistic_calibration,
        "gaze_calibration": gaze_calibration,
        "lip_calibration": lip_calibration,
        "calibration_answers": len(answers),
        "calibration_windows": len(windows),
        "timeline_frames": len(timeline),
        "contrastive_engine": True,
    }
    save_baseline_profile(profile, out_path)

    elapsed = round(time.perf_counter() - t0, 2)
    log.log(
        "complete",
        f"Calibration saved ({elapsed}s)",
        phase="calibrate",
        metrics={"elapsed_sec": elapsed, "output": str(out_path)},
    )
    video_proc.close()
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

    _emit(progress, 2, "Starting analysis...", log.log("start", f"Interview: {video.name}", phase="analyze"))

    reading_profile = profile.get("acoustic_reading_profile")
    if not reading_profile:
        raise ValueError("Calibration profile missing acoustic_reading_profile")
    gpu_reading_profile = profile.get("gpu_reading_profile")

    gaze_calibration = profile.get("gaze_calibration", {})
    lip_calibration = profile.get("lip_calibration", {})
    linguistic_calibration = profile.get("linguistic_calibration", {})

    gpu_client = _startup_models(log)
    _emit(progress, 5, "Models loaded", None)

    audio = AudioProcessor()
    video_proc = VideoProcessor()
    transcript_proc = TranscriptProcessor()
    engine = AnalysisEngine()
    gaze_analyzer = GazeAnalyzer()
    lip_analyzer = LipAnalyzer()
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
        log.log(
            "natural_profile",
            "NATURAL profile starts empty (no first-N-seconds assumption)",
            phase="analyze",
            decision="Only high-naturality windows will update NATURAL profile.",
        )

    _emit(progress, 12, "Processing interview audio...")
    interview_answers = audio.process_interview(str(video))
    speaker_sel = audio.last_speaker_selection or {}
    log.log(
        "audio",
        f"Interview: {len(interview_answers)} answer segment(s)",
        phase="analyze",
        metrics={
            "answers": len(interview_answers),
            "candidate_speaker_strategy": config.CANDIDATE_SPEAKER,
            "speaker_selection": speaker_sel,
        },
        decision=f"candidate_track={speaker_sel.get('strategy', config.CANDIDATE_SPEAKER)}",
    )

    _emit(progress, 28, "Processing interview video...")
    video_result = video_proc.process_video(str(video))
    timeline = video_result["timeline"]

    results_answers: list[dict] = []
    n_answers = max(len(interview_answers), 1)

    for idx, answer in enumerate(interview_answers):
        pct = 30 + int(55 * (idx / n_answers))
        aid = answer["answer_id"]
        _emit(progress, pct, f"Scoring answer {aid + 1}/{n_answers}...")

        ac_score, ac_breakdown = engine.score_answer(answer["windows"], reading_profile)
        duration = float(answer["end_sec"]) - float(answer["start_sec"])
        transcript = transcript_proc.transcribe_answer(
            answer_id=answer["answer_id"],
            audio_bytes=answer.get("audio_bytes", b""),
            start_sec=answer["start_sec"],
            end_sec=answer["end_sec"],
        )
        tech_density = _answer_technical_density(transcript)
        ac_score = AnalysisEngine.calibrate_channel_score(
            ac_score,
            duration_sec=duration,
            technical_density=tech_density,
        )
        t_window = VideoProcessor.slice_timeline(
            timeline, answer["start_sec"], answer["end_sec"]
        )
        gaze_score, gaze_breakdown = gaze_analyzer.analyze(t_window, gaze_calibration)
        lip_score, lip_breakdown = lip_analyzer.analyze(t_window, lip_calibration)

        ling_score, ling_breakdown = linguistic_analyzer.analyze(
            transcript, linguistic_calibration
        )

        gpu = _gpu_score(
            gpu_client,
            answer.get("audio_bytes", b""),
            gpu_reading_profile or {},
            duration,
        )

        fused = scorer.score_answer(
            answer_id=answer["answer_id"],
            scores={
                "acoustic": ac_score,
                "linguistic": ling_score,
                "gaze": gaze_score,
                "lip": lip_score,
                "gpu": gpu,
            },
            start_sec=answer["start_sec"],
            end_sec=answer["end_sec"],
        )

        contrastive_summary: dict | None = None
        if use_contrastive:
            contrastive_summary = contrastive.process_answer(answer, transcript, timeline)
            fused["contrastive"] = contrastive_summary
            fused["confidence"] = contrastive_summary.get("confidence", "LOW")
            fused["status"] = contrastive_summary.get("status", fused["status"])
            fused["smoothed_score"] = contrastive_summary.get(
                "composite_score",
                contrastive_summary.get("ewma_score", fused["smoothed_score"]),
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
                "gaze": gaze_score,
                "lip": lip_score,
                "fused_raw": fused.get("raw_score"),
                "fused_ewma": fused.get("smoothed_score"),
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
                "timeline_frames": len(t_window),
            }
        )

    payload: dict[str, Any] = {
        "version": 5 if use_contrastive else 4,
        "video": str(video),
        "calibration": cal_label,
        "timeline_path": video_result["timeline_path"],
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

    video_proc.close()
    _emit(progress, 100, "Analysis complete", None)

    payload["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return payload
