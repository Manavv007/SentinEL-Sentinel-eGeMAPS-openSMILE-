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
            "Kaggle GPU offload active — skipping local Whisper load for interview",
            phase="system",
            metrics={"kaggle_url": config.KAGGLE_GPU_URL},
        )
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
) -> list[tuple[dict[str, Any] | None, float | None]]:
    """Parallel Kaggle transcription (+ GPU score when profile available)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(answers)
    out: list[tuple[dict[str, Any] | None, float | None]] = [(None, None)] * n
    workers = min(config.KAGGLE_PARALLEL_ANSWERS, max(n, 1))

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
        f"Offloading {n} answer(s) to Kaggle GPU (workers={workers})",
        phase="analyze",
        metrics={"parallel_workers": workers, "has_gpu_profile": bool(gpu_reading_profile)},
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i, a) for i, a in enumerate(answers)]
        for fut in as_completed(futures):
            idx, transcript, gpu = fut.result()
            out[idx] = (transcript, gpu)
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

    _emit(progress, 2, "Starting analysis...", log.log("start", f"Interview: {video.name}", phase="analyze"))

    reading_profile = profile.get("acoustic_reading_profile")
    if not reading_profile:
        raise ValueError("Calibration profile missing acoustic_reading_profile")
    gpu_reading_profile = profile.get("gpu_reading_profile")

    linguistic_calibration = profile.get("linguistic_calibration", {})

    gpu_client = _startup_models(log)
    _emit(progress, 5, "Models loaded", None)

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
        log.log(
            "natural_profile",
            "NATURAL profile starts empty (no first-N-seconds assumption)",
            phase="analyze",
            decision="Only high-naturality windows will update NATURAL profile.",
        )

    intra_session = None
    if config.ENABLE_INTRA_INDIVIDUAL:
        from engine.intra_individual import IntraIndividualSession

        intra_session = IntraIndividualSession.from_profile(profile)
        log.log(
            "personal_baseline",
            "Intra-individual modeling active — deviation from personal baseline",
            phase="analyze",
            metrics=intra_session.baseline.summary(),
        )

    use_kaggle_segment = gpu_client.offload_segmentation_active
    if use_kaggle_segment:
        log.log(
            "kaggle_segment",
            f"Kaggle segmentation ({config.KAGGLE_SEGMENT_MODE}) — local pyannote skipped",
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
    av_msg = (
        f"Kaggle {config.KAGGLE_SEGMENT_MODE} segment + local windows..."
        if use_kaggle_segment
        else "Processing interview audio..."
    )
    _emit(progress, 12, av_msg)

    def _run_audio() -> list[dict[str, Any]]:
        if not use_kaggle_segment:
            log.log(
                "audio",
                "Local pyannote diarization (recommended for AI vs candidate accuracy)",
                phase="analyze",
                metrics={
                    "candidate_speaker": config.CANDIDATE_SPEAKER,
                    "num_speakers": config.DIARIZATION_NUM_SPEAKERS,
                    "min_candidate_segment_sec": config.MIN_CANDIDATE_SEGMENT_SEC,
                },
            )
            return audio.process_interview(str(video))

        wav_path = audio.extract_audio(str(video))
        try:
            t_seg = time.perf_counter()
            segment_payload = gpu_client.segment_interview(str(video), wav_path=wav_path)
            seg_sec = round(time.perf_counter() - t_seg, 2)
            answers_payload = (segment_payload or {}).get("answers") or []

            if not answers_payload:
                seg_err = (segment_payload or {}).get("error") if segment_payload else None
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
                return audio.process_interview(str(video))

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
                },
                decision=f"candidate_track={config.CANDIDATE_SPEAKER}",
            )
            return audio.process_interview_from_segmentation(
                str(video),
                answers_payload,
                speaker_selection=(segment_payload or {}).get("speaker_selection"),
                wav_path=wav_path,
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def _run_video() -> dict[str, Any]:
        # Video gaze/lip scanning removed (performance + robustness).
        return {"timeline_path": "", "timeline": [], "native_fps": 0.0}

    t_av = time.perf_counter()
    interview_answers = _run_audio()
    video_result = _run_video()
    av_sec = round(time.perf_counter() - t_av, 2)

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
        _emit(progress, 28, "Kaggle GPU transcription (parallel)...")
        kaggle_cache = _kaggle_prefetch_answers(
            gpu_client,
            interview_answers,
            gpu_reading_profile,
            log,
        )
    elif interview_answers:
        _emit(progress, 28, f"Local Whisper transcription ({config.WHISPER_MODEL_SIZE})...")
        t_asr = time.perf_counter()
        local_transcript_cache = transcript_proc.transcribe_answers(interview_answers)
        asr_sec = round(time.perf_counter() - t_asr, 2)
        log.log(
            "asr",
            f"Local interview transcription batch ({asr_sec}s)",
            phase="analyze",
            metrics={
                "answers": len(interview_answers),
                "elapsed_sec": asr_sec,
                "skip_align": config.WHISPER_SKIP_ALIGN_INTERVIEW,
                "model": config.WHISPER_MODEL_SIZE,
            },
        )

    results_answers: list[dict] = []
    n_answers = max(len(interview_answers), 1)

    for idx, answer in enumerate(interview_answers):
        pct = 30 + int(55 * (idx / n_answers))
        aid = answer["answer_id"]
        _emit(progress, pct, f"Scoring answer {aid + 1}/{n_answers}...")

        ac_score, ac_breakdown = engine.score_answer(answer["windows"], reading_profile)
        duration = float(answer["end_sec"]) - float(answer["start_sec"])

        gpu: float | None = None
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

    _emit(progress, 100, "Analysis complete", None)

    payload["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    return payload
