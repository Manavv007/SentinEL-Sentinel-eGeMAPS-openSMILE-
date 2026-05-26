"""Answer segment detection from interview video."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnswerSegment:
    """Time-bounded slice of an interview corresponding to one answer."""

    index: int
    start_sec: float
    end_sec: float


def detect_answer_segments(
    video_path: str,
    *,
    min_duration_sec: float = 2.0,
) -> list[AnswerSegment]:
    """
    Detect answer boundaries in an interview recording.

    Placeholder: returns a single segment spanning the full clip.
    Replace with VAD / diarization / question-boundary logic.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()

    duration = frame_count / fps if frame_count > 0 else min_duration_sec
    if duration < min_duration_sec:
        duration = min_duration_sec

    return [AnswerSegment(index=0, start_sec=0.0, end_sec=duration)]
