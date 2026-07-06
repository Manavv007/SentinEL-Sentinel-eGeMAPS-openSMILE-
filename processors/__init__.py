from processors.audio_processor import AudioProcessor, ProcessingMode
from processors.transcript_processor import TranscriptProcessor

# VideoProcessor pulls in cv2/mediapipe, which are optional (gaze/lip scanning is not
# used in the calibrate/analyze paths). Import it lazily so the pipeline runs in
# audio-only / Kaggle-offload environments where those libs are not installed.
try:
    from processors.video_processor import VideoProcessor
except Exception:  # pragma: no cover - optional dependency
    VideoProcessor = None  # type: ignore[assignment]

__all__ = [
    "AudioProcessor",
    "ProcessingMode",
    "TranscriptProcessor",
    "VideoProcessor",
]
