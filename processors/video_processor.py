"""Video processing: OpenCV + MediaPipe FaceMesh gaze/lip timeline extraction."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

import config

logger = logging.getLogger(__name__)

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
except ImportError:
    mp_tasks = None  # type: ignore[assignment]
    vision = None  # type: ignore[assignment]

# Iris (refine_landmarks=True)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_LEFT_CORNER = 263
LEFT_EYE_RIGHT_CORNER = 362
RIGHT_EYE_LEFT_CORNER = 33
RIGHT_EYE_RIGHT_CORNER = 133

# Eyelids (EAR)
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Lips
LIP_TOP_CENTER = 13
LIP_BOTTOM_CENTER = 14
LIP_LEFT_CORNER = 61
LIP_RIGHT_CORNER = 291
UPPER_LIP_TOP = 0
LOWER_LIP_BOTTOM = 17

TIMELINE_FPS = 10.0
BUCKET_SEC = 1.0 / TIMELINE_FPS
TIMELINE_FILENAME = "gaze_lip_timeline.json"

_NUMERIC_KEYS = (
    "gaze_x_ratio",
    "gaze_y_ratio",
    "ear",
    "lip_aperture",
    "lip_width",
    "lip_left_disp",
    "lip_right_disp",
)


class VideoProcessor:
    """Extract per-frame gaze/lip features and save a downsampled timeline."""

    def __init__(self) -> None:
        self._landmarker: Any = None
        self._opencv_fallback = False

    @staticmethod
    def _ensure_landmarker_model() -> Path:
        model_path = Path(__file__).resolve().parent / "face_landmarker.task"
        if model_path.is_file():
            return model_path
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)
        return model_path

    def _get_face_landmarker(self) -> Any:
        if self._opencv_fallback:
            return None
        if self._landmarker is not None:
            return self._landmarker
        if vision is None or mp_tasks is None:
            self._opencv_fallback = True
            return None
        try:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(self._ensure_landmarker_model())
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            logger.warning(
                "MediaPipe FaceLandmarker unavailable (%s). Using OpenCV fallback.",
                exc,
            )
            self._opencv_fallback = True
            return None
        return self._landmarker

    def process_video(
        self,
        video_path: str,
        *,
        timeline_path: str | Path | None = None,
        timeline_fps: float | None = None,
    ) -> dict[str, Any]:
        """
        Process video and build a downsampled gaze/lip timeline.

        Returns dict with keys: timeline_path, timeline, native_fps.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        out_path = Path(timeline_path) if timeline_path else path.parent / TIMELINE_FILENAME
        target_fps = float(timeline_fps or config.VIDEO_TIMELINE_FPS)
        bucket_sec = 1.0 / max(target_fps, 1.0)
        native_fps = self._read_fps(str(path))

        native_frames = self._extract_native_frames(str(path), sample_fps=target_fps)
        timeline = self._downsample_timeline(
            native_frames, native_fps=native_fps, bucket_sec=bucket_sec
        )

        payload = {
            "video": str(path.resolve()),
            "native_fps": native_fps,
            "timeline_fps": target_fps,
            "frames": timeline,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return {
            "timeline_path": str(out_path),
            "timeline": timeline,
            "native_fps": payload["native_fps"],
        }

    @staticmethod
    def load_timeline(timeline_path: str | Path) -> list[dict[str, Any]]:
        data = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
        return data.get("frames", data if isinstance(data, list) else [])

    @staticmethod
    def slice_timeline(
        timeline: list[dict[str, Any]],
        start_sec: float,
        end_sec: float,
    ) -> list[dict[str, Any]]:
        return [
            f
            for f in timeline
            if start_sec <= float(f["timestamp_sec"]) < end_sec and f.get("face_detected")
        ]

    def close(self) -> None:
        if self._landmarker is not None and hasattr(self._landmarker, "close"):
            self._landmarker.close()
            self._landmarker = None

    def _read_fps(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return float(fps)

    def _extract_native_frames(
        self,
        video_path: str,
        *,
        sample_fps: float | None = None,
    ) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = 1
        if sample_fps and sample_fps > 0 and fps > sample_fps:
            stride = max(1, int(round(fps / sample_fps)))

        landmarker = self._get_face_landmarker()
        frames: list[dict[str, Any]] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            timestamp_sec = frame_idx / fps

            if landmarker is None:
                frames.append(self._opencv_fallback_frame(frame, timestamp_sec))
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                timestamp_ms = int(timestamp_sec * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if not result.face_landmarks:
                    frames.append(self._empty_frame(timestamp_sec))
                else:
                    frames.append(
                        self._frame_features(result.face_landmarks[0], timestamp_sec)
                    )

            frame_idx += 1

        cap.release()
        return frames

    @staticmethod
    def _opencv_fallback_frame(frame: np.ndarray, timestamp_sec: float) -> dict[str, Any]:
        """Degraded features when MediaPipe is unavailable (e.g. Python 3.13 on Windows)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return {
            "timestamp_sec": timestamp_sec,
            "gaze_x_ratio": float(gray.mean()) / 255.0,
            "gaze_y_ratio": float(gray.std()) / 128.0,
            "ear": 0.25,
            "lip_aperture": float(gray.std()) / 128.0,
            "lip_width": 0.2,
            "lip_left_disp": 0.05,
            "lip_right_disp": 0.05,
            "face_detected": True,
        }

    @staticmethod
    def _empty_frame(timestamp_sec: float) -> dict[str, Any]:
        return {
            "timestamp_sec": timestamp_sec,
            "gaze_x_ratio": 0.0,
            "gaze_y_ratio": 0.0,
            "ear": 0.0,
            "lip_aperture": 0.0,
            "lip_width": 0.0,
            "lip_left_disp": 0.0,
            "lip_right_disp": 0.0,
            "face_detected": False,
        }

    def _frame_features(
        self,
        lm: list,
        timestamp_sec: float,
    ) -> dict[str, Any]:
        def nx(idx: int) -> float:
            return lm[idx].x

        def ny(idx: int) -> float:
            return lm[idx].y

        face_height = max(
            abs(ny(LOWER_LIP_BOTTOM) - ny(UPPER_LIP_TOP)),
            1e-6,
        )

        left_gaze_x = self._gaze_x_ratio(
            LEFT_IRIS_CENTER, LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER, lm
        )
        right_gaze_x = self._gaze_x_ratio(
            RIGHT_IRIS_CENTER, RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER, lm
        )
        left_gaze_y = self._gaze_y_ratio(
            LEFT_IRIS_CENTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, lm
        )
        right_gaze_y = self._gaze_y_ratio(
            RIGHT_IRIS_CENTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, lm
        )

        ear_left = self._ear(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER, lm)
        ear_right = self._ear(
            RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER, lm
        )

        lip_top = ny(LIP_TOP_CENTER)
        lip_bottom = ny(LIP_BOTTOM_CENTER)
        lip_left = nx(LIP_LEFT_CORNER)
        lip_right = nx(LIP_RIGHT_CORNER)

        face_center_x = (nx(LEFT_EYE_LEFT_CORNER) + nx(RIGHT_EYE_RIGHT_CORNER)) / 2.0
        lip_left_disp = abs(lip_left - face_center_x)
        lip_right_disp = abs(lip_right - face_center_x)

        return {
            "timestamp_sec": round(timestamp_sec, 6),
            "gaze_x_ratio": float((left_gaze_x + right_gaze_x) / 2.0),
            "gaze_y_ratio": float((left_gaze_y + right_gaze_y) / 2.0),
            "ear": float((ear_left + ear_right) / 2.0),
            "lip_aperture": float(abs(lip_bottom - lip_top) / face_height),
            "lip_width": float(abs(lip_right - lip_left) / face_height),
            "lip_left_disp": float(lip_left_disp),
            "lip_right_disp": float(lip_right_disp),
            "face_detected": True,
        }

    @staticmethod
    def _gaze_x_ratio(iris: int, eye_left: int, eye_right: int, lm: list) -> float:
        iris_x = lm[iris].x
        left_x = lm[eye_left].x
        right_x = lm[eye_right].x
        eye_width = right_x - left_x
        if abs(eye_width) < 1e-6:
            return 0.5
        return (iris_x - left_x) / eye_width

    @staticmethod
    def _gaze_y_ratio(iris: int, eye_top: int, eye_bottom: int, lm: list) -> float:
        iris_y = lm[iris].y
        top_y = lm[eye_top].y
        bottom_y = lm[eye_bottom].y
        eye_height = bottom_y - top_y
        if abs(eye_height) < 1e-6:
            return 0.5
        return (iris_y - top_y) / eye_height

    @staticmethod
    def _ear(top: int, bottom: int, left: int, right: int, lm: list) -> float:
        top_y = lm[top].y
        bottom_y = lm[bottom].y
        left_x = lm[left].x
        right_x = lm[right].x
        width = right_x - left_x
        if abs(width) < 1e-6:
            return 0.0
        return (top_y - bottom_y) / width

    @staticmethod
    def _downsample_timeline(
        native_frames: list[dict[str, Any]],
        *,
        native_fps: float,
        bucket_sec: float = BUCKET_SEC,
    ) -> list[dict[str, Any]]:
        if not native_frames:
            return []

        max_t = native_frames[-1]["timestamp_sec"]
        buckets: dict[int, list[dict[str, Any]]] = {}

        for frame in native_frames:
            bucket_idx = int(frame["timestamp_sec"] / bucket_sec)
            buckets.setdefault(bucket_idx, []).append(frame)

        timeline: list[dict[str, Any]] = []
        last_bucket = int(max_t / bucket_sec) + 1

        for b in range(last_bucket + 1):
            group = buckets.get(b, [])
            if not group:
                continue
            detected = [g for g in group if g.get("face_detected")]
            if not detected:
                timeline.append(
                    {
                        "timestamp_sec": round(b * bucket_sec, 4),
                        "gaze_x_ratio": 0.0,
                        "gaze_y_ratio": 0.0,
                        "ear": 0.0,
                        "lip_aperture": 0.0,
                        "lip_width": 0.0,
                        "lip_left_disp": 0.0,
                        "lip_right_disp": 0.0,
                        "face_detected": False,
                    }
                )
                continue

            averaged: dict[str, Any] = {"timestamp_sec": round(b * bucket_sec, 4)}
            for key in _NUMERIC_KEYS:
                averaged[key] = round(
                    float(np.mean([float(d[key]) for d in detected])), 6
                )
            averaged["face_detected"] = True
            timeline.append(averaged)

        return timeline
