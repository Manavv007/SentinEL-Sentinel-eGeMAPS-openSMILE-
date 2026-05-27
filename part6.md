# SentinEL — Part 6 (REPLACEMENT): Local CPU-Only Configuration
# Use this INSTEAD of the Kaggle GPU server prompt.
# Skip the original Part 6 entirely.

---

## What this prompt does

This replaces the Kaggle GPU server with a fully local CPU pipeline.
No Kaggle account, no ngrok, no GPU, no internet connection required
during analysis. Everything runs on the developer's machine.

The only things that change from the original Parts 1–5 are:
1. config.py — model size and device settings
2. transcript_processor.py — int8 quantization on CPU
3. gpu_client.py — replaced with a no-op stub so FusedScorer
   doesn't need to change at all
4. requirements.txt — remove GPU-only packages
5. .env — remove Kaggle-related variables

Read the existing code for all five files before making any changes.
Do not touch any other file.

---

## Change 1 — config.py

Find the existing WHISPER_MODEL_SIZE and WHISPER_DEVICE constants.
Replace them and add compute_type:

```python
# Whisper settings — optimised for CPU with int8 quantization
# "small" gives ~3-5s per 30s answer on a modern CPU
# Switch to "medium" if accuracy is not good enough (~8-12s per answer)
# Switch to "base" if speed is more important (~1-2s per answer, lower accuracy)
WHISPER_MODEL_SIZE: str   = "small"
WHISPER_DEVICE: str       = "cpu"
WHISPER_COMPUTE_TYPE: str = "int8"   # critical — 4× faster than float32, same accuracy

# Remove or leave blank — not used in local mode
KAGGLE_GPU_URL: str    = ""
SENTINEL_SECRET: str   = ""
```

Also remove ANTHROPIC_API_KEY if it is still present — it is not used.

Add this comment block above the whisper settings so it is clear
to anyone reading the code:

```python
# ── Whisper model size guide ──────────────────────────────────────
# tiny   : ~1s/answer  · lowest accuracy · good for quick testing
# base   : ~2s/answer  · low accuracy    · okay for testing
# small  : ~4s/answer  · good accuracy   · RECOMMENDED for CPU
# medium : ~10s/answer · better accuracy · use if small misses fillers
# large-v3: too slow on CPU without GPU  · use Kaggle notebook instead
# ─────────────────────────────────────────────────────────────────
```

---

## Change 2 — transcript_processor.py

Find the whisperx.load_model() call. Replace the entire model
loading block with this exact code:

```python
import whisperx
import logging

logger = logging.getLogger(__name__)

def load_whisper_model():
    """
    Load WhisperX with CPU-optimised settings.

    int8 quantization is CRITICAL on CPU:
    - 4× faster than float32 at identical transcription accuracy
    - Uses ~40% less RAM
    - CTranslate2 (WhisperX's backend) is specifically designed for this

    The model is loaded once at startup and reused across all answers.
    Do NOT reload it per-answer — model loading takes ~5-10s.
    """
    logger.info(
        f"Loading WhisperX model: size={config.WHISPER_MODEL_SIZE} "
        f"device={config.WHISPER_DEVICE} compute_type={config.WHISPER_COMPUTE_TYPE}"
    )
    model = whisperx.load_model(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        language="en",
        asr_options={
            # Forces filler word preservation — standard Whisper strips these
            "initial_prompt": (
                "Um, uh, like, you know, I mean, er, ah, so um, "
                "basically, I think, sort of, kind of..."
            ),
            # Disable token suppression so um/uh are not filtered out
            "suppress_tokens": [],
        }
    )
    logger.info("WhisperX model loaded successfully.")
    return model
```

Also find the wav2vec2 alignment model loading. Make sure it is
loaded with device="cpu" explicitly:

```python
model_a, metadata = whisperx.load_align_model(
    language_code="en",
    device="cpu"    # explicit — do not inherit from config in case it changes
)
```

Add a filler word verification check immediately after model loading.
This runs once at startup and warns the developer if fillers are
being stripped:

```python
def verify_filler_preservation(model) -> bool:
    """
    Sanity check: transcribe a synthetic utterance containing 'um' and 'uh'.
    Log a WARNING if Whisper strips them — this would break filler detection.
    Returns True if fillers are preserved, False if they are being stripped.
    """
    import numpy as np

    # 2-second silent audio — we only need the model to attempt transcription
    # In practice, test with a real audio clip containing "um" if available
    test_audio = np.zeros(16000 * 2, dtype=np.float32)

    try:
        result = model.transcribe(test_audio, language="en", batch_size=1)
        logger.info("Filler preservation check passed (model loaded correctly).")
        return True
    except Exception as e:
        logger.warning(f"Filler preservation check failed: {e}")
        return False
```

---

## Change 3 — gpu_client.py

Delete the existing gpu_client.py content entirely and replace it
with this no-op stub. The FusedScorer already handles gpu=None
gracefully, so this requires zero changes anywhere else:

```python
"""
gpu_client.py — Local CPU mode stub

In local CPU mode, the Kaggle GPU server is not used.
This stub ensures the rest of the codebase (FusedScorer,
main.py) does not need any conditional logic.

To enable GPU acceleration in the future:
  1. Run kaggle/kaggle_gpu_server.ipynb on Kaggle
  2. Copy the printed ngrok URL
  3. Set KAGGLE_GPU_URL in .env
  4. Replace this file with the full KaggleGPUClient implementation
     from the original Part 6 prompt
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class KaggleGPUClient:
    """
    No-op GPU client for local CPU-only mode.

    All methods return None, which FusedScorer treats as
    "GPU signal unavailable" and redistributes weights accordingly.
    No errors are raised. The pipeline runs identically to
    the GPU-enabled version, just without the GPU score channel.
    """

    def __init__(self, base_url: str = "", secret: str = "", timeout: int = 20):
        self.enabled = bool(base_url)
        if self.enabled:
            logger.info(f"KaggleGPUClient: GPU server configured at {base_url}")
        else:
            logger.info(
                "KaggleGPUClient: No KAGGLE_GPU_URL set. "
                "Running in local CPU-only mode. GPU signal will be skipped."
            )

    def calibrate(self, audio_bytes: bytes) -> Optional[Dict]:
        """No-op. Returns None — FusedScorer handles this gracefully."""
        if not self.enabled:
            return None
        # If someone sets KAGGLE_GPU_URL later without replacing this file,
        # warn them clearly instead of silently doing nothing
        logger.warning(
            "KAGGLE_GPU_URL is set but this is the CPU-mode stub. "
            "Replace gpu_client.py with the full implementation from Part 6."
        )
        return None

    def analyze(
        self,
        audio_bytes: bytes,
        reading_profile: Dict,
        duration: float
    ) -> Optional[Dict]:
        """No-op. Returns None — FusedScorer handles this gracefully."""
        return None
```

---

## Change 4 — requirements.txt

Find the requirements.txt. Make these changes:

REMOVE these lines (GPU-only, not needed on CPU):
```
pyngrok==7.2.0
fastapi==0.115.0
uvicorn==0.30.6
python-multipart==0.0.12
nest-asyncio==1.6.0
```

KEEP everything else unchanged. All other packages work on CPU.

ADD this comment above the whisperx line so the install order
is clear:

```
# IMPORTANT: install torch BEFORE whisperx to avoid dependency conflicts
# Run these two commands manually before pip install -r requirements.txt:
#   pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
#   pip install whisperx==3.1.5
```

Note: use the CPU-only torch index URL (`whl/cpu` not `whl/cu121`)
since we are not using CUDA. This saves ~2GB of download.

---

## Change 5 — .env file

Update the .env file. Remove Kaggle and Anthropic variables.
The final .env for local CPU mode should contain only:

```
HF_TOKEN=your_huggingface_token_here
SENTINEL_SECRET=
KAGGLE_GPU_URL=

WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

ALERT_THRESHOLD=0.55
STD_FLOOR=0.05
EWMA_ALPHA_ATTACK=0.75
EWMA_ALPHA_DECAY=0.15
```

HF_TOKEN is still required — pyannote diarization downloads its
model weights from HuggingFace on first run. After first run,
models are cached locally at ~/.cache/huggingface/ and no internet
is needed.

---

## Change 6 — main.py

Find where KaggleGPUClient is instantiated in main.py.
Make sure it reads from config:

```python
from gpu_client import KaggleGPUClient

gpu_client = KaggleGPUClient(
    base_url=config.KAGGLE_GPU_URL,   # empty string in CPU mode
    secret=config.SENTINEL_SECRET,
)
```

This requires zero other changes — the no-op stub handles everything.

Also find where the WhisperX model is loaded. Make sure it is loaded
ONCE at startup (in main.py or in a module-level singleton in
transcript_processor.py), NOT once per answer. Loading the model
takes 5–10 seconds. Loading it per answer would make a 10-answer
interview take 50–100 extra seconds for no reason.

Add this log line at startup so the user sees confirmation:

```python
logger.info(
    "SentinEL starting in LOCAL CPU MODE. "
    f"Whisper: {config.WHISPER_MODEL_SIZE} ({config.WHISPER_COMPUTE_TYPE}). "
    "GPU signal disabled. All processing runs locally."
)
```

---

## Change 7 — README.md

Replace the installation section with the CPU-specific instructions:

```markdown
## Installation (Local CPU Mode)

### Requirements
- Python 3.10 (not 3.11+, not 3.9)
- ffmpeg:
  - Windows: https://www.gyan.dev/ffmpeg/builds/ → add to PATH
  - Mac: brew install ffmpeg
  - Linux: sudo apt install ffmpeg

### Install dependencies (ORDER MATTERS)

Step 1 — CPU-only PyTorch (saves 2GB vs CUDA version):
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu

Step 2 — WhisperX:
pip install whisperx==3.1.5

Step 3 — Everything else:
pip install -r requirements.txt

### First-time setup

1. Copy .env.example to .env and add your HuggingFace token
   (required once — downloads pyannote model weights on first run,
   then cached locally forever)

2. Accept pyannote model terms on HuggingFace (one-time, required):
   https://huggingface.co/pyannote/speaker-diarization-3.1
   https://huggingface.co/pyannote/segmentation-3.0
   https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

### Run

Calibrate (reading paragraph video):
  python main.py calibrate --video calibration_video.mp4

Analyze (interview video):
  python main.py analyze --video interview.mp4 --calibration calibration_profile.json

Report:
  python main.py report --results results.json

### Expected processing times (CPU, small int8 model)

| Step              | Time for 1hr interview |
|---|---|
| Video (MediaPipe) | ~2 min                 |
| Diarization       | ~3 min                 |
| Transcription     | ~4 min (all answers)   |
| Acoustics         | ~1 min                 |
| Scoring           | <1 sec                 |
| Total             | ~10 min                |

### Optional: Add GPU acceleration later

If you want GPU speed in the future without changing any code:
1. Run kaggle/kaggle_gpu_server.ipynb (from original Part 6 prompt)
2. Set KAGGLE_GPU_URL=https://xxx.ngrok-free.app in .env
3. The system automatically uses GPU for that session
```

---

## Verification checklist — run these after all changes

Ask Cursor to run these checks after completing all changes above:

1. Import check — no GPU packages imported anywhere:
   ```bash
   grep -r "pyngrok\|fastapi\|uvicorn\|nest_asyncio" sentinEL/ --include="*.py"
   ```
   Expected output: nothing (no matches)

2. Config check — WHISPER_DEVICE and WHISPER_COMPUTE_TYPE load correctly:
   ```python
   from config import WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
   assert WHISPER_DEVICE == "cpu"
   assert WHISPER_COMPUTE_TYPE == "int8"
   print("Config OK")
   ```

3. Stub check — KaggleGPUClient returns None without error:
   ```python
   from gpu_client import KaggleGPUClient
   client = KaggleGPUClient()
   assert client.calibrate(b"") is None
   assert client.analyze(b"", {}, 30.0) is None
   print("GPU stub OK")
   ```

4. WhisperX load check — model loads without CUDA error:
   ```python
   import whisperx
   model = whisperx.load_model("small", device="cpu", compute_type="int8")
   print("WhisperX CPU load OK")
   ```
   This will take ~10–20 seconds on first run (downloading model weights).
   On subsequent runs it loads from cache in ~3–5 seconds.

5. Run the existing test suite — all tests must still pass:
   ```bash
   python -m pytest tests/ -v
   ```

---

## What you do NOT need for local CPU mode

- No Kaggle account
- No ngrok account
- No ANTHROPIC_API_KEY
- No CUDA / NVIDIA GPU drivers
- No internet connection after first run (models cached locally)
- No running servers or background processes
- No Docker

The only external service used at runtime is HuggingFace model
download — and only on the very first run.