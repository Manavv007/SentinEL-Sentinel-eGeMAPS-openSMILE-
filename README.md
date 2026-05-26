# SentinEL

Multi-modal interview integrity analysis: acoustic, linguistic, gaze, and lip signals are fused to flag probable script reading during answers.

## Installation (Local CPU Mode)

### Requirements

- Python 3.10 (not 3.11+, not 3.9)
- ffmpeg:
  - Windows: https://www.gyan.dev/ffmpeg/builds/ → add to PATH
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

### Install dependencies (ORDER MATTERS)

**Step 1 — CPU-only PyTorch** (saves ~2GB vs CUDA version):

```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

**Step 2 — WhisperX:**

```bash
pip install whisperx==3.1.5
```

**Step 3 — Everything else:**

```bash
pip install -r requirements.txt
```

### First-time setup

1. Copy `.env.example` to `.env` and add your HuggingFace token  
   (required once — downloads pyannote model weights on first run, then cached locally)

2. Accept pyannote model terms on HuggingFace (one-time, required):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

### Run

**Calibrate** (reading paragraph video):

```bash
python main.py calibrate --video caliberation_file/pre-train-demo.mp4
```

**Analyze** (interview video):

```bash
python main.py analyze --video interview_files/demo-8.webm --calibration calibration_profile.json
```

**Report:**

```bash
python main.py report --results results.json
```

### Web UI (recommended)

Upload calibration and interview videos in the browser, watch progress, and explore charts + decision logs.

```powershell
pip install fastapi uvicorn python-multipart
.\run_web.ps1
```

**Important:** Always start the UI with `.\run_web.ps1` — do **not** use `python -m uvicorn` directly unless that same Python has `whisperx` installed. The launcher auto-picks a Python where `whisperx` imports successfully.

Open **http://127.0.0.1:8765**

1. **Calibrate** — upload reading video → builds SCRIPT profile  
2. **Analyze** — upload interview + pick calibration job (or upload `.json`)  
3. **Results** — timeline chart, per-answer scores, full decision log with every metric

### Faster calibration (30s video)

With default **fast calibration** (see `.env.example`), a 30s reading clip is typically **~20–40 seconds** instead of ~2 minutes:

| Optimization | Effect |
|--------------|--------|
| Skip pyannote diarization | Single-speaker reading — no speaker-ID model |
| Whisper `tiny` for calibration | Faster ASR; interview still uses `WHISPER_MODEL_SIZE` |
| Skip word alignment on calibrate | Saves wav2vec pass |
| Video at 5 fps | Fewer face-mesh frames |
| Parallel audio + video | Both run at once |
| Model preload on web start | First job skips cold Whisper load |

Set `FAST_CALIBRATION=false` in `.env` for maximum calibration quality (slower).

### Expected processing times (CPU, small int8 model)

| Step              | Time for 1hr interview |
|-------------------|------------------------|
| Video (MediaPipe) | ~2 min                 |
| Diarization       | ~3 min                 |
| Transcription     | ~4 min (all answers)   |
| Acoustics         | ~1 min                 |
| Scoring           | <1 sec                 |
| **Total**         | **~10 min**            |

### Optional: filler-preserving fallback

If WhisperX strips `um`/`uh`, install:

```bash
pip install whisper-timestamped
```

### Optional: Kaggle GPU offload (Whisper large-v3 + GPU scoring)

Local CPU mode works without Kaggle. For faster transcription and the **GPU fusion channel**:

1. **Kaggle notebook** (GPU T4 x2 recommended)
   - Upload `kaggle_gpu_server.ipynb` to Kaggle (or copy cells into a new notebook).
   - **Settings → Accelerator → GPU**.
   - Add secrets if needed: `HF_TOKEN` (HuggingFace, for pyannote/align models).
   - Run **Cell 1** (install + kernel restart), then **Cell 2** (loads WhisperX, starts server + ngrok).
   - Copy the printed URL, e.g. `https://xxxx.ngrok-free.dev`.

2. **Local `.env`** (must match the notebook secret):

```env
KAGGLE_GPU_URL=https://xxxx.ngrok-free.dev
KAGGLE_SECRET=sentinEL2026
SENTINEL_SECRET=sentinEL2026
```

3. **Verify connection** (from project root):

```bash
pip install httpx
python scripts/test_kaggle_gpu.py
```

4. **Calibrate then analyze** as usual (`restart_web.ps1` or CLI). Calibration calls Kaggle `/calibrate` and saves `gpu_reading_profile` into `calibration_profile.json`. Interview analysis calls `/analyze_batch` per answer for the GPU score channel.

**Notes**

- ngrok URL **changes every time** you restart the Kaggle notebook — update `.env` each session.
- Keep the Kaggle notebook **running** while analyzing locally.
- Re-calibrate after enabling GPU so `gpu_reading_profile` exists (older profiles only have CPU openSMILE baselines).
- Local Whisper can stay `small` on CPU; Kaggle runs `large-v3` for the GPU path only.

## Dual-profile contrastive engine (v5)

Calibration video = **intentional script reading**. The system builds a **SCRIPT profile** from that video (how this user sounds while reading).

During the interview, a **NATURAL profile** is built **only** from windows with high naturality confidence. The first N seconds are **never** assumed natural (no baseline poisoning if the candidate reads from second 1).

Per 4s window:

| Signal | Meaning |
|--------|---------|
| `script_similarity` | Similarity to calibration reading behavior |
| `natural_similarity` | Similarity to opportunistically learned spontaneous behavior |
| `contrastive_score` | `script_similarity - natural_similarity` (primary) |
| `naturality_score` | Cognitive spontaneity estimate (gates NATURAL profile updates) |

Alert when contrastive EWMA exceeds `CONTRASTIVE_MARGIN` with temporal persistence. Confidence: `LOW` / `MEDIUM` / `HIGH`.

`results.json` includes `window_logs` (per-window debug) when contrastive mode is on. Re-run **calibrate** to produce a v5 `script_profile` if you have an older v4 profile.

Toggle in `.env`: `ENABLE_CONTRASTIVE_ENGINE`, `CONTRASTIVE_MARGIN`, `NATURALITY_UPDATE_THRESHOLD`, etc.

### AI interviewer + human candidate

Pyannote labels speakers anonymously. Choose who is scored as the **candidate** via `CANDIDATE_SPEAKER` in `.env`:

| Value | Use when |
|-------|----------|
| `most_speech` | Candidate talks the most (legacy default) |
| `least_speech` | AI interviewer talks more total time than the human |
| `longest_turns` | AI asks short prompts; candidate gives longer answers (recommended) |

Set `CANDIDATE_TURN_MIN_SEC=3` to ignore short AI question bursts when using `longest_turns`.

## Project layout

```
├── config.py
├── gpu_client.py              # CPU no-op stub (GPU optional later)
├── main.py
├── processors/
│   ├── audio_processor.py
│   ├── video_processor.py
│   └── transcript_processor.py
├── engine/
│   ├── analysis_engine.py
│   ├── contrastive_engine.py  # dual-profile orchestrator
│   ├── profile_memory.py      # SCRIPT / NATURAL profiles
│   ├── feature_extraction.py
│   ├── naturality_scorer.py
│   ├── transition_detector.py
│   ├── temporal_evidence.py
│   ├── fused_scorer.py
│   ├── linguistic_analyzer.py
│   ├── gaze_analyzer.py
│   └── lip_analyzer.py
└── scoring/
    └── baseline.py
```

## What you do NOT need for local CPU mode

- No Kaggle account
- No ngrok account
- No ANTHROPIC_API_KEY
- No CUDA / NVIDIA GPU drivers
- No internet connection after first run (models cached locally)
- No running servers or background processes
- No Docker

The only external service used at runtime is HuggingFace model download — and only on the very first run.
