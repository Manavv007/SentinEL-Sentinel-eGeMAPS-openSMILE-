# SentinEL — Tech Stack (What We Used for What)

A plain map of **technologies → role in the project**. Use this when presenting the system or onboarding someone new.

---

## At a glance

| Layer | Main technologies |
|-------|-------------------|
| Language & config | Python 3.10, `python-dotenv`, `config.py` |
| Media I/O | FFmpeg, `soundfile`, `librosa` |
| Who spoke when | **pyannote.audio** (speaker diarization) |
| Speech-to-text | **WhisperX** (local CPU); optional **Kaggle GPU** + `large-v3` |
| Voice features | **openSMILE** (eGeMAPS-style), **Parselmouth** (Praat) |
| Scoring & fusion | NumPy, SciPy, custom Python engines (contrastive, linguistic, semantic) |
| Web app | **FastAPI**, **Uvicorn**, HTML/CSS/JS |
| Optional GPU offload | **Kaggle notebook**, **ngrok**, **httpx** |
| Video (optional / mostly off) | OpenCV, MediaPipe |

---

## Core runtime

| Technology | Used for |
|------------|----------|
| **Python 3.10** | Entire backend — calibration, analyze pipeline, scoring, web API |
| **python-dotenv** | Load secrets and settings from `.env` (`HF_TOKEN`, Kaggle URL, thresholds) |
| **NumPy** | Feature vectors, similarities, EWMA, profile statistics |
| **SciPy** | Supporting math where needed in signal processing |

---

## Input: video and audio

| Technology | Used for |
|------------|----------|
| **FFmpeg** (`ffmpeg-python`) | Extract mono **16 kHz** WAV from interview/calibration video |
| **soundfile** | Read/write WAV; pass in-memory audio to pyannote on Windows |
| **librosa** | Audio utilities in the processing chain |

**Why 16 kHz mono?** Standard for speech models (Whisper, pyannote) and smaller/faster processing.

---

## Speaker diarization (candidate vs interviewer)

| Technology | Used for |
|------------|----------|
| **pyannote.audio 3.1** | Split interview audio into **who spoke when** (SPEAKER_00, SPEAKER_01, …) |
| **HuggingFace Hub** (`HF_TOKEN`) | Download pyannote model weights (one-time; terms must be accepted) |
| **PyTorch + torchaudio** | Backend for pyannote and WhisperX |
| **Custom `speaker_selection.py`** | Choose which speaker is the **human candidate** (not the AI interviewer) |

**Runs:** Local CPU by default; optional same logic on **Kaggle GPU** when `KAGGLE_OFFLOAD_SEGMENTATION=true`.

**Without this:** You would score the AI bot’s questions as “candidate answers.”

---

## Speech-to-text (transcription)

| Technology | Used for |
|------------|----------|
| **WhisperX** | Transcribe each answer segment; word-level timestamps when alignment is enabled |
| **OpenAI Whisper** (via WhisperX, `small` local / `large-v3` on Kaggle) | Actual ASR model |
| **whisper-timestamped** (optional) | Fallback when WhisperX strips fillers (`um`, `uh`) — better spontaneity signals |
| **faster-whisper / ctranslate2** | Engine under WhisperX for fast CPU inference (`int8` on local) |

| Mode | Typical setup |
|------|----------------|
| **Local CPU** | `WHISPER_MODEL_SIZE=small`, `int8` — ~minutes per interview |
| **Kaggle GPU** | `large-v3` on GPU via HTTP — faster, higher quality |
| **Fast calibration** | `tiny` Whisper — quick reading-video transcription only |

**Output:** `transcript` text + `words[]` with `start`/`end` for linguistic features.

---

## Acoustic / voice-quality features

| Technology | Used for |
|------------|----------|
| **openSMILE** | eGeMAPS-style features per **4 s window** (pitch variability, voiced/unvoiced segment lengths) |
| **Parselmouth** (Praat) | Jitter, shimmer, HNR, pitch range — “reading vs natural” voice quality |
| **Custom `AnalysisEngine`** | Build **reading profile** from calibration; score interview windows vs that profile |

**Windowing:** 4 s windows, 2 s hop — matches contrastive engine and profile memory.

---

## Linguistic & cognitive signals (from transcript + timing)

| Technology | Used for |
|------------|----------|
| **Custom `LinguisticAnalyzer`** | Words-per-second, pause gaps, pause entropy, fillers, self-corrections |
| **NLTK** (Kaggle server) | Tokenization/POS on remote server where needed |
| **Rule-based Python** (`semantic_specificity.py`) | Personal narrative vs memorized technical prose; generic essay detection |
| **Custom engines** | Cognitive spontaneity, sourcing inference, answer synthesis, intra-individual baseline |

**Optional LLM judge** (`engine/llm_judge.py`, off by default): when `ENABLE_LLM_JUDGE=true`, runs only on **AMBIGUOUS** answers after all rule-based layers. Pluggable providers (`openai`, `openrouter`, `anthropic`, `ollama` via `httpx`). OpenRouter works with models like `openai/gpt-oss-120b`. Primary NLP remains **rule-based** in `semantic_specificity.py`.

---

## Dual-profile contrastive scoring (core ML design)

| Component | Used for |
|-----------|----------|
| **`ContrastiveEngine`** | Compare each window to **SCRIPT** (calibration reading) vs **NATURAL** (learned spontaneous) profiles |
| **`BehavioralProfile` / `profile_memory`** | Store and update speaker behavior fingerprints |
| **`temporal_evidence` + EWMA** | Smooth suspicion over time; tiered STRONG/WEAK windows |
| **`FusedScorer`** | Weight acoustic + linguistic + specificity (+ optional GPU channel) |

**Not a single end-to-end neural classifier** — hybrid of profiles, similarities, and explicit decision layers.

---

## Web application & CLI

| Technology | Used for |
|------------|----------|
| **FastAPI** | REST API: upload video, start jobs, poll progress, fetch results |
| **Uvicorn** | ASGI server (`run_web.ps1` → `http://127.0.0.1:8765`) |
| **python-multipart** | File uploads (calibration + interview video) |
| **HTML / CSS / JavaScript** (`web/static/`) | Dashboard: progress, decision log, timeline chart, per-answer cards |
| **`main.py` CLI** | `calibrate`, `analyze`, `report` without browser |

**Same pipeline:** `services/pipeline.py` powers both CLI and web.

---

## Optional: Kaggle GPU offload

| Technology | Used for |
|------------|----------|
| **Kaggle Notebook** (GPU T4) | Run heavy Whisper + optional diarization + Parselmouth scoring remotely |
| **`kaggle_gpu_server.ipynb`** | FastAPI server inside notebook |
| **ngrok** | Public HTTPS tunnel so local machine can call Kaggle |
| **httpx** (`gpu_client.py`) | Local client: `/health`, `/transcribe_answer`, `/segment_interview`, `/calibrate` |
| **Shared secret** (`SENTINEL_SECRET`) | Auth between local SentinEL and remote server |

**What stays local:** Orchestration, contrastive engine, semantic specificity, intra-individual logic, results JSON, web UI.

**What can move to Kaggle:** Transcription (main win), optional segmentation, optional GPU acoustic channel.

---

## Video / visual modality (limited in current build)

| Technology | Used for |
|------------|----------|
| **OpenCV** | Frame read when video timeline is built |
| **MediaPipe** | Face mesh — gaze/lip features (designed for reading detection) |

**Current status:** Gaze/lip path is **mostly disabled** in analyze for speed and stability. Scoring runs on **audio + text**; gaze/lip often `null` in results.

---

## Configuration & artifacts

| Item | Used for |
|------|----------|
| **`.env`** | Tokens, Kaggle URL, Whisper size, speaker strategy, feature flags |
| **`calibration_profile.json`** | SCRIPT profile, acoustic baseline, personal baseline, linguistic calibration |
| **`results.json`** | Per-answer verdicts, window logs, decision explanations |

---

## What we deliberately did **not** use

| Not used | Why |
|----------|-----|
| **LLM as primary verdict** | Rules + profiles stay authoritative; optional LLM is tie-breaker on AMBIGUOUS only |
| **Docker / Kubernetes** | Local-first dev; Kaggle as optional remote GPU |
| **Cloud-hosted app (Vercel, etc.)** | Batch video analysis needs long CPU/GPU jobs, not serverless HTTP |
| **Dedicated database** | Jobs and JSON files on disk (`web_data/jobs/`) |
| **Real-time streaming** | Offline batch: upload → analyze → report |

---

## Typical data flow (tech order)

```
Video (.webm / .mp4)
    → FFmpeg (audio extract)
    → pyannote (diarization)
    → speaker_selection (pick candidate)
    → openSMILE + Parselmouth (per-window acoustics)
    → WhisperX or Kaggle Whisper (transcription)
    → LinguisticAnalyzer + semantic_specificity (text)
    → ContrastiveEngine + fused_scorer + session layers
    → results.json + web UI
```

---

## Environment requirements summary

| Requirement | Purpose |
|-------------|---------|
| Python **3.10** | Compatible stack with pinned deps |
| **FFmpeg** on PATH | Video → audio |
| **HF_TOKEN** | pyannote model access |
| **HuggingFace model terms** | speaker-diarization-3.1, segmentation, embedding |
| **~10+ min CPU** per long interview | Diarization + Whisper on CPU without Kaggle |
| **Kaggle + ngrok** (optional) | Faster/better ASR and GPU segmentation |

---

## Related docs

| File | Contents |
|------|----------|
| `README.md` | Install and run |
| `explanation.md` | Full architecture and workflow |
| `problem.md` | General problems a new builder will face |
| `problems.md` | Technical issues we hit in this repo |
| `.env.example` | All tunable settings |

---

*Last updated for SentinEL dual-profile v5, local CPU + optional Kaggle GPU path.*
