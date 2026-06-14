# SentinEL — Complete Project Explanation

This document explains **what SentinEL is**, **how it works end-to-end**, **what technologies it uses**, and **how each major component fits together**. It is written so you can understand the system deeply and present it to others (interviews, demos, reports).

---

## Table of contents

1. [What problem does SentinEL solve?](#1-what-problem-does-sentinEL-solve)
2. [High-level idea in one minute](#2-high-level-idea-in-one-minute)
3. [System architecture](#3-system-architecture)
4. [User workflow (how you actually use it)](#4-user-workflow-how-you-actually-use-it)
5. [Technology stack](#5-technology-stack)
6. [End-to-end data flow](#6-end-to-end-data-flow)
7. [Phase 1 — Calibration (building the SCRIPT fingerprint)](#7-phase-1--calibration-building-the-script-fingerprint)
8. [Phase 2 — Interview analysis](#8-phase-2--interview-analysis)
9. [Core concept: dual-profile contrastive engine](#9-core-concept-dual-profile-contrastive-engine)
10. [Signal channels (what we measure)](#10-signal-channels-what-we-measure)
11. [Scoring layers (how a verdict is decided)](#11-scoring-layers-how-a-verdict-is-decided)
12. [Answer statuses and confidence](#12-answer-statuses-and-confidence)
13. [Semantic specificity (transcript intelligence)](#13-semantic-specificity-transcript-intelligence)
14. [Intra-individual modeling (person-relative)](#14-intra-individual-modeling-person-relative)
15. [Session-level reasoning](#15-session-level-reasoning)
16. [Kaggle GPU offload (optional)](#16-kaggle-gpu-offload-optional)
17. [Web UI and CLI](#17-web-ui-and-cli)
18. [Project folder structure](#18-project-folder-structure)
19. [Configuration (.env)](#19-configuration-env)
20. [Outputs and artifacts](#20-outputs-and-artifacts)
21. [Design philosophy and trade-offs](#21-design-philosophy-and-trade-offs)
22. [How to present / demo the project](#22-how-to-present--demo-the-project)
23. [Glossary](#23-glossary)

---

## 1. What problem does SentinEL solve?

In technical interviews (especially remote ones), a candidate might **read answers from a hidden script** instead of speaking spontaneously. That is hard to detect with a single signal:

- A scripted answer can sound **fluent** (low fillers).
- A natural answer about AWS can sound **technical** (high jargon).
- Acoustic-only systems often **false-positive** nervous but honest candidates.

**SentinEL** (Sentinel for interview integrity) analyzes interview recordings and flags answers that are **probable script reading**, while preserving uncertainty when evidence is weak.

It does **not** claim courtroom proof. It produces **explainable per-answer judgments** with logs showing *why* each decision was made.

---

## 2. High-level idea in one minute

1. **Calibrate** on a short video where the *same person* deliberately **reads a paragraph aloud** (script-reading mode).
2. The system learns how **that person sounds and behaves when reading** → the **SCRIPT profile**.
3. During the **interview**, for each candidate answer:
   - Compare short **4-second windows** to SCRIPT vs a growing **NATURAL profile** (learned only from spontaneous-looking windows).
   - Transcribe speech, analyze wording, acoustic voice quality, and (optionally) GPU features.
   - Fuse everything into a status: **CLEAR**, **AMBIGUOUS**, or **PROBABLE_SCRIPT_READING**.

The key insight: **script reading ≠ “bad speech.”** It is **behavioral mismatch** — delivery looks more like calibration reading than like the person’s own spontaneous baseline, *plus* content that reads like memorized definitions rather than personal experience.

---

## 3. System architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER (Browser or CLI)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                    │ upload videos                    │ read results
                    ▼                                  ▼
┌──────────────────────────────┐            ┌──────────────────────────────┐
│   Web UI (FastAPI + static)   │            │  results.json + decision log  │
│   web/app.py, run_web.ps1     │            │  charts, per-answer breakdown │
└──────────────────────────────┘            └──────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    services/pipeline.py  (orchestrator)                      │
│   run_calibrate()  ·  run_analyze()  ·  progress logs  ·  runtime tags       │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                         │
         ▼                    ▼                         ▼
┌─────────────────┐  ┌─────────────────┐    ┌─────────────────────────────┐
│ processors/     │  │ engine/         │    │ gpu_client.py (optional)    │
│ audio, video,   │  │ contrastive,    │    │ HTTP → Kaggle notebook      │
│ transcript      │  │ fusion, NLP,    │    │ Whisper large-v3 + GPU score│
└─────────────────┘  │ intra-individual│    └─────────────────────────────┘
                     └─────────────────┘
```

**Two runtimes:**

| Runtime | When | Tag in UI |
|---------|------|-----------|
| **Local CPU** | Default; diarization, openSMILE, Whisper `small` | `[Local CPU]` |
| **Kaggle GPU** | When `KAGGLE_GPU_URL` is set | `[Kaggle GPU]` / `[Local + Kaggle]` |

Local machine orchestrates everything; Kaggle only runs heavy ASR/GPU scoring when configured.

---

## 4. User workflow (how you actually use it)

### Step A — One-time setup

1. Install Python **3.10**, **ffmpeg**, dependencies (`requirements.txt`, WhisperX, CPU PyTorch).
2. Copy `.env.example` → `.env`, set `HF_TOKEN` (HuggingFace for pyannote models).
3. Accept pyannote model terms on HuggingFace (one-time).

### Step B — Calibrate (SCRIPT fingerprint)

**Input:** Short video of the candidate **reading** a paragraph aloud (30–60 seconds is enough in fast mode).

**Output:** `calibration_profile.json` containing:

- `acoustic_reading_profile` — openSMILE + Parselmouth stats from reading windows
- `script_profile` — full behavioral profile for contrastive engine (v5)
- `personal_baseline` — median/MAD of voice metrics for intra-individual comparison
- `linguistic_calibration` — filler rate, WPS, etc. from reading transcript
- Optional `gpu_reading_profile` if Kaggle calibration ran

**CLI:**

```bash
python main.py calibrate --video path/to/reading.mp4
```

**Web:** Upload calibration video → job runs → download profile JSON.

### Step C — Analyze (interview)

**Input:** Interview recording (e.g. `.webm`) + calibration profile.

**Output:** `results.json` with per-answer status, scores, transcripts, window logs, full decision log.

**CLI:**

```bash
python main.py analyze --video interview.webm --calibration calibration_profile.json
```

**Web:** Upload interview + select calibration job → progress with live logs → results page.

### Step D — Report (optional)

```bash
python main.py report --results results.json
```

Prints a table of answers with contrastive scores and channel breakdowns.

---

## 5. Technology stack

| Layer | Technology | Role |
|-------|------------|------|
| **Language** | Python 3.10 | Entire backend |
| **Config** | `python-dotenv`, `config.py` | All tunable thresholds via `.env` |
| **Audio I/O** | `ffmpeg`, `soundfile`, `librosa` | Extract 16 kHz mono from video |
| **Speaker diarization** | `pyannote.audio` 3.3 | Who spoke when (interview only) |
| **Voice features** | **openSMILE** (eGeMAPS-style features) | Pitch, voicing segments, etc. |
| **Voice quality** | **Parselmouth** (Praat) | Jitter, shimmer, HNR, pitch range |
| **Speech-to-text** | **WhisperX** | Transcription + word timestamps |
| **Deep learning runtime** | **PyTorch** (CPU or GPU on Kaggle) | Whisper, pyannote backends |
| **Video (optional)** | OpenCV, MediaPipe | Gaze/lip (often disabled for speed) |
| **Web server** | **FastAPI** + **Uvicorn** | REST API + static frontend |
| **Remote GPU** | **Kaggle notebook** + **ngrok** + `httpx` | Optional offload |
| **Math** | NumPy, SciPy | Profiles, similarities, EWMA |

**Optional (off by default):** LLM judge tie-breaker on AMBIGUOUS answers (`ENABLE_LLM_JUDGE`). **Not used:** Docker, cloud deployment (runs locally).

---

## 6. End-to-end data flow

```
VIDEO FILE
    │
    ├─► Extract audio (16 kHz WAV)
    │
    ├─► [Interview] pyannote diarization → speaker turns
    │         └─► select CANDIDATE speaker (auto / most_speech / longest_turns)
    │         └─► merge turns into ANSWER segments (gap > 3s = new answer)
    │
    ├─► Slice each answer into 4s windows, 2s hop
    │         └─► per window: openSMILE + Parselmouth features
    │
    ├─► WhisperX transcribe each answer → words + text
    │
    └─► FOR EACH ANSWER:
            acoustic score (vs calibration reading profile)
            linguistic score (vs calibration linguistic baseline)
            semantic specificity (rule-based NLP on transcript)
            contrastive engine (SCRIPT vs NATURAL per window)
            answer synthesis (behavioral dominance, tiers)
            intra-individual (deviation from personal baseline)
            fused scorer (weighted channels + EWMA)
            session feedforward + finalize_interview_sourcing
            final semantic pass (protect personal answers)
            optional LLM judge (AMBIGUOUS tie-breaker only, if enabled)
            └─► status + confidence + explanation list
```

---

## 7. Phase 1 — Calibration (building the SCRIPT fingerprint)

**Entry point:** `services/pipeline.py` → `run_calibrate()`

### What happens

1. **Audio extraction** — `processors/audio_processor.py` pulls audio from video.
2. **Fast mode** (default) — skips pyannote; treats whole clip as one speaker reading.
3. **Windowing** — 4 s windows, 2 s hop; extract openSMILE + Parselmouth per window.
4. **Acoustic profile** — `engine/analysis_engine.py` builds `acoustic_reading_profile` (mean + robust std per metric).
5. **Transcription** — Whisper (`tiny` in fast calibrate, or configured size).
6. **Linguistic calibration** — `engine/linguistic_analyzer.py` records filler rate, words-per-second, etc. while reading.
7. **SCRIPT profile** — `engine/contrastive_engine.py` builds multi-dimensional `BehavioralProfile` from calibration windows (acoustic + linguistic + cognitive features).
8. **Personal baseline** — `engine/personal_baseline.py` stores median/MAD of key metrics for later intra-individual comparison.
9. **Optional Kaggle** — sends audio to remote `/calibrate` for GPU Parselmouth baseline.

### Why calibration matters

The system is **personalized but not person-dependent for cheating detection**:

- **Personalized:** SCRIPT profile is *this user's* reading voice.
- **Person-independent layers:** semantic specificity (memorized AWS prose vs “I use Instagram metrics”) works for any candidate.

Without calibration, `script_similarity` would be meaningless (no reference).

---

## 8. Phase 2 — Interview analysis

**Entry point:** `services/pipeline.py` → `run_analyze()`

### Major steps (in order)

| Step | Component | Description |
|------|-----------|-------------|
| 1 | `AudioProcessor.process_interview()` | Diarize, pick candidate speaker, segment answers |
| 2 | Kaggle prefetch OR local Whisper | Transcribe all answers (parallel on GPU if configured) |
| 3 | Per-answer loop | Score each answer segment |
| 4 | `AnalysisEngine.score_answer()` | Acoustic similarity to reading profile |
| 5 | `LinguisticAnalyzer.analyze()` | Fluency, fillers, timing vs calibration |
| 6 | `compute_semantic_specificity()` | Transcript content analysis |
| 7 | `FusedScorer.score_answer()` | Weighted fusion of channels |
| 8 | `ContrastiveEngine.process_answer()` | Window-level SCRIPT vs NATURAL |
| 9 | `IntraIndividualSession` | Person-relative deviation |
| 10 | `apply_specificity_to_status()` | Promote/demote based on transcript |
| 11 | `finalize_interview_sourcing()` | Session-level external-source inference |
| 12 | Final semantic pass | Protect personal-narrative answers from session downgrade |
| 13 | `intra_session.finalize_session()` | Cross-answer drift summary |
| 14 | `apply_llm_judge_to_answers()` (optional) | LLM tie-breaker on **AMBIGUOUS** only when `ENABLE_LLM_JUDGE=true` |

Progress messages are tagged `[Local CPU]` or `[Kaggle GPU]` so you always know where work runs.

### Optional LLM judge

When enabled, `engine/llm_judge.py` calls a pluggable provider (`openai`, `anthropic`, or `ollama`) **only** for answers still marked `AMBIGUOUS` after step 13. It can promote to `PROBABLE_SCRIPT_READING`, demote to `CLEAR`, or leave `AMBIGUOUS`. Interviewer questions mis-labeled as answers (`is_interviewer_speech`) force `CLEAR` with a diarization warning. API failures fail open (status unchanged). Results include an `llm_judge` block per answer and `[analyze/llm_judge]` decision-log entries.

---

## 9. Core concept: dual-profile contrastive engine

**File:** `engine/contrastive_engine.py`  
**Profiles:** `engine/profile_memory.py` (`BehavioralProfile`)

### Two profiles

| Profile | Source | Meaning |
|---------|--------|---------|
| **SCRIPT** | Calibration video (intentional reading) | “How this person sounds when reading aloud” |
| **NATURAL** | Interview windows with high **naturality_score** | “How this person sounds when speaking spontaneously” |

NATURAL profile is built **incrementally** during the interview. Windows that look scripted do **not** pollute NATURAL (gated by `naturality_scorer.py` and `profile_purity.py`).

### Per 4-second window

| Metric | Formula / meaning |
|--------|-------------------|
| `script_similarity` | Similarity of window features to SCRIPT profile |
| `natural_similarity` | Similarity to NATURAL profile (0 if profile empty) |
| `contrastive_score` | Roughly `script_similarity − natural_similarity` |
| `naturality_score` | Cognitive spontaneity estimate (pauses, fillers, repair, etc.) |
| `suspicious_flag` | Whether contrastive EWMA crossed margin |
| `suspicion_level` | NONE / WEAK / MODERATE / STRONG |

### Why contrastive?

A nervous candidate might have high script similarity on acoustics alone. If their **natural_similarity** is also high (they always talk that way), contrastive score stays lower → fewer false positives.

### NATURAL profile seeding

On analyze startup, NATURAL can be **seeded from calibration voice anchor** (median features from calibration windows) so `natural_similarity` is not frozen at zero on the first interview answers.

---

## 10. Signal channels (what we measure)

### Acoustic channel (`engine/analysis_engine.py`)

Uses openSMILE + Parselmouth per window, compared to `acoustic_reading_profile`:

- Pitch variability (`F0semitoneFrom27.5Hz_sma3nz_stddevNorm`)
- Voiced / unvoiced segment lengths
- Jitter, shimmer, harmonic-to-noise ratio (HNR)
- Pitch range (Hz)

**Intuition:** Reading aloud often has different prosody and voice stability than spontaneous storytelling.

### Linguistic channel (`engine/linguistic_analyzer.py`)

From Whisper word timestamps:

- Words per second, pause gaps, pause entropy
- Filler words (`um`, `uh`, “you know”)
- Self-corrections (“actually”, “sorry, I mean”)
- Compared to calibration reading baseline

**Intuition:** Scripted delivery may be too smooth OR oddly rhythmic; natural answers often have irregular pauses.

### Semantic specificity channel (`engine/semantic_specificity.py`)

**Rule-based NLP** on transcript text (no extra ML model):

- Proper nouns, numbers, first-person project verbs
- Hedging and sentence-length variance
- **Memorized technical script** — AWS/WebSocket/microservices definition prose
- **Personal narrative** — “I use Instagram metrics…”, “I have been an influencer…”

Feeds `generic_script_likelihood` into fusion and can override status after other layers.

### Gaze / Lip channels (`engine/gaze_analyzer.py`, `engine/lip_analyzer.py`)

Designed for reading eyes/lips vs camera. **Often disabled** in current pipeline for performance; scores appear as `null` in logs. System falls back to acoustic + linguistic + specificity.

### GPU channel (`gpu_client.py` → Kaggle server)

Optional Parselmouth-heavy score from remote server, weighted in `FusedScorer` when present.

---

## 11. Scoring layers (how a verdict is decided)

Think of scoring as **layers of evidence**, not one formula.

```
Layer 1: Window-level contrastive + suspicion tiers
         (temporal_evidence.py, suspicion_calibration.py)
              │
              ▼
Layer 2: Answer behavioral synthesis
         (answer_synthesis.py — dominant mode, STRONG/WEAK windows)
              │
              ▼
Layer 3: Cognitive sourcing
         (cognitive_sourcing.py — external vs internal generation)
              │
              ▼
Layer 4: Fused multi-channel score + EWMA
         (fused_scorer.py)
              │
              ▼
Layer 5: Intra-individual adjustment
         (intra_individual.py — vs personal baseline)
              │
              ▼
Layer 6: Semantic specificity (per answer)
         (semantic_specificity.py)
              │
              ▼
Layer 7: Session sourcing (all answers)
         (finalize_interview_sourcing)
              │
              ▼
Layer 8: Final semantic authority pass
         (pipeline.py — protect personal natural answers)
```

### Layer 1 — Temporal evidence

- Maintains **EWMA** of contrastive scores across windows.
- Uses tiered suspicion: **STRONG** counts more than **WEAK**.
- Prevents a single noisy window from dominating.

### Layer 2 — Answer synthesis

**File:** `engine/answer_synthesis.py`

Aggregates all windows in one answer:

- Suspicious density, peak suspicion, longest STRONG streak
- Cognitive spontaneity vs guided explanation index
- Essay-like thematic continuity
- Semantic-acoustic coherence

Can promote to **PROBABLE** when many independent signals agree, or soften to **AMBIGUOUS** when “prepared but internal” protection applies.

### Layer 3 — Cognitive sourcing

**File:** `engine/cognitive_sourcing.py`

Estimates whether speech patterns look **externally guided** (reading/listening) vs **internally generated** (recalling own experience). Feeds session-level refinement.

### Layer 4 — Fused scorer

**File:** `engine/fused_scorer.py`

Weighted sum of available channels (defaults in `config.py`):

- Acoustic, linguistic, specificity, gaze, lip
- Optional GPU (30% weight when present)

Produces `raw_score`, `smoothed_score` (EWMA across answers), preliminary status.

### Layer 5 — Intra-individual

**File:** `engine/intra_individual.py`

Question: *“Is this answer unusual **for this person**?”*

Uses `personal_baseline` from calibration:

- Relative deviation on jitter, shimmer, HNR, pitch range, etc.
- Cognitive cost flatness, turbulence suppression
- Outputs `P(external|evidence)` — not a final verdict alone, but modulates over-confident flags

### Layers 6–8 — Semantic + session

Described in sections 13 and 15.

---

## 12. Answer statuses and confidence

### Statuses

| Status | Meaning for presenter |
|--------|------------------------|
| **CLEAR** | Evidence favors spontaneous / personal delivery |
| **AMBIGUOUS** | Conflicting or weak evidence — honest uncertainty |
| **PROBABLE_SCRIPT_READING** | Multiple channels suggest memorized/scripted delivery |

### Confidence

**LOW / MEDIUM / HIGH** — how much trust to put in the status (based on STRONG window count, evidence weight, profile health).

### Important UI concepts

- **Composite / EWMA** — smoothed suspicion track over windows or answers
- **Evidence** — weighted sum of suspicion tiers
- **STRONG count** — number of highest-tier suspicious windows
- **Decision explanation** — human-readable bullet list in results

---

## 13. Semantic specificity (transcript intelligence)

**File:** `engine/semantic_specificity.py`  
**Enable flag:** `ENABLE_SEMANTIC_SPECIFICITY=true`

### Why it exists

Acoustic models confuse:

- **Natural** technical speech (candidate explains their project)
- **Scripted** textbook definitions (AWS API Gateway paragraph read verbatim)

Semantic layer reads the **words** and detects:

| Signal | Scripted example | Natural example |
|--------|------------------|-----------------|
| Memorized technical | “HTTP is stateless… WebSocket maintained persistent connection…” | “I use Instagram metrics to see what performs” |
| Personal narrative | Low — no “I/my” workflow | High — “I use…”, “I have been…” |
| Generic essay | Platitudes, no specifics | Names, tools, personal details |

### Key functions

- `compute_semantic_specificity(transcript)` → scores + `reasons[]`
- `is_personal_natural_answer(spec)` → protect natural answers
- `apply_specificity_to_status(...)` → promote PROBABLE or restore CLEAR

### Architecture essay detection

Phrases like **microservices**, **incident management**, **disaster response**, **We implemented…** boost memorized-technical score even without AWS/WebSocket vocabulary — catches system-design scripts.

---

## 14. Intra-individual modeling (person-relative)

**Files:** `engine/intra_individual.py`, `engine/personal_baseline.py`, `engine/cognitive_cost.py`

### Idea

Global thresholds (“high jitter = suspicious”) fail across speakers. Intra-individual modeling asks:

> Compared to **this candidate’s calibration baseline**, is this answer acoustically “flat,” unusually stable, or low-turbulence in a way that matches external guidance?

### Outputs (per answer)

- `p_external_guidance` — probability-like score
- `rel_mean_deviation` — feature deviation from personal median
- `intra_status` — CLEAR / AMBIGUOUS / etc.
- Session finalize: `cross_answer_drift`, `content_uniformity`

### Bootstrap

First N interview answers can **seed** baseline if needed (`PERSONAL_BASELINE_EARLY_ANSWERS`) so deviation is not computed from an empty model.

---

## 15. Session-level reasoning

**Files:** `engine/cognitive_sourcing.py`, `engine/cross_answer_content.py`

### Session sourcing (`finalize_interview_sourcing`)

After all answers are scored individually:

- Computes session-level external vs internal likelihood
- May promote answers to AMBIGUOUS or PROBABLE if **persistent** external pattern across interview
- **Guard:** does not promote if `is_personal_natural_answer()` or strong personal narrative

### Cross-answer content

- TF-IDF-style **content uniformity** — similar vocabulary across unrelated questions suggests rehearsed script bank
- `SessionEvidenceAccumulator` — running session prior fed into per-answer specificity

### Final authority pass

**After** session sourcing, pipeline runs **one more** `apply_specificity_to_status()` on each answer so personal workflow answers (e.g. social media / Canva) are not left as AMBIGUOUS because of session soft evidence.

---

## 16. Kaggle GPU offload (optional)

### Components

| File | Role |
|------|------|
| `kaggle_gpu_server.ipynb` | Runs on Kaggle GPU + ngrok tunnel |
| `gpu_client.py` | HTTP client from local machine |
| `server/kaggle_gpu_server.py` | FastAPI endpoints on Kaggle |

### Endpoints (conceptual)

- `POST /calibrate` — GPU reading baseline from calibration audio
- `POST /analyze` or batch — Whisper large-v3 + GPU score per answer

### Local `.env`

```env
KAGGLE_GPU_URL=https://xxxx.ngrok-free.dev
SENTINEL_SECRET=your-shared-secret
SKIP_LOCAL_WHISPER_WHEN_KAGGLE=true   # optional: skip loading local Whisper
```

### Flow

1. Kaggle notebook stays running (GPU session).
2. Local SentinEL sends audio bytes over HTTPS.
3. Transcripts return to local pipeline; contrastive + specificity still run **locally** (pyannote, openSMILE, profiles).

**Note:** ngrok URL changes when notebook restarts — update `.env` each session.

---

## 17. Web UI and CLI

### Web UI

| Path | Purpose |
|------|---------|
| `web/app.py` | FastAPI routes: upload, job poll, results |
| `web/static/index.html` | Frontend |
| `web/static/app.js` | Charts, log stream, answer cards |
| `web_data/jobs/` | Per-job uploads + `results.json` |
| `run_web.ps1` | Starts server on port 8765 with correct Python |

**Job lifecycle:** `queued` → `running` (progress % + logs) → `done` / `error`

### CLI

| Command | Function |
|---------|----------|
| `python main.py calibrate` | Build profile |
| `python main.py analyze` | Score interview |
| `python main.py report` | Print summary table |

Both CLI and web call the **same** `services/pipeline.py` functions — no duplicate logic.

---

## 18. Project folder structure

```
openHands2/
├── main.py                 # CLI entry
├── config.py               # All environment-driven settings
├── gpu_client.py           # Kaggle HTTP client
├── explanation.md          # This document
├── requirements.txt
├── .env.example
│
├── services/
│   └── pipeline.py         # calibrate + analyze orchestration
│
├── processors/
│   ├── audio_processor.py  # diarization, segmentation, openSMILE
│   ├── transcript_processor.py  # WhisperX load/transcribe
│   ├── video_processor.py  # timeline / gaze (optional)
│   └── speaker_selection.py
│
├── engine/
│   ├── contrastive_engine.py    # SCRIPT vs NATURAL
│   ├── analysis_engine.py       # acoustic calibration/scoring
│   ├── fused_scorer.py          # multi-channel fusion
│   ├── answer_synthesis.py      # per-answer behavioral verdict
│   ├── cognitive_sourcing.py    # external vs internal
│   ├── semantic_specificity.py  # transcript rules
│   ├── intra_individual.py      # person-relative session
│   ├── personal_baseline.py
│   ├── feature_extraction.py    # unified window features
│   ├── linguistic_analyzer.py
│   ├── naturality_scorer.py
│   ├── profile_memory.py
│   └── ... (temporal, recovery, drift, etc.)
│
├── scoring/
│   └── baseline.py         # calibration JSON I/O
│
├── web/
│   ├── app.py
│   └── static/
│
├── server/
│   └── kaggle_gpu_server.py
│
└── kaggle_gpu_server.ipynb
```

---

## 19. Configuration (.env)

Key groups (see `.env.example` for full list):

| Group | Examples | Purpose |
|-------|----------|---------|
| Secrets | `HF_TOKEN`, `KAGGLE_GPU_URL` | Models + remote GPU |
| Whisper | `WHISPER_MODEL_SIZE=small` | ASR speed/quality |
| Fusion weights | `WEIGHT_ACOUSTIC`, `WEIGHT_SPECIFICITY` | Channel importance |
| Contrastive | `ENABLE_CONTRASTIVE_ENGINE`, `CONTRASTIVE_MARGIN` | Dual-profile on/off |
| Speaker ID | `CANDIDATE_SPEAKER=auto` | Which diarization speaker to score |
| Fast calibrate | `FAST_CALIBRATION=true` | Skip diarization on calibrate |
| Semantic | `MEMORIZED_TECHNICAL_PROBABLE_MIN`, `PERSONAL_NARRATIVE_CLEAR_MIN` | NLP thresholds |
| Session | `SESSION_EXTERNAL_LIKELIHOOD_*` | Session promotion levels |

**Rule:** Prefer tuning `.env` over editing engine code for experiments.

---

## 20. Outputs and artifacts

### `calibration_profile.json`

- Version, source video path
- `acoustic_reading_profile`, `script_profile`, `personal_baseline`
- `linguistic_calibration`, window counts
- Optional `gpu_reading_profile`

### `results.json` (analyze)

- `answers[]` — per-answer status, scores, transcript, contrastive block, semantic_specificity
- `decision_log[]` — timestamped steps (what you see in UI logs)
- `window_logs[]` — per-window debug (contrastive mode)
- `profiles_end` — SCRIPT/NATURAL state after interview
- `session_sourcing_inference`, `session_intra_individual`, `session_content_analysis`

### Web job folder

`web_data/jobs/<job-id>/` — uploaded videos, profile, results.

---

## 21. Design philosophy and trade-offs

### What we optimize for

1. **Explainability** — every flag has reasons in logs  
2. **Personal calibration** — SCRIPT profile from the same speaker  
3. **False-positive control** — AMBIGUOUS is a valid outcome; weak evidence suppressed  
4. **Person-independent content checks** — memorized definitions vs personal narrative  
5. **CPU-first** — works without GPU; Kaggle optional for speed/quality  

### Known limitations

- Requires **calibration video** of the same person reading  
- Diarization can mis-label speaker in noisy/multi-party audio  
- Gaze/lip often off → less signal for “reading off screen”  
- English-centric NLP rules  
- Not real-time — minutes per interview on CPU  
- Detects **behavioral/script patterns**, not cryptographic proof  

### Why so many layers?

Single-threshold systems fail on real interviews. Layers let you **present nuanced reasoning**: acoustic says X, language says Y, transcript says Z → final judgment.

---

## 22. How to present / demo the project

### 30-second pitch

> “SentinEL calibrates on a person reading aloud, then analyzes their interview answers using voice, speech patterns, and transcript content. It compares each moment to ‘reading behavior’ vs ‘their own natural behavior’ and flags answers that look like memorized technical scripts — with full logs explaining why.”

### Recommended demo flow

1. Show **calibration video** (reading paragraph).  
2. Show **calibration profile** JSON snippet (`script_profile` samples).  
3. Run **analyze** on interview — open live log with `[Local CPU]` / `[Kaggle GPU]` tags.  
4. Click one **CLEAR** answer — show personal narrative reasons.  
5. Click one **PROBABLE** answer — show memorized technical + window suspicion.  
6. Show one **AMBIGUOUS** — explain uncertainty is intentional.  

### Anticipated questions

| Question | Answer |
|----------|--------|
| Is it an AI cheating detector? | It flags **probable script reading** from multimodal cues, not screen sharing or ChatGPT directly. |
| Why calibrate? | Reading voice is personal; we need *your* reading baseline. |
| Why AMBIGUOUS? | Better than wrong PROBABLE when evidence conflicts. |
| Does it need GPU? | No — CPU works; Kaggle optional for faster Whisper. |
| Can it run in production? | Architecture is batch/offline; real-time would need streaming redesign. |

---

## 23. Glossary

| Term | Definition |
|------|------------|
| **SCRIPT profile** | Behavioral fingerprint from calibration reading |
| **NATURAL profile** | Fingerprint built from spontaneous-looking interview windows |
| **Window** | 4 s audio slice, 2 s hop |
| **Contrastive score** | How much more “reading-like” than “natural-like” |
| **EWMA** | Exponential moving average — smooths scores over time |
| **Diarization** | Who spoke when (pyannote) |
| **openSMILE** | Open-source audio feature extractor |
| **WhisperX** | Speech recognition with alignment |
| **Specificity** | How personal/detailed the transcript is |
| **Intra-individual** | Compared to self, not population average |
| **Session sourcing** | Whole-interview pattern inference |

---

*Document version: aligned with SentinEL v5 contrastive engine, semantic specificity, intra-individual session, and optional Kaggle GPU path. For setup commands, see `README.md`.*
