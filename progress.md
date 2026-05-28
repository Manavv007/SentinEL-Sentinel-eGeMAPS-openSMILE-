# SentinEL — Project Progress Log

**Last updated:** 2026-05-28  
**Repo:** https://github.com/Manavv007/SentinEL-Sentinel-eGeMAPS-openSMILE-  
**Latest commit:** `423c03f` — intra-individual behavioral modeling, Kaggle ngrok auth fix

This file is the handoff document for any agent continuing work. Read it before changing scoring, pipeline, or Kaggle integration.

---

## 1. What SentinEL Is

**SentinEL** is a multi-modal interview integrity system that detects **script-reading** vs **spontaneous** candidate answers during technical interviews.

| Layer | Technology |
|-------|------------|
| Audio features | openSMILE eGeMAPS, Parselmouth (local CPU) |
| Video | **Disabled** (gaze/lip removed for speed + stability) |
| Transcription | WhisperX tiny (calibration) / small local or Kaggle large-v3 (interview) |
| Diarization | pyannote 3.1 (**local CPU default**; optional Kaggle `fast` segment) |
| Scoring | Contrastive v5 + **intra-individual baseline** + cognitive sourcing + session P(external) |
| UI | FastAPI web app (`http://127.0.0.1:8765`) |

**Interview format:** Two speakers — AI interviewer + human candidate. Candidate track must exclude AI speech.

**Primary question (architecture):** *Does this answer deviate from how THIS PERSON naturally behaves when internally generating speech?*

Secondary layer: cognitive sourcing asks whether deviation patterns look like **externally guided articulation** vs internal generation.

---

## 2. Architecture (Do Not Redesign Without User Approval)

```
Interview video
    │
    ├─► Audio: WAV → local pyannote (default) OR Kaggle segment → parallel openSMILE/Parselmouth windows
    │
    ├─► [Optional] Kaggle GPU transcription (parallel per answer)
    │
    ├─► Personal baseline (calibration + early answers + slow CLEAR updates)
    │
    ├─► Per-answer scoring (contrastive v5 + person-relative features)
    │     ├─ temporal_evidence + temporal_reliability
    │     ├─ intra_answer_turbulence, cognitive_cost, recovery_arc, cross_modal
    │     ├─ answer_synthesis + cognitive_sourcing
    │     └─ intra_individual → P(external|evidence) per answer
    │
    └─► Session pass: cross_answer_drift + sourcing + intra_individual refinement
```

**Calibration profile now includes:** `personal_baseline` (person-relative reference), not only SCRIPT reading fingerprint.

**Always local:** Diarization (default), personal baseline, contrastive logic, intra-individual session reasoning, answer synthesis.  
**Optional Kaggle GPU:** `/transcribe_answer`, `/analyze_batch`; `/segment_interview` and `/calibrate` opt-in only.

**Why hybrid (not 100% Kaggle):** GPU for heavy ASR; local for accurate AI vs candidate separation, stateful multi-answer reasoning, explainability, and stable web jobs without notebook/ngrok dependency.

---

## 3. Progress Timeline

### Phase A — Answer synthesis & recall (`d57fa7f` / `ef497a0`)
- `engine/answer_synthesis.py`, `scoring_v3.py`, `profile_purity.py`, `recall_recovery.py`
- Kaggle segmentation + cognitive spontaneity (`engine/cognitive_spontaneity.py`)

### Phase B — Diarization accuracy regression fix
- **`processors/speaker_selection.py`** — `auto` / `most_speech` / `longest_turns`
- **`KAGGLE_OFFLOAD_SEGMENTATION=false` by default** — local pyannote more reliable for AI vs candidate
- **`CANDIDATE_SPEAKER=auto`** recommended
- In-memory audio to pyannote (avoids Windows `torchcodec` warnings)

### Phase C — Remove gaze/lip processing
- Video path no-op; faster runs; fewer false signals

### Phase D — Calibration speed
- Fast cal: tiny Whisper, skip diarization, parallel openSMILE windows, no Kaggle `/calibrate` by default

### Phase E — Temporal evidence reliability
- `engine/temporal_reliability.py` — flat flow, breathing, consistency authority (Answer 5 target)

### Phase F — Kaggle notebook hardening
- Torch stack fixes, ngrok auth, in-memory pyannote on notebook

### Phase G — Profile purification & fluent-speaker protection *(this push)*
- **`engine/profile_disentanglement.py`** — SCRIPT profile purification; weighted similarity; fluent-natural learning path
- **`engine/profile_memory.py`**, **`profile_purity.py`**, **`recall_recovery.py`** — reduce fluent-speaker false positives from profile contamination
- Rehearsal vs guidance separation in **`answer_synthesis.py`** (`guidance_dominance_score`, `prepared_internal_speech_protection`)

### Phase H — Semantic guidedness & generalization-first calibration *(this push)*
- **`engine/cognitive_spontaneity.py`** — essay-like rhythm, thematic stability, emotional grounding; semantic–acoustic coupling per window
- **`answer_synthesis.py`** — behavioral consensus gate, human variability prior, semantic-effort decoupling requirements before PROBABLE
- Config: `GENERALIZATION_*`, `SEMANTIC_EFFORT_DECOUPLING_*`, `SEMANTIC_GUIDEDNESS_*`

### Phase I — Cognitive Sourcing Inference *(this push)*
- **`engine/cognitive_sourcing.py`** (NEW)
  - Semantic–effort covariance, chunk transitions, segment spontaneity variance
  - Interview variability profile + speaker style baseline
  - `internal_generation_likelihood` vs `external_sourcing_likelihood`
  - Session-level soft evidence accumulation (`finalize_interview_sourcing`)
  - Prepared-internalization protection (weakens when uniformity + collapsed covariance)
- Wired in **`temporal_evidence.py`**, **`answer_synthesis.py`**, **`services/pipeline.py`**
- Config: `ENABLE_COGNITIVE_SOURCING`, `SESSION_*`, `SOURCING_*`

### Phase J — Analysis speed optimizations *(this push)*
- **Process-wide cache** for pyannote + openSMILE (`processors/audio_processor.py`) — no reload per web job
- **Parallel interview windows** via `AUDIO_WINDOW_PARALLEL_WORKERS`
- **Parallel multi-answer feature extraction** during local diarization path
- **`WHISPER_SKIP_ALIGN_INTERVIEW`** — faster local ASR fallback
- **Batch local transcription** before scoring loop (`services/pipeline.py`)
- **`PRELOAD_DIARIZATION_ON_STARTUP`** — warm pyannote at web startup (`web/app.py`)
- Removed no-op video thread from analyze path

### Phase K — Web UI timeline & results fix (`a638bb9`)
- **`web/static/app.js`** — tier-scaled suspicion intensity, EWMA line, answer overlays, rich tooltips
- **`index.html`**, **`styles.css`** — timeline legend, chart height, tier key
- **Bug fix:** session sourcing must keep `decision_explanation` as **list** (string caused JS crash after Answer 0); `normalizeExplanation()` in UI

### Phase L — Intra-individual behavioral modeling *(this push)*
- **`engine/personal_baseline.py`** — personal speaking baseline (median/MAD); seeds from calibration; slow updates on CLEAR only
- **`engine/intra_individual.py`** — session orchestrator; wires all person-relative modules into pipeline
- **`engine/intra_answer_turbulence.py`** — micro-variability bursts vs personal norm (variance-of-variance)
- **`engine/cross_answer_drift.py`** — cross-answer behavioral drift / artificial uniformity
- **`engine/cognitive_cost.py`** — question difficulty vs flat cognitive cost profile
- **`engine/cross_modal_correlation.py`** — gaze/prosody/pacing coupling (neutral when video off)
- **`engine/recovery_arc.py`** — recovery trajectory after disfluency events
- **`engine/session_probabilistic.py`** — `P(external_guidance)` from 0.5; wide AMBIGUOUS band
- **`services/pipeline.py`** — saves/loads `personal_baseline`; logs `intra_individual` + `session_intra_individual`
- Config: `ENABLE_INTRA_INDIVIDUAL`, `INTRA_INDIVIDUAL_AUTHORITY`, `SESSION_P_*`, `PERSONAL_BASELINE_*`

### Phase M — Kaggle ngrok auth fix *(this push)*
- **`kaggle_gpu_server.ipynb` Cell 2** — loads `NGROK_AUTHTOKEN` from Kaggle Secrets via `UserSecretsClient` (not placeholder env)
- Fails fast if token missing or still `YOUR_NGROK_AUTHTOKEN_HERE`

---

## 4. Key Files Map

| File | Role |
|------|------|
| `config.py` | All thresholds, Kaggle flags, sourcing, intra-individual, speed knobs |
| `engine/personal_baseline.py` | **NEW** — person-relative baseline model |
| `engine/intra_individual.py` | **NEW** — intra-individual session orchestrator |
| `engine/intra_answer_turbulence.py` | **NEW** — within-answer micro-variability |
| `engine/cross_answer_drift.py` | **NEW** — session drift / uniformity |
| `engine/cognitive_cost.py` | **NEW** — cognitive cost vs difficulty |
| `engine/cross_modal_correlation.py` | **NEW** — multimodal coupling |
| `engine/recovery_arc.py` | **NEW** — post-disfluency recovery shape |
| `engine/session_probabilistic.py` | **NEW** — P(external) accumulation |
| `engine/cognitive_sourcing.py` | Internal vs external cognition inference + session pass |
| `engine/profile_disentanglement.py` | SCRIPT purification, style leak dampening |
| `engine/cognitive_spontaneity.py` | Spontaneity, guidedness, semantic–acoustic alignment |
| `engine/answer_synthesis.py` | Layer 3 final status; sourcing promotion; consensus gates |
| `engine/temporal_evidence.py` | EWMA + behavioral + sourcing enrichment |
| `engine/temporal_reliability.py` | Flat flow, breathing, consistency authority |
| `processors/audio_processor.py` | Cached models; parallel windows/answers |
| `processors/transcript_processor.py` | Whisper; skip align interview; filler fallback flag |
| `services/pipeline.py` | Calibrate/analyze; personal baseline; batch ASR; session passes |
| `web/static/app.js` | Timeline chart, answer cards, explanation normalization |
| `gpu_client.py` | Kaggle HTTP |
| `kaggle_gpu_server.ipynb` | Kaggle GPU server + ngrok |
| `progress.md` | This handoff doc |

---

## 5. Environment Configuration (Recommended)

```env
HF_TOKEN=...
KAGGLE_GPU_URL=https://....ngrok-free.dev
KAGGLE_SECRET=sentinEL2026

# Interview GPU (ASR) — use when notebook is running
KAGGLE_OFFLOAD=true
KAGGLE_OFFLOAD_TRANSCRIPTION=true
SKIP_LOCAL_WHISPER_WHEN_KAGGLE=true
KAGGLE_PARALLEL_ANSWERS=4
KAGGLE_SKIP_ALIGN_INTERVIEW=true
KAGGLE_TRANSCRIBE_ONLY=true          # faster than /analyze_batch if GPU score unused

# Diarization — local default (best AI vs candidate)
KAGGLE_OFFLOAD_SEGMENTATION=false
CANDIDATE_SPEAKER=auto
MIN_CANDIDATE_SEGMENT_SEC=4.0

# Speed
FAST_CALIBRATION=true
SKIP_DIARIZATION_CALIBRATION=true
WHISPER_CALIBRATION_MODEL_SIZE=tiny
WHISPER_SKIP_ALIGN_CALIBRATION=true
WHISPER_SKIP_ALIGN_INTERVIEW=true
AUDIO_WINDOW_PARALLEL_WORKERS=4
CALIBRATION_WINDOW_PARALLEL_WORKERS=4
PRELOAD_DIARIZATION_ON_STARTUP=true
KAGGLE_OFFLOAD_CALIBRATION=false
PRELOAD_CALIBRATION_MODEL_ONLY=true

# Scoring
ENABLE_CONTRASTIVE_ENGINE=true
ENABLE_COGNITIVE_SPONTANEITY=true
ENABLE_COGNITIVE_SOURCING=true
PREPARED_INTERNALIZATION_PROTECTION=true

# Intra-individual (person-relative baseline)
ENABLE_INTRA_INDIVIDUAL=true
INTRA_INDIVIDUAL_AUTHORITY=false
INTRA_INDIVIDUAL_PRESERVE_UNCERTAINTY=true
INTRA_INDIVIDUAL_SESSION_REFINEMENT=true
PERSONAL_BASELINE_EARLY_ANSWERS=2
SESSION_P_PRIOR=0.5
SESSION_P_PROBABLE_MIN=0.62
```

**Faster segmentation (tradeoff):** `KAGGLE_OFFLOAD_SEGMENTATION=true` + `KAGGLE_SEGMENT_MODE=fast`

**Kaggle Secrets:** add `NGROK_AUTHTOKEN` (real token from ngrok dashboard) for notebook Cell 2.

---

## 6. Intra-Individual (Explainability Fields)

Per answer (`intra_individual`):

| Field | Meaning |
|-------|---------|
| `person_relative.rel_mean_deviation` | Mean deviation from this person's baseline |
| `intra_answer_turbulence.intra_turbulence_suppression` | Suppressed micro-variability within answer |
| `cognitive_cost.cognitive_cost_flatness` | Flat cost under semantic load |
| `p_external_guidance` | Session-updated P(external) after this answer |
| `intra_status` / `intra_reasons` | Person-relative verdict layer |

Session (`session_intra_individual`):

| Field | Meaning |
|-------|---------|
| `personal_baseline` | Final adapted baseline stats |
| `session_probability.history` | P(external) evolution per answer |
| `cross_answer_drift.cross_answer_uniformity` | Low drift → artificially consistent delivery |

Set **`INTRA_INDIVIDUAL_AUTHORITY=true`** to make person-relative probability the final verdict authority.

---

## 7. Cognitive Sourcing (Explainability Fields)

Per answer (`contrastive.behavioral_synthesis`):

| Field | Meaning |
|-------|---------|
| `semantic_effort_covariance_score` | Coupling of semantic load vs acoustic effort |
| `segment_spontaneity_variance` | High = natural chunk variation; low = uniform delivery |
| `external_sourcing_likelihood` | Externally guided articulation plausibility |
| `internal_generation_likelihood` | Internally retrieved cognition plausibility |
| `external_soft_evidence` | Accumulated weak signal (AMBIGUOUS is not discarded) |

Interview-level: `session_sourcing_inference` in results JSON.

---

## 8. Known Issues & Diagnostics

| Issue | Cause / fix |
|-------|-------------|
| Only Answer 0 in UI | Fixed: `decision_explanation` must be list; refresh browser; re-analyze for new runs |
| Slow first analyze after restart | pyannote preload ~30–90s; subsequent jobs faster (cached) |
| Slow full interview on CPU | Local pyannote dominates; optional Kaggle `fast` segment |
| Answer 5 still CLEAR | Check `flat_suspicious_flow_active`; re-analyze after code update |
| Old results missing new fields | Re-run analyze; pre-upgrade `results.json` lacks sourcing metrics |
| ngrok URL changes | Update `KAGGLE_GPU_URL` each Kaggle session |
| ngrok ERR_NGROK_105 | Kaggle Secret key must be `NGROK_AUTHTOKEN` with real token (not placeholder) |
| Intra fields missing | Re-calibrate once for `personal_baseline`; then re-analyze |

---

## 9. How to Run

```powershell
pip install -r requirements.txt
# Configure .env (see section 5)
.\restart_web.ps1

# Kaggle: notebook Cell 1 → restart → Cell 2; copy ngrok URL to KAGGLE_GPU_URL
python scripts/test_kaggle_gpu.py
```

**After code changes:** restart web app; **re-calibrate** for `personal_baseline`; re-run analyze.

---

## 10. Git History (Recent)

```
423c03f    Intra-individual behavioral modeling + Kaggle ngrok Secrets fix
a638bb9    Cognitive sourcing inference, speed optimizations, timeline UI, profile disentanglement
6855a91    Update progress.md with commit 2f07d31
2f07d31    Temporal reliability, calibration speed, local diarization fixes
```

**Do not commit:** `.env`, `web_data/jobs/*`, `__pycache__/`, interview videos.

---

## 11. Agent Quick Start Checklist

1. Read `engine/personal_baseline.py`, `engine/intra_individual.py`, `config.py` before changing verdicts.
2. Default diarization: **local** (`KAGGLE_OFFLOAD_SEGMENTATION=false`).
3. Person-relative reasoning: use deviation from `personal_baseline`, not population fluency heuristics.
4. Do not convert `decision_explanation` to string — UI requires `list[str]`.
5. Re-calibrate after enabling intra-individual (baseline stored in profile).
6. `INTRA_INDIVIDUAL_AUTHORITY=false` by default — contrastive remains primary until explicitly enabled.
7. Do not re-enable gaze/lip without user request.
8. Update this file after major milestones.

---

*End of progress log.*
