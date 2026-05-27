# SentinEL — Project Progress Log

**Last updated:** 2026-05-27  
**Repo:** https://github.com/Manavv007/SentinEL-Sentinel-eGeMAPS-openSMILE-  
**Latest commit:** `2f07d31` — temporal reliability, calibration speed, local diarization fixes

This file is the handoff document for any agent continuing work. Read it before changing scoring, pipeline, or Kaggle integration.

---

## 1. What SentinEL Is

**SentinEL** is a multi-modal interview integrity system that detects **script-reading** vs **spontaneous** candidate answers during technical interviews.

| Layer | Technology |
|-------|------------|
| Audio features | openSMILE eGeMAPS, Parselmouth (local CPU) |
| Video | **Disabled** (gaze/lip removed for speed + stability) |
| Transcription | WhisperX tiny (calibration) / small or Kaggle large-v3 (interview) |
| Diarization | pyannote 3.1 (**local CPU default**; optional Kaggle GPU segment) |
| Scoring | Dual-profile contrastive engine (SCRIPT vs NATURAL) + temporal reliability |
| UI | FastAPI web app (`http://127.0.0.1:8765`) |

**Interview format:** Two speakers — AI interviewer + human candidate. Candidate track must exclude AI speech.

---

## 2. Architecture (Do Not Redesign Without User Approval)

```
Interview video
    │
    ├─► [Parallel Phase 1]
    │     ├─ Audio: extract WAV → local pyannote (default) OR Kaggle segment → openSMILE/Parselmouth windows
    │     └─ Video: no-op (timeline empty; gaze/lip removed)
    │
    ├─► [Optional] Kaggle GPU transcription per answer
    │
    └─► Per-answer scoring (contrastive v5)
          ├─ temporal_evidence + temporal_reliability (consistency authority)
          ├─ answer_synthesis → CLEAR / AMBIGUOUS / PROBABLE_SCRIPT_READING
          └─ cognitive spontaneity modulation (fluent natural protection)
```

**Always local:** Diarization (default), contrastive logic, profile purity, answer synthesis, consistency authority.  
**Optional Kaggle GPU:** `/transcribe_answer`, `/analyze_batch`; `/segment_interview` and `/calibrate` opt-in only.

---

## 3. Progress Timeline

### Phase A — Answer synthesis & recall (committed `d57fa7f` / `ef497a0`)
- `engine/answer_synthesis.py`, `scoring_v3.py`, `profile_purity.py`, `recall_recovery.py`
- Kaggle segmentation + cognitive spontaneity (`engine/cognitive_spontaneity.py`)

### Phase B — Diarization accuracy regression fix
- **`processors/speaker_selection.py`** — shared `auto` / `most_speech` / `longest_turns` logic
- **`KAGGLE_OFFLOAD_SEGMENTATION=false` by default** — local pyannote more reliable for AI vs candidate
- **`CANDIDATE_SPEAKER=auto`** recommended in `.env.example`
- In-memory audio to pyannote (avoids Windows `torchcodec` warnings)

### Phase C — Remove gaze/lip processing
- **`services/pipeline.py`** — video path is no-op; gaze/lip scores `None` in fusion
- Faster runs; fewer MediaPipe false signals

### Phase D — Calibration speed (30s video target &lt;2 min after warm load)
- **`KAGGLE_OFFLOAD_CALIBRATION=false`** — skip slow Kaggle `/calibrate` (large-v3 + align) by default
- **`PRELOAD_CALIBRATION_MODEL_ONLY=true`** — preload Whisper `tiny` only at web startup
- Parallel openSMILE windows: `CALIBRATION_WINDOW_PARALLEL_WORKERS=4`
- No whisper-timestamped fallback during fast calibration
- Shorter Kaggle calibrate timeouts when opt-in

### Phase E — Temporal evidence reliability (short answers + Answer 5)
- **`engine/temporal_reliability.py`** — duration-aware metrics, weak consistency, recovery/breathing
- Short-answer suppression — Answer 0 should stay CLEAR (few moderate windows)
- **Consistency authority** — detects **flat elevated contrastive flow** (0.18–0.21 band) even when tiers are downgraded to NONE by naturality caps
- **`flat_suspicious_flow_active`** — promotes Answer 5 style adaptive fake-spontaneity without STRONG peaks
- **`natural_breathing_detected`** — protects answers with dip-and-recover (0.17→0.08→0.17)

### Phase F — Kaggle notebook hardening
- Cell 1: matched torch stack, uninstall torchvision (fixes `torchvision::nms` / whisperx import)
- Cell 2: `NGROK_AUTHTOKEN` setup, in-memory pyannote for `/segment_interview`
- ngrok auth before tunnel start

---

## 4. Target Classification (User Requirements)

| Answer | Desired status | Notes |
|--------|----------------|-------|
| 0 | CLEAR | Short-answer reliability guard |
| 1 | PROBABLE or AMBIGUOUS | Scripted |
| 2 | CLEAR | Natural |
| 3 | PROBABLE or AMBIGUOUS | Scripted |
| 4 | CLEAR | Natural |
| 5 | **PROBABLE_SCRIPT_READING** | Flat weak suspicious flow — main fix target |
| 6 | PROBABLE or AMBIGUOUS | Scripted |

**Constraints:** No architecture redesign; keep profile purity; incremental threshold tuning only.

---

## 5. Key Files Map

| File | Role |
|------|------|
| `config.py` | All thresholds, Kaggle flags, consistency authority knobs |
| `engine/temporal_reliability.py` | **NEW** — flat flow, breathing, consistency authority |
| `engine/temporal_evidence.py` | EWMA + reliability wiring to synthesis |
| `engine/answer_synthesis.py` | Layer 3 final status; consistency authority promotion |
| `engine/suspicion_calibration.py` | Tiers + short-answer + flat-flow PROBABLE paths |
| `processors/speaker_selection.py` | **NEW** — candidate vs AI speaker pick |
| `processors/audio_processor.py` | In-memory pyannote; parallel cal windows |
| `services/pipeline.py` | Calibrate/analyze; no video scan |
| `gpu_client.py` | Kaggle HTTP; optional calibrate/segment |
| `kaggle_gpu_server.ipynb` | Kaggle GPU server + ngrok |
| `progress.md` | This handoff doc |

---

## 6. Environment Configuration (Recommended)

```env
HF_TOKEN=...
KAGGLE_GPU_URL=https://....ngrok-free.dev
KAGGLE_SECRET=sentinEL2026
NGROK_AUTHTOKEN=...              # Kaggle notebook only

# Interview GPU (ASR)
KAGGLE_OFFLOAD=true
KAGGLE_OFFLOAD_TRANSCRIPTION=true
SKIP_LOCAL_WHISPER_WHEN_KAGGLE=true

# Diarization — local default (best AI vs candidate)
KAGGLE_OFFLOAD_SEGMENTATION=false
CANDIDATE_SPEAKER=auto
MIN_CANDIDATE_SEGMENT_SEC=4.0

# Calibration speed
FAST_CALIBRATION=true
SKIP_DIARIZATION_CALIBRATION=true
WHISPER_CALIBRATION_MODEL_SIZE=tiny
WHISPER_SKIP_ALIGN_CALIBRATION=true
KAGGLE_OFFLOAD_CALIBRATION=false
PRELOAD_CALIBRATION_MODEL_ONLY=true
CALIBRATION_WINDOW_PARALLEL_WORKERS=4

# Scoring
ENABLE_CONTRASTIVE_ENGINE=true
ENABLE_COGNITIVE_SPONTANEITY=true
```

---

## 7. Consistency Authority (Answer 5 Logic)

**Problem:** Adaptive scripted answers keep contrastive scores ~0.18–0.21 with low variance, but tier labels often become NONE → coverage looked zero → stayed CLEAR.

**Solution:** Score **contrastive continuity**, not only tier counts.

| Metric | Meaning |
|--------|---------|
| `elevated_contrastive_ratio` | Windows with score ≥ 0.16 |
| `suspicious_stability_score` | Low std + high elevated ratio |
| `natural_breathing_detected` | Mid-answer dips (natural cognition) |
| `flat_suspicious_flow_active` | Flat guided delivery signature |
| `consistency_authority_score` | Combined continuity authority |

Promotion: `flat_suspicious_flow_active` → PROBABLE without STRONG peaks (if not short answer, not natural breathing).

---

## 8. Known Issues & Diagnostics

| Issue | Cause / fix |
|-------|-------------|
| Answer 5 still CLEAR | Re-analyze after restart; check `flat_suspicious_flow_active` in results |
| Answer 0 AMBIGUOUS | Should be fixed by short-answer guards; verify `natural_breathing_detected` |
| Calibration &gt;5 min | Set `KAGGLE_OFFLOAD_CALIBRATION=false`; wait for preload; first run loads Whisper |
| torchcodec warning (local) | Warning only if using file-path pyannote; pipeline uses in-memory audio |
| Kaggle Cell 2 import error | Run Cell 1, restart kernel, set `NGROK_AUTHTOKEN` |
| Poor diarization on Kaggle segment | Use `KAGGLE_OFFLOAD_SEGMENTATION=false` (local pyannote) |
| ngrok URL changes | Update `KAGGLE_GPU_URL` each session |

---

## 9. How to Run

```powershell
pip install -r requirements.txt
# Configure .env (see section 6)
.\restart_web.ps1

# Kaggle: run notebook Cell 1 → restart → Cell 2; copy ngrok URL to KAGGLE_GPU_URL
python scripts/test_kaggle_gpu.py
```

**After code changes:** restart web app; re-run analyze (old `results.json` unchanged).

---

## 10. Git History (Recent)

```
2f07d31    Temporal reliability, consistency authority, calibration speed, diarization fixes
ef497a0    Add Kaggle interview segmentation and cognitive spontaneity scoring
d57fa7f    SentinEL v6 - contrastive engine, answer synthesis, profile purity
```

**Do not commit:** `.env`, `web_data/jobs/*`, `__pycache__/`, interview videos.

---

## 11. Agent Quick Start Checklist

1. Read `config.py` + `engine/temporal_reliability.py` before changing verdicts.
2. Default diarization: **local** (`KAGGLE_OFFLOAD_SEGMENTATION=false`).
3. Answer 5 fixes: tune `CONSISTENCY_*` and `PERSISTENT_WEAK_*` in config — not global thresholds.
4. Do not re-enable gaze/lip without user request.
5. Kaggle notebook: Cell 1 → kernel restart → Cell 2; update ngrok URL locally.
6. Update this file after major milestones.

---

*End of progress log.*
