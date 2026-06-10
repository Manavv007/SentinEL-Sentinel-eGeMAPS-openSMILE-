# SentinEL — Problems We Faced (Journey Log)

This document records the main issues encountered while building, tuning, and running SentinEL. It is meant for handoff, presentations, and future debugging — not as a bug tracker.

**Project:** Multi-modal interview script-reading detection  
**Repo:** https://github.com/Manavv007/SentinEL-Sentinel-eGeMAPS-openSMILE-

---

## Table of contents

1. [Setup and environment](#1-setup-and-environment)
2. [Broken scoring pipeline (critical)](#2-broken-scoring-pipeline-critical)
3. [False positives and accuracy](#3-false-positives-and-accuracy)
4. [Speaker diarization and AI vs candidate](#4-speaker-diarization-and-ai-vs-candidate)
5. [Kaggle GPU and ngrok](#5-kaggle-gpu-and-ngrok)
6. [Performance and “is it stuck?”](#6-performance-and-is-it-stuck)
7. [Web UI bugs](#7-web-ui-bugs)
8. [Calibration and dependencies](#8-calibration-and-dependencies)
9. [Session-level scoring side effects](#9-session-level-scoring-side-effects)
10. [7-answer interview — remaining accuracy gaps](#10-7-answer-interview--remaining-accuracy-gaps)
11. [Development workflow issues](#11-development-workflow-issues)
12. [Summary: resolved vs ongoing](#12-summary-resolved-vs-ongoing)

---

## 1. Setup and environment

### Missing or wrong `.env` variables

**Problem:** App failed to start or Kaggle auth failed.

| Symptom | Cause |
|---------|--------|
| Crash on import | `HF_TOKEN` missing (required for pyannote model download) |
| Kaggle requests rejected | `SENTINEL_SECRET` missing — config reads `SENTINEL_SECRET`, not `KAGGLE_SECRET` alone |

**Fix:** Copy `.env.example` → `.env`; set `HF_TOKEN`, `SENTINEL_SECRET` (and optionally duplicate `KAGGLE_SECRET` value under `SENTINEL_SECRET`).

---

### Python version and dependency order

**Problem:** WhisperX / torch conflicts, wrong Python version.

- Project targets **Python 3.10** (not 3.11+ for stability with pinned stack).
- **PyTorch must be installed before WhisperX** (CPU wheel recommended for local use).
- Running `uvicorn` directly with a Python that lacks `whisperx` breaks the web UI.

**Fix:** Use `.\run_web.ps1` / `.\restart_web.ps1` — launcher picks a Python where `whisperx` imports successfully.

---

### HuggingFace model access

**Problem:** First run fails on pyannote download.

**Fix:** Accept model terms on HuggingFace for speaker-diarization, segmentation, and wespeaker models; provide valid `HF_TOKEN`.

---

## 2. Broken scoring pipeline (critical)

At one point the contrastive engine was **running but not receiving valid inputs**, which made every downstream verdict unreliable.

### Symptoms in logs

| Metric | Broken value | Expected |
|--------|----------------|----------|
| `script_similarity` | **0** everywhere | Non-zero vs calibration SCRIPT profile |
| `script_profile` samples | **0** | Matches calibration windows (e.g. 17+) |
| `natural_similarity` | **Frozen ~0.287** | Varies per window; grows as NATURAL profile learns |
| `fused_ewma` | **0** | Tracks suspicion over windows/answers |
| Session outcome | **All AMBIGUOUS** | Mix of CLEAR / AMBIGUOUS / PROBABLE |

### Root causes

1. **SCRIPT profile build dropped rows** when cognitive/linguistic features were missing — profile ended up empty.
2. **NATURAL profile never seeded** — cold start kept natural similarity flat.
3. **EWMA did not update on NONE-tier windows** — behavioral track stayed at zero.
4. **`promote_ambiguous` in session sourcing** pushed almost everyone to AMBIGUOUS without guards.

### Fixes applied

- `calibration_feature_row()` — keep acoustic features + fallback when cog/ling missing.
- Seed `natural_profile` from calibration voice anchor on analyze start.
- `EWMA_BEHAVIORAL_TRACK_SCALE` — update EWMA even on low-tier windows (with decay).
- Guard session promotion with `is_personal_natural_answer()` and final semantic pass after session sourcing.
- Validation + logging for minimum calibration windows/samples.

---

## 3. False positives and accuracy

### All answers marked PROBABLE

**Problem:** Interview with mixed natural + scripted answers; system flagged **every** answer as script reading.

**Contributing factors:**

- Empty or weak SCRIPT/NATURAL contrast (see §2).
- Acoustic-only suspicion on fluent technical speech.
- No transcript-level check for “textbook definition” vs personal narrative.

**Fixes:**

- Dual-profile contrastive engine (SCRIPT vs NATURAL).
- **Semantic specificity** module — memorized technical script, personal narrative, generic essay detection.
- **Cognitive sourcing** — external vs internal generation likelihood.
- **Intra-individual modeling** — deviation from *this person’s* baseline, not population average.
- Answer synthesis consensus gates — require multiple independent signals before PROBABLE.

---

### Overfitting to one person / poor generalization

**Problem:** Tuning that worked for one candidate failed on another interview style.

**Approach taken:**

- Person-relative baseline from calibration (`personal_baseline`).
- Person-**independent** transcript rules (AWS/WebSocket/microservices definition prose).
- Generalization safety — preserve AMBIGUOUS when human variability prior is high.
- Prepared-internalization protection — don’t punish polished but genuinely internal speech.

---

### 4-answer test case (ground truth: natural 0,1 — scripted 2,3)

**Problem:** After early fixes, some scripted answers still missed or naturals false-flagged.

**Progress:** Eventually reached mostly correct separation on 4-answer set using memorized-technical detector for answers 2 and 3 (AWS/WebSocket-style prose).

---

## 4. Speaker diarization and AI vs candidate

### Wrong speaker scored

**Problem:** Two-speaker interviews (AI interviewer + human candidate). Scoring the **AI track** produces nonsense verdicts.

**Symptoms:**

- Very short “answers” (AI question bursts).
- Low or chaotic contrastive scores.
- Wrong answer count.

**Fixes:**

- `CANDIDATE_SPEAKER` strategies: `auto`, `most_speech`, `least_speech`, `longest_turns`.
- **`CANDIDATE_SPEAKER=auto`** recommended — votes on duration, turn length, segment count.
- **`MIN_CANDIDATE_SEGMENT_SEC=4`** — filter out short AI prompts.
- **Local pyannote default** — `KAGGLE_OFFLOAD_SEGMENTATION=false` because Kaggle fast segment was less reliable for AI vs candidate separation.

---

### pyannote on Windows

**Problem:** pyannote 4.x `torchcodec` / file-path diarization failures on Windows.

**Fix:** Pass **in-memory waveform** to pipeline instead of file path only; handle pyannote 4 API (`speaker_diarization` attribute).

---

## 5. Kaggle GPU and ngrok

### ngrok URL changes every session

**Problem:** Local `.env` `KAGGLE_GPU_URL` becomes stale after Kaggle notebook restart.

**Fix:** Re-run Kaggle notebook Cell 2; copy new URL into local `.env`; keep notebook **running** during analyze.

---

### ngrok ERR_NGROK_105 / auth failure

**Problem:** Kaggle notebook could not start tunnel — placeholder token in notebook.

**Fix:** Add real `NGROK_AUTHTOKEN` to **Kaggle Secrets** (not hardcoded in notebook); notebook loads via `UserSecretsClient`.

---

### Secret mismatch local ↔ Kaggle

**Problem:** GPU requests return 401/403.

**Fix:** `SENTINEL_SECRET` / `KAGGLE_SECRET` must **match** on local machine and Kaggle server.

---

### Kaggle on another device

**Clarification (not a bug):** Local SentinEL does **not** need Kaggle on the same PC — only reachable `KAGGLE_GPU_URL` + matching secrets. Diarization and contrastive scoring still run locally.

---

### Missing `gpu_reading_profile`

**Problem:** GPU score channel empty for older calibration files.

**Fix:** Re-calibrate after enabling Kaggle with `KAGGLE_OFFLOAD_CALIBRATION=true` (optional); local openSMILE profile still works without GPU profile.

---

## 6. Performance and “is it stuck?”

### Analyze appears frozen at 14% Local CPU

**Problem:** UI shows `[Local CPU] Diarizing interview audio (pyannote — may take several minutes)...` for a long time — looks stuck.

**Reality:** **Normal** for CPU. A ~7–8 minute interview can take **5–10+ minutes** for pyannote alone. Progress bar does not move much during this step.

**How to tell working vs stuck:**

| Working | Stuck |
|---------|--------|
| `log_count` increases when polling `/api/jobs/{id}` | Same `log_count` for 10+ min |
| Python process CPU 30–100% | Python CPU 0% for 5+ min |
| Message eventually moves to “Audio ready” / “Transcription” | Same message + same % forever |

**Fix / mitigation:**

- `PRELOAD_DIARIZATION_ON_STARTUP=true` — warm models at web start (first job faster).
- Parallel window workers, batch transcription, skip align on interview.
- Optional Kaggle fast segmentation (accuracy tradeoff).

---

### Slow first job after restart

**Problem:** First analyze after `run_web.ps1` waits 30–90s+ loading Whisper + pyannote.

**Expected:** Subsequent jobs reuse cached models in-process.

---

### Terminal only shows `GET /api/jobs/... 200 OK`

**Problem:** User thinks nothing is happening.

**Reality:** Browser polling every ~1.2s — **not** pipeline progress. Check UI decision log or job API `message` / `logs`.

---

## 7. Web UI bugs

### Only Answer 0 visible / UI crash after first answer

**Problem:** Results page broke mid-render; only first answer shown.

**Cause:** `decision_explanation` was sometimes a **string** instead of **`list[str]`** after session sourcing merge — JavaScript expected array.

**Fix:** Keep `decision_explanation` as list; `normalizeExplanation()` in `web/static/app.js`.

---

### No runtime visibility (Kaggle vs Local)

**Problem:** Could not tell whether transcription ran on Kaggle GPU or local CPU.

**Fix:** Progress messages prefixed with `[Kaggle GPU]`, `[Local CPU]`, or `[Local + Kaggle]` in pipeline + UI badges.

---

### Poll / fetch failures on large results

**Problem:** Browser “Failed to fetch” on huge job payloads.

**Fix:** Lite poll mode — trim logs; load full `results.json` via separate `/api/jobs/{id}/result` endpoint.

---

## 8. Calibration and dependencies

### Calibration very slow (~2 min for 30s video)

**Problem:** Full diarization + medium Whisper + video gaze on a reading clip.

**Fix:** Fast calibration mode — skip diarization (single speaker reading), tiny Whisper, skip align, lower video FPS, parallel audio features.

---

### Whisper strips fillers (`um`, `uh`)

**Problem:** Linguistic spontaneity signals weakened.

**Mitigation:** `initial_prompt` in WhisperX; optional `whisper-timestamped` fallback when installed.

---

### Gaze / lip disabled

**Problem:** Originally planned multi-modal; gaze/lip added noise and runtime cost.

**Decision:** Video path no-op for analyze; scores show `gaze: null`, `lip: null`. System uses acoustic + linguistic + specificity (+ optional GPU).

---

## 9. Session-level scoring side effects

### Natural answers downgraded to AMBIGUOUS

**Problem:** 7-answer interview — answers **2** and **4** (natural Instagram/Canva workflow) ended **AMBIGUOUS** despite `personal_narrative_score = 1.0`.

**Cause:** `finalize_interview_sourcing()` promoted CLEAR → AMBIGUOUS via “session soft evidence accumulation”; final semantic pass did not restore CLEAR because `is_personal_natural_answer()` required specificity ≥ 0.357 while those answers had **0.345** and **0.32**.

**Fix (attempted in session, may or may not be on current GitHub branch):**

- `has_strong_personal_narrative()` guard.
- Relaxed specificity threshold for personal workflow answers.
- Block `promote_ambiguous` when strong personal narrative present.

---

## 10. 7-answer interview — remaining accuracy gaps

**Ground truth (user-labeled):**

| Answer | Expected |
|--------|----------|
| 0, 2, 4 | Natural |
| 1, 3, 5, 6 | Scripted |

**Results before latest semantic tweaks (~5/7):**

| Answer | System | Issue |
|--------|--------|-------|
| 0 | CLEAR | OK |
| 1 | PROBABLE | OK (memorized AWS/WebSocket prose) |
| 2 | AMBIGUOUS | Wrong — session sourcing |
| 3 | PROBABLE | OK |
| 4 | AMBIGUOUS | Wrong — session sourcing |
| 5 | AMBIGUOUS | Miss — microservices essay; mem_tech ~0.22 |
| 6 | CLEAR | Miss — short disaster/microservices script |

**Gap:** Memorized-technical detector strong on AWS/WebSocket phrases but weak on **architecture essay** prose (“We implemented microservices…”, “traffic spikes”, “independent services”) without classic definition keywords.

**User decision:** Reverted local uncommitted semantic tweaks and reset to GitHub `main` (`fe5501f`) when overall accuracy felt worse — trade-off between fixes and regression risk.

---

## 11. Development workflow issues

### Local code vs GitHub sync confusion

**Problem:** Unsure whether local matched remote; mixed committed and uncommitted changes.

**Resolution:**

- `git fetch origin` + `git status -sb` + `git diff origin/main`
- Hard reset to `origin/main` when aligning with GitHub
- Pushed `explanation.md` as commit `1af5b48`

**Do not commit:** `.env`, `web_data/jobs/*`, interview videos.

---

### Kiro CLI “not installing” in this project

**Problem:** `kiro-cli` not recognized in Cursor terminal for openHands2.

**Reality:** Kiro installs **globally per user**, not per repo. Found at:

`C:\Users\BAPS\AppData\Local\Kiro-Cli\kiro-cli.exe`

**Cause:** Terminal PATH not refreshed after install, or expecting project-local install.

**Fix:** New PowerShell tab, or refresh PATH; run `kiro-cli chat` from project root. Optional project config: `.kiro/steering/*.md`.

---

### progress.md out of date

**Problem:** `progress.md` referenced older commit (`423c03f`) while `main` advanced to `fe5501f` / `1af5b48`.

**Note:** Use git log + this file for journey context; update `progress.md` after major milestones.

---

## 12. Summary: resolved vs ongoing

### Largely resolved

- Empty SCRIPT profile / zero script similarity
- Frozen natural similarity (cold start)
- Zero fused EWMA
- pyannote Windows / API compatibility
- Wrong speaker (AI vs candidate) with `auto` + local diarization
- ngrok auth on Kaggle notebook
- UI crash from `decision_explanation` type
- Runtime tags for Kaggle vs Local CPU
- Fast calibration path
- 4-answer scripted/natural separation (with semantic + contrastive fixes)

### Partially resolved / tuning ongoing

- 7-answer full accuracy (especially architecture-script answers 5 & 6)
- Session sourcing vs personal narrative balance (answers 2 & 4)
- Diarization speed on CPU (inherent cost)
- Generalization across different candidates and interview styles
- `progress.md` sync with latest commits

### Design limitations (not bugs)

- Requires calibration video (same person reading)
- English-centric NLP rules
- Batch/offline — not real-time streaming
- Probabilistic labels — AMBIGUOUS is intentional uncertainty
- No direct detection of screen sharing or ChatGPT — infers script-reading patterns

---

## Related docs

| File | Purpose |
|------|---------|
| `README.md` | Install and run |
| `explanation.md` | Full architecture and workflow |
| `progress.md` | Agent handoff / phase timeline |
| `.env.example` | Configuration reference |

---

*Last updated: 2026-05-30 — compiled from project development sessions and test runs (4-answer and 7-answer interviews).*
