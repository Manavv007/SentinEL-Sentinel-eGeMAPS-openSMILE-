# SentinEL — Project Goal

This document describes **what we are trying to build and why**. It is written for a new contributor or agent starting with no prior context about this repository.

**Important:** The system described here is **not complete**. Current results are **inaccurate and unreliable** in real interviews. Do not treat anything in this codebase as the correct approach. You are free to rethink architecture, models, and workflow from first principles as long as you move toward the goal below.

---

## The problem we are solving

In remote technical interviews, a candidate may **read answers from a hidden script** (on another screen, notes, or memorized text) instead of answering spontaneously from their own knowledge and experience.

This is difficult to catch because:

- Scripted answers can sound **fluent, confident, and technically correct**
- Honest candidates can sound **nervous, stiff, or overly formal**
- Technical answers naturally use **textbook language** even when genuine
- Interview recordings usually contain **two speakers** — an interviewer (often an AI bot) and the candidate — and scoring the wrong speaker invalidates everything

We are building **SentinEL**: a system that analyzes interview recordings and produces **per-answer integrity judgments** — whether each candidate response is likely spontaneous or likely read from an external script.

This is **not** a hiring decision engine. It is an **integrity signal** with explainable reasoning. It should prefer honest uncertainty over false confidence.

---

## Interview format we must support

Real sessions look like this:

1. An **AI or human interviewer** opens the session (welcome, instructions, role context)
2. The interviewer asks a series of **technical questions** (often with follow-ups)
3. The **human candidate** answers in turn

Example pattern (simplified):

```
Interviewer: Hello… I'm [name]. Let's begin. Could you introduce yourself?
Candidate:   Yeah, sure. I'm [name]… [personal background and why they want the role]

Interviewer: How would you ensure encryption of sensitive data at rest and in transit?
Candidate:   [technical answer — may be genuine or scripted]

Interviewer: What key management practices would you implement…?
Candidate:   [answer — may drift off-topic, refuse, or ask to change subject]
```

The system must:

- Identify **only candidate answers** as scorable units
- **Never** treat interviewer speech (intros, questions, acknowledgments) as candidate answers
- Handle **long AI monologues**, **short AI prompts**, and **variable-length candidate answers**
- Produce one judgment per **candidate answer segment**, aligned with what was actually said

---

## End-to-end user workflow

The product has two phases:

### Phase 1 — Calibration

The user uploads a short recording of **the same candidate deliberately reading a script aloud** (e.g., reading a paragraph from a document).

Purpose: establish what **this specific person** sounds and behaves like when they are **known to be reading**, so later comparisons are personal, not generic.

Output: a **calibration profile** for that candidate that can be saved and reused.

### Phase 2 — Interview analysis

The user uploads the **full interview recording** and attaches the calibration profile from Phase 1.

Purpose: for each candidate answer in the interview, estimate whether delivery and content are more consistent with **spontaneous speech** or **script reading**.

Output: a structured report the user can review in a UI and export (e.g., JSON).

---

## What each answer judgment must include

For every candidate answer segment, the system should provide:

| Field | Meaning |
|-------|---------|
| **Status** | One of: likely spontaneous (**CLEAR**), likely scripted (**PROBABLE_SCRIPT_READING**), or **AMBIGUOUS** when evidence is insufficient |
| **Confidence** | How strongly the status is supported (e.g., LOW / MEDIUM / HIGH) |
| **Time range** | Start and end time of the answer in the recording |
| **Transcript** | What the candidate actually said in that segment |
| **Explanation** | Human-readable reasons for the judgment — not opaque scores |

At session level, a **summary** is useful: how many answers flagged, overall pattern, and whether the run looks trustworthy (e.g., segmentation quality).

---

## What “good” looks like (success criteria)

A working SentinEL should pass these checks on real interview data:

### Segmentation correctness (prerequisite)

- Interviewer intros and questions are **excluded**, not scored
- Candidate answers match **real spoken content** (not fragments of AI questions, not random unrelated topics from wrong time slices)
- Most real answers in a typical 10–20 question interview are **found and transcribed**, not missing

### Detection quality

- **Obvious script reading** (definition-heavy, essay-like, no personal anchor) tends toward **PROBABLE_SCRIPT_READING**
- **Clearly personal, situational answers** (specific project, constraint, failure, outcome) tend toward **CLEAR**
- **Mixed or weak evidence** stays **AMBIGUOUS** rather than forced into a binary label
- **False positives on honest nervous candidates** are minimized
- **False negatives on fluent memorized answers** are reduced but not at the cost of rampant false positives

### Explainability

- A reviewer can read the explanation and understand **why** an answer was flagged or cleared
- Logs make it possible to debug **where** a wrong judgment came from (bad segment vs bad scoring vs bad transcript)

### Practical use

- Runnable via a **web UI** (upload → calibrate → analyze → results) and/or CLI
- Accepts common **video/audio** interview uploads
- Completes in reasonable time for typical interview length (exact target is a product decision)
- Can run in environments where **heavy compute may need to be offloaded** (e.g., cloud GPU) — but *how* is not prescribed here

---

## Core scientific question

For each candidate answer, estimate:

> **Was this response produced by internally generating speech (recall, reasoning, experience), or by articulating externally prepared text (reading, reciting, paraphrasing a script)?**

Signals that *may* help (non-exhaustive; you decide what to use):

- **Delivery / voice behavior** compared to the same person’s known reading baseline
- **Linguistic content**: generic definitions vs situational, personal detail
- **Timing and fluency**: pause patterns, repair, mid-thought pivots vs smooth recitation
- **Multimodal cues** (if reliable): gaze, lip movement, etc.

No single signal is sufficient. The hard part is **fusion without over-penalizing honest technical speech**.

---

## Hard constraints and known failure modes

Any solution must explicitly address:

1. **Two-speaker separation** — wrong speaker = wrong transcripts = nonsense results
2. **AI interviewer dominance** — the bot often talks more total time than the candidate; simple “longest speaker” heuristics fail
3. **Text vs audio mismatch** — transcripts may omit fillers and alter how “natural” speech looks
4. **Technical ≠ scripted** — jargon-heavy genuine answers must not be flagged solely for sounding formal
5. **Personal but vague answers** — short first-person workflow answers are not the same as textbook essays
6. **Off-topic or refusal answers** — candidate may deflect (“ask about cricket instead”); these need sensible handling, not random mis-segmentation
7. **Calibration dependency** — comparisons should be **relative to this candidate**, not global thresholds
8. **Uncertainty is valid** — forcing CLEAR/SCRIPTED on weak evidence creates false trust

---

## Current state (honest assessment)

**The goal above is not yet achieved.**

On real interview recordings (AI interviewer + human candidate), the system today often:

- Scores **interviewer speech** as candidate answers
- Produces **wrong transcript snippets** (unrelated content, truncated AI questions, missing real answers)
- Returns judgments that **do not match** what was actually said in the session
- Fails the basic prerequisite: **correct answer segmentation**

Until segmentation and speaker attribution are reliable, downstream detection accuracy cannot be trusted.

**Do not optimize scoring layers while segments are wrong.** Fix “what is being judged” before tuning “how it is judged.”

---

## Out of scope (unless explicitly requested)

- Automated hiring decisions or candidate ranking
- Proving cheating in a legal or disciplinary sense
- Live real-time interception during an ongoing interview (initial target is **post-hoc analysis** of a recording)
- Detecting other integrity issues (identity fraud, proxy test-taker, etc.) unless added as separate goals later

---

## Deliverables expected from this project

1. **Calibration pipeline** — ingest calibration recording → produce reusable candidate profile
2. **Interview analysis pipeline** — ingest interview recording + profile → per-answer judgments
3. **Review interface** — show answers, statuses, transcripts, and explanations
4. **Exportable results** — machine-readable output for integration or audit
5. **Documentation** — how to run, interpret results, and known limitations

---

## Guidance for a new agent

- Read this file first. Treat the repository as **experimental**, not as a validated design.
- Validate against **real multi-speaker interview recordings**, not only synthetic or single-speaker clips.
- When results look wrong, ask: **Is the wrong text being attributed to the candidate?** before tuning detection logic.
- Prefer **explainable, conservative** judgments over aggressive flagging.
- You may replace, remove, or rewrite any part of the existing codebase if it does not serve the goal.
- Success is defined by **correct segments + sensible per-answer judgments on real interviews**, not by architectural complexity.

---

## Glossary

| Term | Definition |
|------|------------|
| **Script reading** | Delivering an answer by reading or reciting externally prepared text rather than spontaneously generating it |
| **Spontaneous speech** | Answer produced through recall, reasoning, or lived experience, even if imperfect or rehearsed at a high level |
| **Calibration** | Reference recording where the candidate is known to be reading aloud |
| **Answer segment** | One contiguous candidate response between interviewer turns |
| **CLEAR** | Evidence favors spontaneous delivery |
| **PROBABLE_SCRIPT_READING** | Evidence favors scripted or externally sourced delivery |
| **AMBIGUOUS** | Evidence is mixed, weak, or insufficient for a confident call |

---

*Last intent: build SentinEL into a trustworthy interview integrity analyzer for AI-led technical interviews. The path to get there is open.*
