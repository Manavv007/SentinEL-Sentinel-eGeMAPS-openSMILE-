# Problems You Will Likely Face — Interview Script-Reading Detection

This document lists **general problems** we encountered while building a system that detects when a candidate is **reading a script** versus speaking **naturally** in a technical interview. It is written for someone starting fresh on the same problem — not as a debugging guide for this repo.

If you are building something similar, expect these challenges early. Most are **design and data problems**, not “install the wrong library” problems.

---

## 1. The problem is harder than it looks

At first glance it seems simple: *listen to the interview and flag cheating.*

In practice you are trying to infer **how speech was produced** (memorized external text vs internal recall), not just whether someone sounds confident or fluent. Fluent scripted answers often look **more polished** than nervous honest ones. A system that flags “bad speech” will fail. A system that flags “too perfect speech” will also fail.

**Takeaway:** You are detecting a **behavioral and cognitive pattern**, not audio quality or grammar.

---

## 2. You cannot detect cheating without knowing the candidate’s baseline

Script reading is **personal**. The same person sounds different when:

- reading a paragraph aloud (calibration)
- answering spontaneously in an interview
- reciting a memorized technical answer

A one-size-fits-all threshold (“high jitter = suspicious”) breaks across speakers, accents, microphones, and nervousness.

**Takeaway:** You almost always need a **personal reference** — typically a short clip where the same person deliberately reads text, so you can compare interview answers to *their* reading voice, not a global average.

---

## 3. Two-speaker interviews break naive audio pipelines

Modern interviews often have:

- an **AI or human interviewer** asking questions
- a **human candidate** answering

If your pipeline treats “all speech in the video” as candidate answers, you will score the **interviewer’s questions** as candidate responses. Transcripts will say things like *“Could you describe…”* and *“Thanks for sharing…”* — which is exactly wrong.

**Takeaway:** Speaker separation (who is the candidate?) is not optional for this use case. Getting it wrong invalidates everything downstream.

---

## 4. The AI interviewer often talks more than the candidate

A common mistake is assuming the **candidate talks the most**. In bot-led interviews:

- the AI gives a **long opening** (intro, instructions, first question)
- the AI asks **many short questions**
- the candidate may speak **less total time** but in **longer answer bursts**

Heuristics based on “most total speech = candidate” frequently pick the **wrong speaker**.

**Takeaway:** Use **turn structure** (question → answer, opener vs responder), not duration alone.

---

## 5. Acoustic signals alone produce many false positives

Voice features (pitch, jitter, pauses, fluency) can suggest “reading-like” delivery, but:

- **Nervous honest** candidates can sound stiff
- **Prepared but genuine** candidates can sound smooth
- **Technical answers** naturally sound formal and structured

Relying on acoustics only tends to mark **everyone** suspicious or miss **fluent script readers**.

**Takeaway:** Treat acoustics as **one channel**, not the verdict.

---

## 6. Transcription changes the evidence

Speech-to-text systems often **remove filler words** (`um`, `uh`, `like`) even when they are present in the audio. Those fillers matter for spontaneity scoring.

You may see a warning that fillers are missing — that can mean either:

- the model cleaned them up, or
- the speaker truly had none (scripted or very fluent speech)

**Takeaway:** Do not assume the transcript is a perfect mirror of speech. Linguistic scoring depends on words you never see.

---

## 7. “Technical answer” and “memorized script” look almost the same in text

Candidates giving real answers about AWS, databases, or system design use the same vocabulary as textbook definitions. A rule like “lots of technical terms = scripted” will hurt honest engineers.

Conversely, memorized answers often use:

- definition-style prose (*“Unlike traditional HTTP…”*)
- impersonal architecture essays (*“We implemented microservices by separating…”*)
- no personal anchor (*“I built…”, “on my project…”*)

**Takeaway:** Content analysis must distinguish **personal experience** from **generic technical exposition**, not “technical vs non-technical.”

---

## 8. Personal natural answers can still look generic

Answers like *“I use Instagram metrics to see what performs well”* are genuinely personal but may score low on “specificity” compared to answers with company names, numbers, or long project stories.

Session-level logic that says *“this interview feels externally guided overall”* can **downgrade** these natural answers to “uncertain” even when the transcript is clearly first-person workflow speech.

**Takeaway:** Per-answer content signals and session-level signals can **conflict**. You need clear rules for which layer wins.

---

## 9. “Uncertain” is a feature, not a failure — but users hate it

Real interviews produce **mixed evidence**:

- acoustic: slightly suspicious
- text: personal and specific
- session: weak external pattern

Forcing every answer into **CLEAR** or **SCRIPTED** creates false confidence. Keeping **AMBIGUOUS** is honest but frustrates users who want a yes/no.

**Takeaway:** Design for **three outcomes** from the start: clear natural, clear scripted, and genuinely uncertain. Explain *why* in plain language.

---

## 10. Ground truth is hard to obtain

To know if your system works, you need interviews where **you know** which answers were scripted and which were natural. That means:

- controlled test recordings
- labeled answers (often only you know the truth)
- multiple interview lengths (4 answers vs 7+ answers behave differently)

Small test sets feel accurate; longer mixed interviews expose gaps quickly.

**Takeaway:** Build a **labeled test set** early. One good 4-answer test is not enough for production confidence.

---

## 11. Longer interviews introduce session effects

As more answers accumulate, the system builds beliefs like:

- “this person’s delivery is unusually uniform”
- “external guidance seems persistent across questions”

That can help catch **repeated script use** but can also **bias** later answers or **override** a clearly personal answer from earlier logic.

**Takeaway:** Session-level reasoning helps and hurts. Test on **multi-answer** interviews, not single clips.

---

## 12. Calibration and analyze are different problems

**Calibration** (reading a paragraph) should be fast and stable — one speaker, controlled text.

**Analyze** (full interview) is slow — diarization, many answers, optional remote transcription.

Optimizing one path can break expectations on the other. Users often expect analyze to be as fast as calibration; it will not be on CPU.

**Takeaway:** Set expectations: **minutes per interview** on local hardware is normal.

---

## 13. Optional GPU offload adds operational burden

Using a remote GPU for faster transcription helps, but introduces:

- keeping a **remote session alive**
- **URLs or tunnels that change** when the session restarts
- **matching secrets** between local machine and remote server
- fallback behavior when remote is down

The local machine still does most of the “thinking”; GPU is often just **faster speech-to-text**.

**Takeaway:** GPU is an **accelerator**, not a replacement for local orchestration and scoring — unless you redesign the whole pipeline for cloud-only.

---

## 14. Progress can look “stuck” while work is running

The slowest step is usually **who-spoke-when analysis** on long interviews. The UI may sit on one message for many minutes while CPU is busy. That often means **working**, not frozen.

**Takeaway:** Show **stage-based progress** and **live logs**; teach users that diarization dominates runtime.

---

## 15. Tuning fixes one interview and breaks another

We repeatedly saw:

- good separation on a **4-answer** test
- wrong labels on a **7-answer** test
- fixes for **architecture essay** scripts affecting **social media workflow** answers

Every new rule (promote scripted, protect natural, session prior) trades one error type for another.

**Takeaway:** Expect **iterative tuning** with regression checks on multiple labeled interviews, not one golden demo.

---

## 16. What you are not detecting

Be clear about scope. This class of system generally does **not** detect:

- reading from a screen the camera cannot see
- answers typed into ChatGPT live
- someone listening to an earpiece
- cheating with notes off-camera

It infers **script-reading-like delivery and content** from audio and text — probabilistic, explainable, not legal proof.

**Takeaway:** Position the product as **decision support**, not a lie detector.

---

## 17. Summary checklist for a new builder

Before you invest months in models, confirm you can handle:

| Challenge | Question to answer |
|-----------|-------------------|
| Baseline | How will you calibrate **per candidate**? |
| Speakers | How will you exclude the **interviewer**? |
| Labels | Do you have **ground-truth** scripted vs natural answers? |
| Channels | Acoustic + linguistic + content — **who wins** on conflict? |
| Fluency trap | Will fluent honest speakers be flagged? |
| Script trap | Will memorized technical prose be missed? |
| Session bias | Will answer 5 be influenced by answers 1–4? |
| Runtime | Is **5–15 min per interview** acceptable on CPU? |
| UX | Can you explain **AMBIGUOUS** to a hiring manager? |
| Scope | Are stakeholders okay with **probable**, not proven? |

---

## 18. Problems we learned the hard way (one line each)

1. Scoring the **AI bot** instead of the candidate.  
2. Treating **all suspicion** as scripted — killing trust.  
3. **Empty or flat profiles** making every metric meaningless.  
4. **Session logic** marking personal “I use…” answers as uncertain.  
5. **AWS/WebSocket scripts** caught but **microservices essays** missed.  
6. **Fluent scripted** speech passing acoustic checks.  
7. **Short calibration** vs **long interview** time expectations mismatch.  
8. Remote GPU **works until the session dies** — then confusion.  
9. **Filler stripping** weakening spontaneity signals.  
10. **One demo interview** feeling “done” until a longer real run fails.

---

*This is a general awareness document derived from building SentinEL. For technical setup, see `README.md`. For architecture, see `explanation.md`. For detailed incident-style notes from our repo work, see `problems.md`.*
