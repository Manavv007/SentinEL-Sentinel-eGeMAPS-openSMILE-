# SentinEL 🛡️

SentinEL is a state-of-the-art **multi-modal interview integrity analysis engine**. By fusing acoustic, linguistic, cognitive, and (optionally) visual signals, SentinEL identifies behavioral anomalies to flag probable script-reading and memorized delivery during remote interviews.

Rather than relying on generic rules, SentinEL uses a **dual-profile contrastive design** that calibrates specifically to the candidate's unique voice and speech patterns, ensuring high-accuracy detection while minimizing false positives for honest but nervous candidates.

---

## 🚀 Key Features

*   **Dual-Profile Contrastive Engine (v5)**: Learns a candidate's reading style during a quick calibration step (**SCRIPT profile**), and dynamically extracts their spontaneous speech style (**NATURAL profile**) during the interview.
*   **Intra-Individual Behavioral Modeling**: Compares the candidate's active delivery against their own calibrated baseline (voice quality, pitch variability, micro-level turbulence) instead of generic population averages.
*   **Cognitive Sourcing Inference**: Analyzes speech timing, pause entropy, and semantic-effort covariance to classify whether answers are internally generated (authentic recalling) vs. externally sourced (script reading/listening).
*   **Semantic Specificity Layer**: Uses rule-based NLP to detect essay-like textbook descriptions (e.g., memorized AWS architecture definitions) while protecting personal narrative responses (e.g., "In my previous role, I resolved...").
*   **Performance Optimized**: Features process-wide model caching, parallel window extraction, and asynchronous local/remote execution.
*   **Web Dashboard & CLI**: Start jobs, monitor progress, and review timelines and decision trees via the interactive web client or terminal interface.

---

## 📐 High-Level Architecture & Signal Flow

```
                  ┌──────────────────────────────────────────┐
                  │          VIDEO / AUDIO INPUT             │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │    Audio Extraction & Preprocessing      │
                  │             (16 kHz Mono)                │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │      Diarization (pyannote.audio)        │
                  │    Isolates candidate speaker turns      │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │      Segmentation into Answer Turns      │
                  └──────────┬────────────────────┬──────────┘
                             │                    │
                             ▼                    ▼
                ┌────────────────────────┐    ┌────────────────────────┐
                │    4s Window Slices    │    │   WhisperX ASR (CPU)   │
                │     (2s hop interval)  │    │   or Kaggle GPU Tunnel │
                └────────────┬───────────┘    └───────────┬────────────┘
                             │                            │
                             ▼                            ▼
                ┌────────────────────────┐    ┌────────────────────────┐
                │   Acoustic features    │    │  Linguistic features   │
                │ (openSMILE/Parselmouth)│    │ (WPS, Pause Entropy)   │
                └────────────┬───────────┘    └───────────┬────────────┘
                             │                            │
                             └─────────────┬──────────────┘
                                           │
                                           ▼
                  ┌──────────────────────────────────────────┐
                  │       Dual-Profile Contrastive Scorer    │
                  │        (SCRIPT vs. NATURAL memory)       │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │    8-Layer Decision & Aggregation Hub    │
                  │   Fuses signals and applies thresholds   │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │         EXPLAINABLE FINAL VERDICT        │
                  │   CLEAR / AMBIGUOUS / PROBABLE SCRIPT    │
                  └──────────────────────────────────────────┘
```

### The 8 Scoring Layers

SentinEL evaluates integrity through successive filters rather than a single formula:
1.  **Window-Level Contrastive Tiers**: Computes `script_similarity - natural_similarity` per 4s window.
2.  **Answer-Level Behavioral Synthesis**: Evaluates streak persistence, density, and cognitive spontaneity index per answer turn.
3.  **Cognitive Sourcing Inference**: Determines likelihood of external reading vs. internal recall.
4.  **Fused Multi-Channel Scorer**: Aggregates normalized acoustic, linguistic, specificity, and optional GPU channels.
5.  **Intra-Individual Session Drift**: Adjusts criteria relative to candidate baseline deviations and session-level voice drift.
6.  **Semantic Specificity Filter**: Demotes scores for conversational/personal narrative; promotes for generic textbook jargon.
7.  **Session-Level Sourcing Aggregation**: Refines per-answer probabilities based on whole-interview consistency.
8.  **Final Semantic Authority Pass**: Protects narrative-rich answers from session-level downgrades.

---

## 🛠️ Installation & Setup (Local CPU Mode)

SentinEL runs locally on CPU by default. 

### Prerequisites

*   **Python 3.10** (Required. Do not use 3.9 or 3.11+).
*   **FFmpeg**: Must be installed and added to your system `PATH`.
    *   **Windows**: Download builds from [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
    *   **macOS**: `brew install ffmpeg`
    *   **Linux**: `sudo apt install ffmpeg`

### Step 1: Install PyTorch (CPU-Optimized)
Installing the CPU-specific wheel saves ~2GB of space and avoids CUDA library version mismatches.
```bash
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install WhisperX
```bash
pip install whisperx==3.1.5
```

### Step 3: Install Remaining Requirements
```bash
pip install -r requirements.txt
```

### Step 4: Environment Variables & HuggingFace Models
SentinEL uses **pyannote.audio** for speaker segmentation, which requires agreeing to user terms.

1.  Copy [`.env.example`](file:///c:/Users/Manav/Downloads/openHands2/.env.example) to Create a new file [`.env`](file:///c:/Users/Manav/Downloads/openHands2/.env):
    ```bash
    cp .env.example .env
    ```
2.  Add your HuggingFace User Token (`HF_TOKEN`) inside your [`.env`](file:///c:/Users/Manav/Downloads/openHands2/.env).
3.  Accept the user terms on HuggingFace for these three repositories:
    *   [Speaker Diarization 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   [Segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0)
    *   [WeSpeaker VoxCeleb ResNet34 LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

---

## 💻 How to Run

SentinEL has a modular command-line interface as well as an interactive web app.

### Command Line Interface (CLI)

#### 1. Calibrate (SCRIPT Profile Generation)
Run calibration on a short (30–60 second) video of the candidate reading a paragraph aloud.
```bash
python main.py calibrate --video caliberation_file/pre-train-demo.mp4 --output calibration_profile.json
```

#### 2. Analyze (Interview Recording Analysis)
Analyze the actual interview video against the generated calibration profile.
```bash
python main.py analyze --video interview_files/demo-8.webm --calibration calibration_profile.json --output results.json
```

#### 3. Report (Human-Readable Summary)
Summarize and print the findings from `results.json` directly to the terminal.
```bash
python main.py report --results results.json
```

---

### Web Dashboard (Recommended)

SentinEL includes an interactive FastAPI-based web frontend to upload videos, view live processing logs, check candidate timeline charts, and drill down into per-answer decision logs.

1.  Ensure you have web dependencies:
    ```bash
    pip install fastapi uvicorn python-multipart
    ```
2.  Launch the web server using the helper script:
    *   **Windows**: Run [run_web.ps1](file:///c:/Users/Manav/Downloads/openHands2/run_web.ps1) or [restart_web.ps1](file:///c:/Users/Manav/Downloads/openHands2/restart_web.ps1) in PowerShell.
    *   **macOS / Linux**: `uvicorn web.app:app --host 127.0.0.1 --port 8765`
3.  Open **[http://127.0.0.1:8765](http://127.0.0.1:8765)** in your browser.

> [!IMPORTANT]
> Always launch the web server via `.\run_web.ps1` (or using the exact Python executable where `whisperx` is installed) to avoid import errors.

---

## ☁️ Optional: Kaggle GPU Offload

If you want faster transcription or wish to run Whisper `large-v3` without overloading your local CPU, you can offload transcription and feature extraction to a Kaggle GPU instance.

### Step 1: Start the Remote Server
1.  Upload [kaggle_gpu_server.ipynb](file:///c:/Users/Manav/Downloads/openHands2/kaggle_gpu_server.ipynb) to Kaggle (or create a new notebook with its cells).
2.  Enable GPU (T4 x2 or P100 recommended).
3.  Add `HF_TOKEN` and `NGROK_AUTHTOKEN` as Secrets in your Kaggle notebook settings.
4.  Run Cell 1 (installing dependencies) -> Restart Session -> Run Cell 2 (launches the server and ngrok tunnel).
5.  Copy the printed public ngrok URL (e.g., `https://xxxx.ngrok-free.dev`).

### Step 2: Configure your Local `.env`
Update your local [`.env`](file:///c:/Users/Manav/Downloads/openHands2/.env) file with the tunnel configuration:
```env
KAGGLE_GPU_URL=https://xxxx.ngrok-free.dev
KAGGLE_SECRET=your_shared_secret_here
SENTINEL_SECRET=your_shared_secret_here
SKIP_LOCAL_WHISPER_WHEN_KAGGLE=true
```

### Step 3: Verify the Connection
Test the bridge before running analysis:
```bash
python scripts/test_kaggle_gpu.py
```

---

## ⚙️ Configuration Reference (`.env`)

You can tune SentinEL's behavior by modifying the environment variables in [`.env`](file:///c:/Users/Manav/Downloads/openHands2/.env).

### Primary Detection Thresholds

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `ENABLE_CONTRASTIVE_ENGINE` | `true` | Enables dual-profile (SCRIPT vs NATURAL) similarity tracking. |
| `CONTRASTIVE_MARGIN` | `0.14` | The similarity margin threshold above which windows are flagged as suspicious. |
| `ENABLE_INTRA_INDIVIDUAL` | `true` | Enable comparisons relative to the speaker's own baseline profile. |
| `ENABLE_COGNITIVE_SOURCING` | `true` | Activates internal vs. external speech-sourcing analysis. |
| `ALERT_THRESHOLD` | `0.55` | The base threshold for the fused scoring algorithm to raise alerts. |

### Speed & Performance Tuning

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FAST_CALIBRATION` | `true` | Skips speaker diarization during calibration, speeding up the process. |
| `WHISPER_MODEL_SIZE` | `small` | Whisper model used for local ASR (`tiny`, `base`, `small`, `medium`). |
| `WHISPER_CALIBRATION_MODEL_SIZE` | `tiny` | Smaller Whisper size to expedite the calibration step. |
| `AUDIO_WINDOW_PARALLEL_WORKERS` | `4` | Number of parallel threads extracting acoustic features. |
| `WHISPER_SKIP_ALIGN_INTERVIEW` | `true` | Skips wav2vec alignment for interview turns to save ~1–3s per answer on CPU. |

---

## 📂 Project Structure

*   [**`main.py`**](file:///c:/Users/Manav/Downloads/openHands2/main.py): CLI interface entry point.
*   [**`config.py`**](file:///c:/Users/Manav/Downloads/openHands2/config.py): Configuration schema loader.
*   [**`processors/`**](file:///c:/Users/Manav/Downloads/openHands2/processors):
    *   [`audio_processor.py`](file:///c:/Users/Manav/Downloads/openHands2/processors/audio_processor.py): Audio feature extraction (openSMILE/Parselmouth) and turn isolation.
    *   [`transcript_processor.py`](file:///c:/Users/Manav/Downloads/openHands2/processors/transcript_processor.py): Local WhisperX speech-to-text runner.
    *   [`speaker_selection.py`](file:///c:/Users/Manav/Downloads/openHands2/processors/speaker_selection.py): Candidate-to-AI speaker assignment logic.
*   [**`engine/`**](file:///c:/Users/Manav/Downloads/openHands2/engine):
    *   [`contrastive_engine.py`](file:///c:/Users/Manav/Downloads/openHands2/engine/contrastive_engine.py): SCRIPT/NATURAL profile comparative engine.
    *   [`personal_baseline.py`](file:///c:/Users/Manav/Downloads/openHands2/engine/personal_baseline.py): Stores and tracks individual baseline parameters.
    *   [`intra_individual.py`](file:///c:/Users/Manav/Downloads/openHands2/engine/intra_individual.py): Session-level individual deviation manager.
    *   [`cognitive_sourcing.py`](file:///c:/Users/Manav/Downloads/openHands2/engine/cognitive_sourcing.py): Cognitive generation/retrieval source classifier.
    *   [`semantic_specificity.py`](file:///c:/Users/Manav/Downloads/openHands2/engine/semantic_specificity.py): Rule-based transcript NLP specificity scorer.
    *   [`answer_synthesis.py`](file:///c:/Users/Manav/Downloads/openHands2/engine/answer_synthesis.py): Fuses all layer signals to yield final answer-level verdicts.
*   [**`web/`**](file:///c:/Users/Manav/Downloads/openHands2/web):
    *   [`app.py`](file:///c:/Users/Manav/Downloads/openHands2/web/app.py): Web service routes and asynchronous job manager.
*   [**`kaggle_gpu_server.ipynb`**](file:///c:/Users/Manav/Downloads/openHands2/kaggle_gpu_server.ipynb): Offload server notebook for Kaggle environment.
