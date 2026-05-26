"""
FastAPI server for Kaggle notebook GPU offload.

Run inside a Kaggle notebook with GPU enabled, then expose via ngrok
and set KAGGLE_GPU_URL in your local .env.
"""

from __future__ import annotations

import config
from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="SentinEL GPU Server")


def _verify_secret(x_sentinel_secret: str | None) -> None:
    if x_sentinel_secret != config.SENTINEL_SECRET:
        raise HTTPException(status_code=401, detail="Invalid SENTINEL_SECRET")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(
    *,
    x_sentinel_secret: str | None = Header(default=None),
) -> dict[str, str]:
    """Placeholder — wire WhisperX transcription here."""
    _verify_secret(x_sentinel_secret)
    return {"status": "not_implemented", "message": "Attach WhisperX pipeline"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
