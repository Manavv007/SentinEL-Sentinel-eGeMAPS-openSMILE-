"""Quick check that Kaggle GPU server is reachable (loads .env)."""

from __future__ import annotations

import sys

import config
from gpu_client import KaggleGPUClient


def main() -> int:
    url = config.KAGGLE_GPU_URL
    secret = config.SENTINEL_SECRET
    if not url:
        print("KAGGLE_GPU_URL is not set in .env")
        return 1

    print(f"URL: {url}")
    print(f"Secret configured: {bool(secret)}")

    client = KaggleGPUClient(base_url=url, secret=secret, timeout=60)
    try:
        health = client.health()
        print("Health:", health)
        if health.get("status") == "ok":
            print("OK — Kaggle GPU server is live.")
            return 0
        print("Unexpected health response.")
        return 2
    except Exception as exc:
        print(f"FAILED: {exc}")
        print(
            "\nTroubleshooting:\n"
            "  1. Kaggle notebook running with GPU enabled?\n"
            "  2. Ran all cells including ngrok (URL changes each session)?\n"
            "  3. KAGGLE_GPU_URL matches printed URL (no trailing path)?\n"
            "  4. KAGGLE_SECRET matches notebook KAGGLE_SECRET?\n"
        )
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
