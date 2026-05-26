#!/usr/bin/env python3
"""SentinEL CLI — calibrate, analyze, and report."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import config
from scoring.baseline import load_baseline_profile
from services.pipeline import run_analyze, run_calibrate

logger = logging.getLogger(__name__)


def _cli_progress(percent: int, message: str, _entry: dict | None) -> None:
    if percent % 10 == 0 or percent >= 100:
        print(f"  [{percent:3d}%] {message}")


def cmd_calibrate(args: argparse.Namespace) -> int:
    video = Path(args.video)
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1

    out_path = Path(args.output or "calibration_profile.json")
    print(f"Calibrating from {video} ...")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = run_calibrate(video, output_path=out_path, progress=_cli_progress)
    profile = result["profile"]
    print(
        f"Saved calibration profile -> {out_path}\n"
        f"  Timeline: {profile.get('timeline_path')}\n"
        f"  Audio: {profile.get('calibration_answers')} answers, "
        f"{profile.get('calibration_windows')} windows\n"
        f"  Elapsed: {result['elapsed_sec']}s"
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    video = Path(args.video)
    cal_path = Path(args.calibration)
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 1
    if not cal_path.is_file():
        print(f"Error: calibration profile not found: {cal_path}", file=sys.stderr)
        return 1

    out_path = Path(args.output or "results.json")
    print(f"Analyzing {video} against {cal_path} ...")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    profile = load_baseline_profile(cal_path)
    if not profile.get("acoustic_reading_profile"):
        print("Error: calibration profile missing acoustic_reading_profile", file=sys.stderr)
        return 1

    payload = run_analyze(
        video,
        cal_path,
        output_path=out_path,
        progress=_cli_progress,
    )
    print(f"Saved results -> {out_path} ({payload.get('elapsed_sec')}s)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"Error: results not found: {results_path}", file=sys.stderr)
        return 1

    data = json.loads(results_path.read_text(encoding="utf-8"))
    answers = data.get("answers", [])
    if not answers:
        print("No answers in results file.")
        return 0

    print()
    print(f"SentinEL Report — {results_path}")
    if data.get("video"):
        print(f"Video: {data['video']}")
    if data.get("timeline_path"):
        print(f"Timeline: {data['timeline_path']}")
    print()

    contrastive_mode = data.get("contrastive_engine", False)
    if contrastive_mode:
        header = (
            f"{'#':>3}  {'Start':>7}  {'End':>7}  {'Contr':>7}  {'Conf':<6}  "
            f"{'Status':<26}  Acoustic  Linguistic  Gaze      Lip"
        )
    else:
        header = (
            f"{'#':>3}  {'Start':>7}  {'End':>7}  {'Fused':>7}  {'EWMA':>7}  "
            f"{'Status':<26}  Acoustic  Linguistic  Gaze      Lip"
        )
    print(header)
    print("-" * len(header))

    for row in answers:
        sig = row.get("signals") or {}
        bd = row.get("signal_breakdown") or {}
        conf = row.get("confidence", "")
        contr = row.get("contrastive") or {}
        score_col = (
            contr.get("ewma_score", row.get("smoothed_score", row.get("ewma_score", 0)))
            if contrastive_mode
            else row.get("raw_score", row.get("fused_score", 0))
        )
        ewma_col = (
            conf
            if contrastive_mode
            else row.get("smoothed_score", row.get("ewma_score", 0))
        )
        print(
            f"{row.get('answer_id', row.get('index', 0)):>3}  "
            f"{row.get('start_sec', 0):>7.1f}  "
            f"{row.get('end_sec', 0):>7.1f}  "
            f"{score_col:>7.4f}  "
            f"{str(ewma_col):>7}  "
            f"{row.get('status', ''):<26}  "
            f"{_bd_score(bd, sig, 'acoustic'):>8.4f}  "
            f"{_bd_score(bd, sig, 'linguistic'):>10.4f}  "
            f"{_bd_score(bd, sig, 'gaze'):>8.4f}  "
            f"{_bd_score(bd, sig, 'lip'):>8.4f}"
        )
        if contrastive_mode and contr:
            print(
                f"      script~natural margin={config.CONTRASTIVE_MARGIN:.2f}  "
                f"suspicious_ratio={contr.get('suspicious_window_ratio', 0):.2f}  "
                f"natural_samples={contr.get('natural_profile_samples', 0)}"
            )
        strongest = row.get("strongest_signal")
        if strongest:
            print(f"      strongest signal: {strongest}")

    alerts = sum(1 for a in answers if a.get("status") == "PROBABLE_SCRIPT_READING")
    print()
    print(f"Total answers: {len(answers)}  |  Alerts: {alerts}")
    print()
    return 0


def _bd_score(
    breakdown: dict,
    signals: dict,
    channel: str,
) -> float:
    entry = breakdown.get(channel, {})
    if entry.get("score") is not None:
        return float(entry["score"])
    return float(signals.get(channel, 0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="SentinEL — multi-modal script-reading detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="Build reading baseline from calibration video")
    p_cal.add_argument("--video", required=True, help="Calibration video (scripted reading)")
    p_cal.add_argument(
        "--output",
        default="calibration_profile.json",
        help="Output profile path (default: calibration_profile.json)",
    )
    p_cal.set_defaults(func=cmd_calibrate)

    p_an = sub.add_parser("analyze", help="Score interview answers against calibration")
    p_an.add_argument("--video", required=True, help="Interview video")
    p_an.add_argument(
        "--calibration",
        required=True,
        help="Path to calibration_profile.json",
    )
    p_an.add_argument(
        "--output",
        default="results.json",
        help="Output results path (default: results.json)",
    )
    p_an.set_defaults(func=cmd_analyze)

    p_rep = sub.add_parser("report", help="Print human-readable summary from results.json")
    p_rep.add_argument("--results", required=True, help="Path to results.json")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
