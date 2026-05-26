import json
import sys

from engine.answer_synthesis import (
    compute_answer_behavioral_metrics,
    synthesize_final_decision,
)

path = sys.argv[1] if len(sys.argv) > 1 else (
    "web_data/jobs/8ff1bf8b-9a4e-4aa0-90bf-73efbc041cc8/results.json"
)
with open(path, encoding="utf-8") as f:
    data = json.load(f)
answers = data["answers"] if "answers" in data else data["result"]["answers"]

for a in answers:
    aid = a.get("answer_id", a.get("index"))
    c = a.get("contrastive", {})
    wins = c.get("windows", [])
    if not wins:
        print(f"Answer {aid}: no windows")
        continue
    wd = [
        {
            "suspicion_level": w["suspicion_level"],
            "contrastive_score": w["contrastive_score"],
            "script_similarity": w["script_similarity"],
            "natural_similarity": w["natural_similarity"],
            "naturality_score": w["naturality_score"],
        }
        for w in wins
    ]
    mom = {
        "longest_strong_streak": c.get("longest_strong_streak", 0),
        "suspicion_momentum": c.get("suspicion_momentum", 0),
        "peak_ewma": c.get("peak_ewma", 0),
    }
    beh = compute_answer_behavioral_metrics(wd, momentum_summary=mom)
    temporal = {
        "composite_score": c.get("composite_score", 0),
        "peak_ewma": c.get("peak_ewma", 0),
        "weighted_evidence": c.get("weighted_evidence", 0),
        "status": c.get("status", "CLEAR"),
        "confidence": c.get("confidence", "LOW"),
    }
    st, conf, _ = synthesize_final_decision(temporal, beh)
    print(
        f"Answer {aid}: {temporal['status']} -> {st} "
        f"dom={beh['dominant_script_reading_score']} "
        f"peak={beh['peak_suspicion']:.3f} rec={beh['recovery_strength']:.2f} "
        f"strong={beh['strong_window_count']}"
    )
