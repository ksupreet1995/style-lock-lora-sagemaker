"""Read back the logged runs from MLflow to confirm what persisted.

Also the query you would run to compare candidate LoRA versions
side by side.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

EXPERIMENT = config.EXPERIMENT


def fmt(v, spec=".3f"):
    """Metric names differ between versions, so tolerate absences."""
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def main():
    mlflow.set_tracking_uri(config.tracking_arn())
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"experiment {EXPERIMENT} not found")

    print(f"experiment: {exp.name}  (id={exp.experiment_id})")

    runs = client.search_runs([exp.experiment_id], max_results=50)
    print(f"runs: {len(runs)}\n")

    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]


    for p in sorted(parents, key=lambda r: r.data.tags.get("mlflow.runName", "")):
        name = p.data.tags.get("mlflow.runName", "?")
        m = p.data.metrics
        print(f"PARENT  {name}")
        print(f"  status   : {p.info.status}")
        print(f"  outcome  : {p.data.tags.get('outcome', 'approved/active')}")
        print(f"  duration : {m.get('duration_min')} min   "
              f"cost ${m.get('est_cost_usd')}")
        print(f"  rank     : {p.data.params.get('hp_rank')}   "
              f"steps: {p.data.params.get('hp_max-steps')}   "
              f"targets: {p.data.params.get('target_modules', 'attn')}")
        print(f"  holdout  : style={fmt(m.get('headline_holdout_style_score'))} "
              f"cov_err={fmt(m.get('headline_holdout_coverage_error'))}")
        if "headline_style_score_lora" in m:
            print(f"  ab       : lora={fmt(m.get('headline_style_score_lora'))} "
                  f"base={fmt(m.get('headline_style_score_base'))} "
                  f"stability={fmt(m.get('headline_stability_ratio'), '.1f')}x")

        kids = [c for c in children
                if c.data.tags["mlflow.parentRunId"] == p.info.run_id]
        for c in kids:
            cname = c.data.tags.get("mlflow.runName", "?")
            print(f"\n  CHILD  {cname}  ({c.info.status})")
            for k, v in sorted(c.data.metrics.items()):
                print(f"    {k:<34} {v:.4f}")
            arts = client.list_artifacts(c.info.run_id, "comparisons")
            if arts:
                print(f"    artifacts: {[a.path.split('/')[-1] for a in arts]}")

        print()

        # Confirm the stepped metrics came back as a chartable series.
        drift = next((c for c in kids
                      if c.data.tags.get("mlflow.runName") == "eval-drift"), None)
        if drift:
            for series in ("style_score_base_by_pass", "style_score_lora_by_pass"):
                hist = client.get_metric_history(drift.info.run_id, series)
                vals = [round(h.value, 3) for h in sorted(hist, key=lambda h: h.step)]
                print(f"  {series}: {vals}")


if __name__ == "__main__":
    main()
