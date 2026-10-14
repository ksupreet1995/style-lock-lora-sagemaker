"""Read traces back and show the span breakdown.

This is the output that makes the traces-vs-logs distinction concrete: a
per-span latency breakdown for a single request, which neither logs nor
metrics can give you.
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


def main():
    mlflow.set_tracking_uri(config.tracking_arn())
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"experiment {EXPERIMENT} not found")

    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id], max_results=20, return_type="list"
    )
    print(f"traces on server: {len(traces)}\n")
    if not traces:
        print("None yet. Generate some with:")
        print("  python scripts/18_trace_generation.py --endpoint <name>")
        return

    for t in traces:
        tags = t.info.tags or {}
        total = getattr(t.info, "execution_duration", None)

        print("=" * 66)
        print(f"request  : {t.info.request_id}")
        print(f"strategy : {tags.get('strategy', '?')}   "
              f"subject: {tags.get('subject', '?')}   "
              f"style_score: {tags.get('style_score', '?')}")
        print(f"state    : {t.info.state}   total: {total}ms")
        print(f"{'span':<26} {'type':<10} {'ms':>8} {'% total':>8}")
        print("-" * 66)

        for s in t.data.spans:
            ms = ((s.end_time_ns or 0) - (s.start_time_ns or 0)) / 1e6
            pct = (ms / total * 100) if total else 0
            print(f"{s.name:<26} {(s.span_type or '-')[:10]:<10} "
                  f"{ms:>8.0f} {pct:>7.1f}%")
        print()

    print("The endpoint span typically dominates. That is the actionable")
    print("detail: scoring and prompt assembly are rounding errors, so")
    print("latency work belongs at the endpoint (instance type, step count).")


if __name__ == "__main__":
    main()
