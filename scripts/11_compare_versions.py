"""Compare two registered LoRA versions on the same holdout set.

This is the full MLOps loop made visible: evaluate, find a flaw, retrain,
compare versions, promote the better one. It is a stronger story than one
version plus a caveat, because it shows the registry doing work rather than
just storing a file.

Requires both endpoints to be live. Deploy v2 with 05_deploy_endpoint.py
before running.
"""

import argparse
import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import boto3
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "style"))

spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "09_evaluate_holdout.py")
ev = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ["x", "--endpoint", "d"]
spec.loader.exec_module(ev)
sys.argv = _argv
sm_metrics = ev._metrics_mod()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()
OUT = ROOT / "outputs"
HOLDOUT = ROOT / "data" / "holdout"


def invoke(endpoint, **kw):
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    r = rt.invoke_endpoint(EndpointName=endpoint, ContentType="application/json",
                           Body=json.dumps(kw))
    p = json.loads(r["Body"].read())
    return Image.open(io.BytesIO(base64.b64decode(p["image_b64"])))


def style(img):
    return (sm_metrics.palette_score(img) + sm_metrics.flatness_score(img)
            + sm_metrics.shadow_present(img)) / 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1-endpoint", required=True)
    p.add_argument("--v2-endpoint", required=True)
    p.add_argument("--lora-scale", type=float, default=0.9)
    args = p.parse_args()

    cases = json.loads((HOLDOUT / "holdout.json").read_text(encoding="utf-8"))
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'subject':<14} {'gt_cov':>7} {'v1_cov':>7} {'v2_cov':>7} "
          f"{'v1_sty':>7} {'v2_sty':>7}")
    print("-" * 56)

    for i, case in enumerate(cases):
        gt = Image.open(HOLDOUT / case["file_name"])
        gt_cov = ev.subject_coverage(gt)
        subj = case["subject"].replace(" ", "_")

        # v2 was trained with "icon" in captions; use each version's own
        # caption convention so neither is handicapped.
        imgs = {}
        for ver, ep, prompt in [
            ("v1", args.v1_endpoint, case["prompt"]),
            ("v2", args.v2_endpoint, f"{case['prompt']} icon"),
        ]:
            img = invoke(ep, prompt=prompt, use_lora=True,
                         lora_scale=args.lora_scale, seed=4000 + i)
            img.save(raw / f"vercmp_{ver}_{subj}.png")
            imgs[ver] = img

        row = {
            "subject": case["subject"],
            "gt_coverage": round(gt_cov, 4),
            "v1_coverage": round(ev.subject_coverage(imgs["v1"]), 4),
            "v2_coverage": round(ev.subject_coverage(imgs["v2"]), 4),
            "v1_coverage_error": round(ev.coverage_error(gt, imgs["v1"]), 4),
            "v2_coverage_error": round(ev.coverage_error(gt, imgs["v2"]), 4),
            "v1_style_score": round(style(imgs["v1"]), 4),
            "v2_style_score": round(style(imgs["v2"]), 4),
            "v1_palette_jsd_fg": round(ev.palette_jsd_foreground(gt, imgs["v1"]), 4),
            "v2_palette_jsd_fg": round(ev.palette_jsd_foreground(gt, imgs["v2"]), 4),
        }
        rows.append(row)
        print(f"{row['subject']:<14} {row['gt_coverage']:>7.3f} "
              f"{row['v1_coverage']:>7.3f} {row['v2_coverage']:>7.3f} "
              f"{row['v1_style_score']:>7.3f} {row['v2_style_score']:>7.3f}")

    def mean(k):
        return round(sum(r[k] for r in rows) / len(rows), 4)

    summary = {
        "n_holdout": len(rows),
        "gt_coverage_mean": mean("gt_coverage"),
        "v1_coverage_mean": mean("v1_coverage"),
        "v2_coverage_mean": mean("v2_coverage"),
        "v1_coverage_error_mean": mean("v1_coverage_error"),
        "v2_coverage_error_mean": mean("v2_coverage_error"),
        "v1_style_score_mean": mean("v1_style_score"),
        "v2_style_score_mean": mean("v2_style_score"),
        "v1_palette_jsd_fg_mean": mean("v1_palette_jsd_fg"),
        "v2_palette_jsd_fg_mean": mean("v2_palette_jsd_fg"),
    }
    summary["coverage_error_reduction"] = round(
        summary["v1_coverage_error_mean"] - summary["v2_coverage_error_mean"], 4)
    summary["style_score_change"] = round(
        summary["v2_style_score_mean"] - summary["v1_style_score_mean"], 4)

    better_comp = summary["coverage_error_reduction"] > 0.05
    kept_style = summary["style_score_change"] > -0.05
    summary["recommendation"] = (
        "promote v2" if (better_comp and kept_style)
        else "keep v1" if not better_comp
        else "v2 fixes composition but regresses style - needs review"
    )

    print("\n--- version comparison ---")
    for k, v in summary.items():
        print(f"  {k:<28} {v}")

    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["version_comparison"] = {"per_subject": rows, "summary": summary}
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n-> {results_path}")


if __name__ == "__main__":
    main()
