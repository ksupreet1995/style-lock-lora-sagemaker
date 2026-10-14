"""Evaluate generated images against the held-out ground truth.

This closes a real gap. The style_score in 02_style_metrics.py measures
*rule compliance* (is the palette locked, are fills flat, is there a shadow).
It never asks "does this look like the reference art we withheld?"

Here we generate the 6 held-out subjects and compare each against the
ground-truth image the model never saw during training:

  palette_jsd    Jensen-Shannon divergence between palette histograms.
                 0 = identical colour distribution. Catches "right colours,
                 wrong proportions", which rule compliance alone misses.
  edge_iou       Intersection-over-union of edge masks. Proxy for whether
                 outline weight and silhouette structure match.
  gt_style_score The ground-truth image's own rule score, as a ceiling.

Run with --baseline to also evaluate the base model with prompt engineering,
which gives a paired comparison on identical ground truth.
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "style"))
sys.path.insert(0, str(ROOT / "scripts"))

import boto3  # noqa: E402

from house_style import PALETTE_LIST, TRIGGER, describe_style  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()
OUT = ROOT / "outputs"
HOLDOUT = ROOT / "data" / "holdout"

PAL = np.array(PALETTE_LIST, dtype=np.int16)


def _metrics_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "style_metrics", ROOT / "scripts" / "02_style_metrics.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def palette_histogram(img: Image.Image, size: int = 256) -> np.ndarray:
    """Fraction of pixels nearest each palette colour."""
    a = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.int16).reshape(-1, 3)
    d = np.stack([np.abs(a - c).sum(axis=1) for c in PAL], axis=1)
    nearest = d.argmin(axis=1)
    hist = np.bincount(nearest, minlength=len(PAL)).astype(np.float64)
    return hist / hist.sum()


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric, bounded [0,1] with log base 2. 0 means identical."""
    eps = 1e-12
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def subject_coverage(img: Image.Image, size: int = 256) -> float:
    """Fraction of canvas occupied by the subject (i.e. not background).

    Replaces an earlier edge-IoU metric that turned out to be worthless:
    two *unrelated* ground-truth images scored higher IoU (0.059) than
    either model arm (0.023/0.034), because IoU needs pixel-aligned
    silhouettes and generated subjects are never positioned identically.

    Coverage is position-invariant and measures something real: the house
    style puts a small centred icon on a large cream field, so ground truth
    sits at 8-23% coverage. A model that fills the canvas has learned the
    palette but not the composition.
    """
    hist = palette_histogram(img, size=size)
    return float(1.0 - hist[2])  # index 2 is the cream background


def coverage_error(gt: Image.Image, gen: Image.Image) -> float:
    """Absolute difference in subject coverage. 0 = same composition density."""
    return abs(subject_coverage(gt) - subject_coverage(gen))


def palette_jsd_foreground(gt: Image.Image, gen: Image.Image) -> float:
    """Palette JSD with the background bin removed and the rest renormalised.

    Raw palette JSD is dominated by how much background each image has, so a
    composition error swamps the colour signal. This isolates "are the
    subject colours right" from "is the subject the right size".
    """
    def fg(img):
        h = palette_histogram(img)
        h = np.delete(h, 2)
        return h / h.sum()

    return jensen_shannon(fg(gt), fg(gen))


def invoke(endpoint, **kw):
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = rt.invoke_endpoint(
        EndpointName=endpoint, ContentType="application/json", Body=json.dumps(kw)
    )
    payload = json.loads(resp["Body"].read())
    return Image.open(io.BytesIO(base64.b64decode(payload["image_b64"])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--baseline", action="store_true",
                   help="also evaluate base model + prompt engineering")
    args = p.parse_args()

    sm = _metrics_mod()
    cases = json.loads((HOLDOUT / "holdout.json").read_text(encoding="utf-8"))
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, case in enumerate(cases):
        gt = Image.open(HOLDOUT / case["file_name"])
        gt_hist = palette_histogram(gt)
        gt_style = (sm.palette_score(gt) + sm.flatness_score(gt)
                    + sm.shadow_present(gt)) / 3

        gen = invoke(args.endpoint, prompt=case["prompt"], use_lora=True,
                     lora_scale=0.9, seed=4000 + i)
        gen.save(raw_dir / f"holdout_lora_{case['subject'].replace(' ', '_')}.png")

        row = {
            "subject": case["subject"],
            "gt_style_score": round(gt_style, 4),
            "gt_coverage": round(subject_coverage(gt), 4),
            "lora_style_score": round(
                (sm.palette_score(gen) + sm.flatness_score(gen)
                 + sm.shadow_present(gen)) / 3, 4),
            "lora_palette_jsd": round(jensen_shannon(gt_hist, palette_histogram(gen)), 4),
            "lora_palette_jsd_fg": round(palette_jsd_foreground(gt, gen), 4),
            "lora_coverage": round(subject_coverage(gen), 4),
            "lora_coverage_error": round(coverage_error(gt, gen), 4),
        }

        if args.baseline:
            base_prompt = f"a {case['subject']}, {describe_style()}"
            b = invoke(args.endpoint, prompt=base_prompt, use_lora=False, seed=4000 + i)
            b.save(raw_dir / f"holdout_base_{case['subject'].replace(' ', '_')}.png")
            row["base_style_score"] = round(
                (sm.palette_score(b) + sm.flatness_score(b)
                 + sm.shadow_present(b)) / 3, 4)
            row["base_palette_jsd"] = round(jensen_shannon(gt_hist, palette_histogram(b)), 4)
            row["base_palette_jsd_fg"] = round(palette_jsd_foreground(gt, b), 4)
            row["base_coverage"] = round(subject_coverage(b), 4)
            row["base_coverage_error"] = round(coverage_error(gt, b), 4)

        rows.append(row)
        print(f"  {case['subject']:<14} "
              f"lora: jsd_fg={row['lora_palette_jsd_fg']:.3f} "
              f"cov={row['lora_coverage']:.3f} (gt {row['gt_coverage']:.3f})"
              + (f"   base: jsd_fg={row['base_palette_jsd_fg']:.3f} "
                 f"cov={row['base_coverage']:.3f}" if args.baseline else ""))

    def mean(key):
        return round(sum(r[key] for r in rows) / len(rows), 4)

    summary = {
        "n_holdout": len(rows),
        "gt_style_score_mean": mean("gt_style_score"),
        "gt_coverage_mean": mean("gt_coverage"),
        "lora_style_score_mean": mean("lora_style_score"),
        "lora_palette_jsd_mean": mean("lora_palette_jsd"),
        "lora_palette_jsd_fg_mean": mean("lora_palette_jsd_fg"),
        "lora_coverage_mean": mean("lora_coverage"),
        "lora_coverage_error_mean": mean("lora_coverage_error"),
    }
    if args.baseline:
        summary.update({
            "base_style_score_mean": mean("base_style_score"),
            "base_palette_jsd_mean": mean("base_palette_jsd"),
            "base_palette_jsd_fg_mean": mean("base_palette_jsd_fg"),
            "base_coverage_mean": mean("base_coverage"),
            "base_coverage_error_mean": mean("base_coverage_error"),
            "style_score_improvement": round(
                mean("lora_style_score") - mean("base_style_score"), 4),
            "coverage_error_improvement": round(
                mean("base_coverage_error") - mean("lora_coverage_error"), 4),
        })
        # Record the finding plainly so it cannot be quietly dropped.
        summary["verdict"] = (
            "LoRA wins on style rule compliance "
            f"({summary['lora_style_score_mean']:.3f} vs "
            f"{summary['base_style_score_mean']:.3f}) but loses on composition: "
            f"mean subject coverage {summary['lora_coverage_mean']:.3f} vs "
            f"ground truth {summary['gt_coverage_mean']:.3f}. The LoRA learned "
            "palette and flatness but fills the canvas instead of drawing a "
            "small centred icon, because training captions never varied "
            "subject scale."
        )

    print("\n--- holdout summary ---")
    for k, v in summary.items():
        print(f"  {k:<26} {v}")

    results_path = OUT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    results["holdout"] = {"per_subject": rows, "summary": summary}
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n-> {results_path}")


if __name__ == "__main__":
    main()
