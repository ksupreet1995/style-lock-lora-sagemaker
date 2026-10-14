"""Log the LoRA demo to SageMaker managed MLflow.

Why this matters for the customer pitch: it turns "we trained a style LoRA"
into an auditable record. An art director can see which training config
produced which style scores, compare two candidate LoRAs side by side, and
promote one to the Model Registry. That is the governance story a studio
actually needs, and it is the difference between a cool demo and something
they can put in a pipeline.

Logged structure:

  experiment: sdxl-lora-style-lock
    run "train-<job>"                  <- parent: config + artifact + summary
      run "eval-ab"                    <- nested: base vs LoRA
      run "eval-drift"                 <- nested: per-pass series
      run "eval-novel"                 <- nested: per-subject scores
      run "eval-dial"                  <- nested: weight sweep

The drift and dial runs log *stepped* metrics, so MLflow renders them as
line charts. That is the visual to put on screen: a flat LoRA line against a
wandering baseline.

Windows note: MLflow prints emoji in its "View run" output and the cp1252
console cannot encode it, which kills the process mid-run and leaves runs
stuck in RUNNING. We reconfigure stdout to UTF-8 to prevent that.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Must happen before mlflow writes anything to stdout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import boto3  # noqa: E402
import mlflow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "style"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from house_style import PALETTE, STYLE_NAME, TRIGGER  # noqa: E402

REGION = config.region()
TRACKING_ARN = config.tracking_arn()
EXPERIMENT = config.EXPERIMENT

OUT = ROOT / "outputs"


def training_metadata(job_name: str) -> dict:
    """Pull the real training config and timings from the SageMaker job."""
    sm = boto3.client("sagemaker", region_name=REGION)
    d = sm.describe_training_job(TrainingJobName=job_name)

    meta = {
        "job_name": job_name,
        "status": d["TrainingJobStatus"],
        "instance_type": d["ResourceConfig"]["InstanceType"],
        "instance_count": d["ResourceConfig"]["InstanceCount"],
        "image": d["AlgorithmSpecification"]["TrainingImage"],
        "artifact": d["ModelArtifacts"]["S3ModelArtifacts"],
    }
    meta.update({f"hp_{k}": v for k, v in d.get("HyperParameters", {}).items()})

    if d.get("TrainingStartTime") and d.get("TrainingEndTime"):
        mins = (d["TrainingEndTime"] - d["TrainingStartTime"]).total_seconds() / 60
        meta["duration_min"] = round(mins, 1)
        meta["est_cost_usd"] = round(mins / 60 * config.INSTANCE_HOURLY_USD, 2)

    return meta


def log_ab(results: dict):
    ab = results.get("ab")
    if not ab:
        print("  (no ab results)")
        return
    with mlflow.start_run(run_name="eval-ab", nested=True):
        mlflow.set_tag("comparison", "fixed_seed_ab")
        mlflow.log_metric("style_score_base", ab["base"])
        mlflow.log_metric("style_score_lora", ab["lora"])
        mlflow.log_metric("style_score_delta", ab["lora"] - ab["base"])
        mlflow.log_metric("relative_improvement_pct",
                          (ab["lora"] - ab["base"]) / ab["base"] * 100)
        p = OUT / "01_ab_comparison.png"
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="comparisons")
    print(f"  ab: base={ab['base']:.3f} lora={ab['lora']:.3f}")


def log_drift(results: dict):
    drift = results.get("drift")
    if not drift:
        print("  (no drift results)")
        return
    with mlflow.start_run(run_name="eval-drift", nested=True):
        mlflow.set_tag("comparison", "iteration_drift")

        # Stepped metrics so MLflow draws these as line charts.
        for i, v in enumerate(drift["base"], start=1):
            mlflow.log_metric("style_score_base_by_pass", v, step=i)
        for i, v in enumerate(drift["lora"], start=1):
            mlflow.log_metric("style_score_lora_by_pass", v, step=i)

        base_spread = max(drift["base"]) - min(drift["base"])
        lora_spread = max(drift["lora"]) - min(drift["lora"])

        mlflow.log_metric("base_spread", base_spread)
        mlflow.log_metric("lora_spread", lora_spread)
        mlflow.log_metric("stability_ratio", base_spread / lora_spread)
        mlflow.log_metric("base_min", min(drift["base"]))
        mlflow.log_metric("lora_min", min(drift["lora"]))
        mlflow.log_metric("base_mean", sum(drift["base"]) / len(drift["base"]))
        mlflow.log_metric("lora_mean", sum(drift["lora"]) / len(drift["lora"]))

        # The headline claim is variance, not monotonic decay. Record that
        # honestly so nobody reading this later overstates it.
        mlflow.set_tag(
            "interpretation",
            "Baseline is erratic rather than monotonically decaying. "
            "Defensible claim is tighter style variance, not steady decay.",
        )

        p = OUT / "02_drift_test.png"
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="comparisons")
    print(f"  drift: base_spread={base_spread:.3f} lora_spread={lora_spread:.3f} "
          f"ratio={base_spread/lora_spread:.1f}x")


def log_novel(results: dict):
    novel = results.get("novel")
    if not novel:
        print("  (no novel results)")
        return
    with mlflow.start_run(run_name="eval-novel", nested=True):
        mlflow.set_tag("comparison", "unseen_subjects")
        for subject, score in novel.items():
            key = "style_score_" + subject.replace(" ", "_").replace("a_", "", 1)
            mlflow.log_metric(key, score)
        vals = list(novel.values())
        mlflow.log_metric("mean_style_score", sum(vals) / len(vals))
        mlflow.log_metric("min_style_score", min(vals))
        mlflow.log_metric("subjects_above_0_95", sum(1 for v in vals if v >= 0.95))
        mlflow.set_tag("weakest_subject", min(novel, key=novel.get))
        p = OUT / "03_novel_subjects.png"
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="comparisons")
    print(f"  novel: mean={sum(vals)/len(vals):.3f} min={min(vals):.3f}")


def log_dial(results: dict):
    dial = results.get("dial")
    if not dial:
        print("  (no dial results)")
        return
    with mlflow.start_run(run_name="eval-dial", nested=True):
        mlflow.set_tag("comparison", "lora_weight_sweep")

        # step must be an int, so index by weight*10.
        for w_str, score in sorted(dial.items(), key=lambda kv: float(kv[0])):
            mlflow.log_metric("style_score_by_weight_x10", score,
                              step=int(float(w_str) * 10))

        best_w = max(dial, key=dial.get)
        mlflow.log_metric("best_weight", float(best_w))
        mlflow.log_metric("best_style_score", dial[best_w])
        mlflow.log_metric("score_at_max_weight", dial[max(dial, key=lambda k: float(k))])
        mlflow.set_tag("recommended_weight_range", "0.8-1.0")
        mlflow.set_tag(
            "interpretation",
            "Monotonic climb to 0.9 then sharp drop at 1.2 where the adapter "
            "over-saturates. The dial has a working range and an optimum.",
        )
        p = OUT / "04_style_dial.png"
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="comparisons")
    print(f"  dial: best_weight={best_w} score={dial[best_w]:.3f}")


def log_holdout(results: dict):
    """Held-out ground-truth evaluation.

    This is the run that keeps the demo honest. Rule compliance says the LoRA
    wins; comparison against withheld reference art says composition is not
    learned by either arm. Both get logged.
    """
    holdout = results.get("holdout")
    if not holdout:
        print("  (no holdout results)")
        return
    summary = holdout["summary"]
    with mlflow.start_run(run_name="eval-holdout", nested=True):
        mlflow.set_tag("comparison", "holdout_ground_truth")
        mlflow.set_tag("n_holdout", summary["n_holdout"])

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        for row in holdout["per_subject"]:
            subj = row["subject"].replace(" ", "_")
            mlflow.log_metric(f"coverage_lora_{subj}", row["lora_coverage"])
            mlflow.log_metric(f"coverage_gt_{subj}", row["gt_coverage"])

        mlflow.set_tag("verdict", summary["verdict"])
        mlflow.set_tag(
            "retired_metric",
            "edge_iou was dropped: two unrelated ground-truth images scored "
            "higher IoU (0.059) than either model arm (0.023/0.034), so it "
            "measured pixel alignment rather than style.",
        )
    print(f"  holdout: lora_style={summary['lora_style_score_mean']:.3f} "
          f"base_style={summary['base_style_score_mean']:.3f} "
          f"coverage {summary['lora_coverage_mean']:.3f} vs gt "
          f"{summary['gt_coverage_mean']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", required=True, help="SageMaker training job name")
    p.add_argument("--endpoint", default=None, help="endpoint used for evaluation")
    p.add_argument("--tracking-arn", default=TRACKING_ARN)
    args = p.parse_args()

    results_path = OUT / "results.json"
    if not results_path.exists():
        raise SystemExit(f"{results_path} not found. Run 06_compare.py first.")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    sections = sorted(k for k in results if not k.startswith("_"))
    print(f"results.json sections: {sections}")

    mlflow.set_tracking_uri(args.tracking_arn)
    mlflow.set_experiment(EXPERIMENT)
    print(f"tracking server: {args.tracking_arn.split('/')[-1]}")
    print(f"experiment     : {EXPERIMENT}")

    meta = training_metadata(args.job_name)

    # Re-running this script should update the existing run for a training
    # job, not stack up duplicate parents. Without this, a few reruns leave
    # the experiment table unreadable for a screen-share.
    run_name = f"train-{args.job_name}"
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    existing = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=50,
    )
    stale = [r for r in existing if "mlflow.parentRunId" not in r.data.tags]
    for r in stale:
        for child in client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{r.info.run_id}'",
            max_results=100,
        ):
            client.delete_run(child.info.run_id)
        client.delete_run(r.info.run_id)
    if stale:
        print(f"replaced {len(stale)} previous run(s) for this job")

    with mlflow.start_run(run_name=run_name) as parent:
        # Config and provenance.
        for k, v in meta.items():
            if k in ("duration_min", "est_cost_usd"):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, v)

        mlflow.log_param("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
        mlflow.log_param("trigger_token", TRIGGER)
        mlflow.log_param("style_name", STYLE_NAME)
        mlflow.log_param("palette_size", len(PALETTE))
        mlflow.log_param("train_images", len(list((ROOT / "data" / "train").glob("*.png"))))
        mlflow.log_param("eval_endpoint", args.endpoint or "n/a")

        mlflow.set_tag("project", "stylised-art-assets")
        mlflow.set_tag("technique", "LoRA on UNet attention, text encoders frozen")
        mlflow.set_tag("dataset", "procedural synthetic (not model-generated)")
        mlflow.set_tag(
            "dataset_rationale",
            "Corpus is drawn in code, not generated by SDXL. If the base model "
            "could already produce the style from a prompt the A/B would be "
            "circular.",
        )

        # Style definition travels with the run so results are reproducible.
        for f in ("house_style.py", "draw_subjects.py"):
            mlflow.log_artifact(str(ROOT / "style" / f), artifact_path="style_definition")
        mlflow.log_artifact(str(results_path), artifact_path="raw_results")

        print("\nlogging nested evaluation runs:")
        log_ab(results)
        log_drift(results)
        log_novel(results)
        log_dial(results)
        log_holdout(results)

        # Headline metrics on the parent so the experiment table is scannable.
        if "ab" in results:
            mlflow.log_metric("headline_style_score_lora", results["ab"]["lora"])
            mlflow.log_metric("headline_style_score_base", results["ab"]["base"])
        if "drift" in results:
            bs = max(results["drift"]["base"]) - min(results["drift"]["base"])
            ls = max(results["drift"]["lora"]) - min(results["drift"]["lora"])
            mlflow.log_metric("headline_stability_ratio", bs / ls)
        if "holdout" in results:
            hs = results["holdout"]["summary"]
            mlflow.log_metric("headline_holdout_style_score", hs["lora_style_score_mean"])
            mlflow.log_metric("headline_holdout_coverage_error", hs["lora_coverage_error_mean"])

        run_id = parent.info.run_id

    print(f"\nparent run id: {run_id}")
    print(f"\nOpen the MLflow UI from SageMaker Studio, or:")
    print(f"  aws sagemaker create-presigned-mlflow-tracking-server-url \\")
    print(f"    --tracking-server-name demo --region {REGION}")


if __name__ == "__main__":
    main()
