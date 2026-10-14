"""Log the v2 experiment and version comparison to MLflow.

Creates a sibling parent run to v1 so the experiment table shows both
side by side with the same metric names. That table is the artifact to show
an art director: two candidate style LoRAs, one promoted, one rejected, with
the numbers behind the decision.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import boto3  # noqa: E402
import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REGION = config.region()
TRACKING_ARN = config.tracking_arn()
EXPERIMENT = config.EXPERIMENT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", required=True)
    args = p.parse_args()

    results = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    vc = results.get("version_comparison")
    if not vc:
        raise SystemExit("no version_comparison in results.json; run 11_compare_versions.py")

    sm = boto3.client("sagemaker", region_name=REGION)
    d = sm.describe_training_job(TrainingJobName=args.job_name)
    hp = d.get("HyperParameters", {})
    mins = (d["TrainingEndTime"] - d["TrainingStartTime"]).total_seconds() / 60

    mlflow.set_tracking_uri(TRACKING_ARN)
    mlflow.set_experiment(EXPERIMENT)

    run_name = f"train-{args.job_name}"
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    for r in client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=50,
    ):
        if "mlflow.parentRunId" not in r.data.tags:
            for c in client.search_runs(
                [exp.experiment_id],
                filter_string=f"tags.mlflow.parentRunId = '{r.info.run_id}'",
                max_results=100,
            ):
                client.delete_run(c.info.run_id)
            client.delete_run(r.info.run_id)
            print("replaced previous v2 run")

    s = vc["summary"]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("version", 2)
        mlflow.log_param("base_model", "stabilityai/stable-diffusion-xl-base-1.0")
        mlflow.log_param("target_modules", hp.get("target-modules", "attn_ff_proj"))
        mlflow.log_param("hp_rank", hp.get("rank"))
        mlflow.log_param("hp_max-steps", hp.get("max-steps"))
        mlflow.log_param("train_images", 44)
        mlflow.log_param("train_subjects", 10)
        mlflow.log_param("scale_variation", "0.85-1.1")
        mlflow.log_param("instance_type", d["ResourceConfig"]["InstanceType"])
        mlflow.log_param("artifact", d["ModelArtifacts"]["S3ModelArtifacts"])

        mlflow.log_metric("duration_min", round(mins, 1))
        mlflow.log_metric("est_cost_usd",
                          round(mins / 60 * config.INSTANCE_HOURLY_USD, 2))

        # Same metric names as v1 so the experiment table lines up.
        mlflow.log_metric("headline_holdout_style_score", s["v2_style_score_mean"])
        mlflow.log_metric("headline_holdout_coverage_error", s["v2_coverage_error_mean"])
        mlflow.log_metric("holdout_coverage_mean", s["v2_coverage_mean"])
        mlflow.log_metric("holdout_palette_jsd_fg_mean", s["v2_palette_jsd_fg_mean"])

        mlflow.set_tag("outcome", "REJECTED")
        mlflow.set_tag("registry_status", "Rejected")
        mlflow.set_tag("failure_mode", "mode collapse")
        mlflow.set_tag(
            "hypothesis",
            "Extend LoRA to feed-forward, resnet conv and projection layers so "
            "the adapter can override SDXL's large-object prior.",
        )
        mlflow.set_tag(
            "root_cause",
            "Adapting conv and projection layers at the same rank and alpha as "
            "attention, on 44 images, destroyed prompt conditioning. Potion and "
            "sword differ by <=0.002 coverage at every weight from 0.2 to 1.0, "
            "and style is already degraded to 0.772 at weight 0.2.",
        )
        mlflow.set_tag(
            "rejected_hypotheses",
            "1) data lacked scale variation - FALSE, v1 coverage was already "
            "0.1463 std 0.047; 2) preprocessing cropped subjects - FALSE, "
            "+0.002 delta; 3) fixable at inference - FALSE, 8-config ablation "
            "reached 0.641 vs 0.146 target.",
        )
        mlflow.set_tag(
            "recommendation",
            "Keep attention-only LoRA for style; solve composition with "
            "inference-time conditioning rather than adapter changes.",
        )

        with mlflow.start_run(run_name="eval-version-comparison", nested=True):
            mlflow.set_tag("comparison", "v1_vs_v2_holdout")
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.set_tag("recommendation", s["recommendation"])

            for row in vc["per_subject"]:
                subj = row["subject"].replace(" ", "_")
                mlflow.log_metric(f"v1_cov_{subj}", row["v1_coverage"])
                mlflow.log_metric(f"v2_cov_{subj}", row["v2_coverage"])
                mlflow.log_metric(f"gt_cov_{subj}", row["gt_coverage"])

            report = ROOT / "outputs" / "version_comparison.json"
            report.write_text(json.dumps(vc, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report), artifact_path="comparisons")
            report.unlink()

    print(f"logged v2 at {time.strftime('%H:%M:%S')}")
    print(f"recommendation: {s['recommendation']}")


if __name__ == "__main__":
    main()
