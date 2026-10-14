"""Register the LoRA in SageMaker Model Registry with evaluation metrics.

This is the governance payoff: "house style v1" becomes a versioned,
approvable artifact with its evaluation attached, not a prompt in someone's
notes.

Two things get attached that matter:

  ModelMetrics            S3-hosted evaluation JSON, so the metrics are
                          queryable and travel with the version. The other
                          registered groups in this account leave this null,
                          which means their evaluations live nowhere.
  CustomerMetadata        flat key/values visible in the Studio UI, including
                          the known composition limitation.

Registered as PendingManualApproval on purpose. An art director approving the
version is the workflow being demonstrated, so approving it automatically
would skip the point.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

import boto3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()
GROUP = config.MODEL_PACKAGE_GROUP
PREFIX = config.prefix()


def ensure_group(sm):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=GROUP)
        print(f"group exists: {GROUP}")
    except sm.exceptions.ClientError:
        sm.create_model_package_group(
            ModelPackageGroupName=GROUP,
            ModelPackageGroupDescription=(
                "SDXL LoRA adapters encoding the flatgem house art style. "
                "Each version is a style-bible revision an art director can "
                "approve and pin to a project."
            ),
        )
        print(f"created group: {GROUP}")


def upload_evaluation(s3, bucket: str, results: dict) -> str:
    """Publish the evaluation report to S3 so ModelMetrics can reference it."""
    holdout = results.get("holdout", {})
    report = {
        "evaluation_type": "style_adherence_and_holdout_similarity",
        "metric_definitions": {
            "style_score": "mean of palette compliance, fill flatness, shadow presence. 1.0 = full house-style compliance",
            "subject_coverage": "fraction of canvas occupied by the subject. house style expects ~0.15",
            "palette_jsd_fg": "Jensen-Shannon divergence of foreground palette histograms vs ground truth. 0 = identical",
            "stability_ratio": "baseline style-score spread divided by LoRA spread across 5 iteration passes",
        },
        "style_adherence": {
            "ab_base": results.get("ab", {}).get("base"),
            "ab_lora": results.get("ab", {}).get("lora"),
            "novel_subjects": results.get("novel"),
            "weight_sweep": results.get("dial"),
        },
        "iteration_stability": results.get("drift"),
        "holdout_evaluation": holdout,
        "known_limitations": [
            "Composition is not learned. Mean subject coverage 0.667 vs ground "
            "truth 0.146. Both the LoRA and the prompt-engineered baseline fail "
            "this almost identically (0.667 vs 0.678), so it is a conditioning "
            "gap rather than a LoRA regression. Fix with varied subject scale in "
            "training data plus ControlNet or layout conditioning at inference.",
            "Style dial degrades sharply above weight 1.0 (0.977 at 0.9 falling "
            "to 0.657 at 1.2). Recommended operating range 0.8-1.0.",
            "Trained on 24 procedurally generated images. Subject coverage is "
            "narrow; the weakest novel subject scored 0.783.",
            "Iteration-drift finding is reduced variance (4.9x tighter spread), "
            "not monotonic decay. The baseline is erratic rather than steadily "
            "degrading.",
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    key = f"{PREFIX}/evaluation/report-{time.strftime('%Y%m%d-%H%M%S')}.json"
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(report, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    uri = f"s3://{bucket}/{key}"
    print(f"evaluation report: {uri}")
    return uri


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", default=None,
                   help="defaults to the most recent completed sdxl-lora job")
    p.add_argument("--approve", action="store_true",
                   help="register as Approved instead of PendingManualApproval")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    bucket = config.bucket()

    job_name = args.job_name or config.latest_training_job()
    d = sm.describe_training_job(TrainingJobName=job_name)
    if d["TrainingJobStatus"] != "Completed":
        raise SystemExit(f"training job status is {d['TrainingJobStatus']}")
    artifact = d["ModelArtifacts"]["S3ModelArtifacts"]
    hp = d.get("HyperParameters", {})

    results_path = ROOT / "outputs" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))

    ensure_group(sm)
    eval_uri = upload_evaluation(s3, bucket, results)

    holdout_summary = results.get("holdout", {}).get("summary", {})
    drift = results.get("drift", {})
    stability = ""
    if drift:
        bs = max(drift["base"]) - min(drift["base"])
        ls = max(drift["lora"]) - min(drift["lora"])
        stability = f"{bs / ls:.1f}x"

    # CustomerMetadataProperties values must match
    # ([\p{L}\p{Z}\p{N}_.:\/=+\-@]*) and be <=256 chars. No parentheses,
    # commas, or semicolons. Detail lives in the S3 evaluation report instead.
    metadata = {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "base_model_license": "CreativeML-Open-RAIL++-M commercial-use-permitted",
        "adapter_type": "LoRA-on-UNet-attention",
        "adapter_targets": "to_k to_q to_v to_out.0",
        "trigger_token": "flatgem style",
        "lora_rank": hp.get("rank", "?"),
        "training_steps": hp.get("max-steps", "?"),
        "resolution": hp.get("resolution", "?"),
        "train_images": "24-procedurally-generated",
        "recommended_lora_weight": "0.8-1.0",
        "style_score_lora": f"{results.get('ab', {}).get('lora', 0):.3f}",
        "style_score_base": f"{results.get('ab', {}).get('base', 0):.3f}",
        "holdout_style_score": f"{holdout_summary.get('lora_style_score_mean', 0):.3f}",
        "holdout_coverage_error": f"{holdout_summary.get('lora_coverage_error_mean', 0):.3f}",
        "iteration_stability_ratio": stability,
        "known_limitation": "composition-not-learned-see-evaluation-report",
        "training_job": job_name,
        "evaluation_report": eval_uri,
        "registered_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def sanitise(v) -> str:
        """Coerce to the allowed CustomerMetadataProperties character set."""
        s = str(v)
        allowed = []
        for ch in s:
            if ch.isalnum() or ch in " _.:/=+-@":
                allowed.append(ch)
            else:
                allowed.append("-")
        return "".join(allowed)[:256] or "-"

    resp = sm.create_model_package(
        ModelPackageGroupName=GROUP,
        ModelPackageDescription=(
            f"flatgem style LoRA, rank {hp.get('rank', '?')}, "
            f"{hp.get('max-steps', '?')} steps. Style score "
            f"{results.get('ab', {}).get('lora', 0):.3f} vs baseline "
            f"{results.get('ab', {}).get('base', 0):.3f}; "
            f"{stability} tighter iteration variance. "
            "Composition not learned (see limitations)."
        ),
        InferenceSpecification={
            "Containers": [{
                "Image": config.inference_image(),
                "ModelDataUrl": artifact,
                "Environment": {"SAGEMAKER_PROGRAM": "inference.py"},
            }],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [
                config.INSTANCE_TYPE, "ml.g5.4xlarge", "ml.g6.2xlarge",
            ],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": eval_uri}
            }
        },
        CustomerMetadataProperties={k: sanitise(v) for k, v in metadata.items()},
        ModelApprovalStatus="Approved" if args.approve else "PendingManualApproval",
    )

    arn = resp["ModelPackageArn"]
    print(f"\nregistered: {arn}")
    print(f"status    : {'Approved' if args.approve else 'PendingManualApproval'}")
    print("\nApprove in Studio, or:")
    print(f"  aws sagemaker update-model-package --model-package-arn {arn} \\")
    print(f"    --model-approval-status Approved --region {REGION}")


if __name__ == "__main__":
    main()
