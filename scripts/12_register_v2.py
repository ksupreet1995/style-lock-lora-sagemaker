"""Register v2 in the Model Registry as Rejected, with its failure analysis.

Registering a failed experiment is not busywork. It is the point of a
registry: the next person to try "adapt the conv layers to fix composition"
can see it was tried, how it failed, and at what cost. A registry that only
holds successes is a trophy cabinet, not a record.

v2 is registered Rejected so it can never be deployed by accident while
remaining visible and auditable.
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


def sanitise(v) -> str:
    s = str(v)
    return "".join(
        ch if (ch.isalnum() or ch in " _.:/=+-@") else "-" for ch in s
    )[:256] or "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", required=True)
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    bucket = config.bucket()

    d = sm.describe_training_job(TrainingJobName=args.job_name)
    artifact = d["ModelArtifacts"]["S3ModelArtifacts"]
    hp = d.get("HyperParameters", {})
    mins = (d["TrainingEndTime"] - d["TrainingStartTime"]).total_seconds() / 60

    results = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    vc = results.get("version_comparison", {}).get("summary", {})

    report = {
        "version": 2,
        "outcome": "REJECTED",
        "hypothesis": (
            "v1 learned style but not composition (mean subject coverage 0.667 "
            "vs ground truth 0.146). Since base SDXL failed identically (0.678), "
            "the large-object prior was assumed to sit outside the attention "
            "blocks. v2 therefore extended LoRA to feed-forward, resnet conv and "
            "projection layers, doubled the dataset to 44 images across 10 "
            "subjects, added mild subject-scale variation, and lowered rank to 16."
        ),
        "result": (
            "Worse on both axes. Style score fell from 0.923 to 0.641 and "
            "coverage error rose from 0.521 to 0.686."
        ),
        "root_cause": (
            "Mode collapse. v2 produces near-identical output regardless of "
            "prompt: across potion and sword the coverage difference is <=0.002 "
            "at every LoRA weight from 0.2 to 1.0, and style is already degraded "
            "to 0.772 at weight 0.2. Adapting conv and projection layers at the "
            "same rank and alpha as attention, on only 44 images, overwhelmed the "
            "base model's structural priors and destroyed prompt conditioning. "
            "The failure cannot be tuned out with the weight dial."
        ),
        "measurements": {
            "v1_style_score_mean": vc.get("v1_style_score_mean"),
            "v2_style_score_mean": vc.get("v2_style_score_mean"),
            "v1_coverage_error_mean": vc.get("v1_coverage_error_mean"),
            "v2_coverage_error_mean": vc.get("v2_coverage_error_mean"),
            "gt_coverage_mean": vc.get("gt_coverage_mean"),
            "per_subject": results.get("version_comparison", {}).get("per_subject"),
            "weight_sweep_note": (
                "potion vs sword coverage difference at weights "
                "0.2/0.4/0.6/0.8/1.0: 0.002/0.001/0.001/0.000/0.001"
            ),
        },
        "rejected_hypotheses": [
            "Training data lacked scale variation. FALSE: v1 training coverage "
            "was already 0.1463 with std 0.047, matching the holdout exactly.",
            "Preprocessing cropped the subjects. FALSE: Resize+CenterCrop on "
            "square inputs changed coverage by only +0.002.",
            "Composition is fixable at inference. FALSE: an 8-config ablation "
            "over CFG 2.5-7.0, composition prompts and negative prompts reached "
            "0.641 coverage at best, against a 0.146 target.",
        ],
        "recommended_next_step": (
            "Keep LoRA on attention only for style, and solve composition with "
            "conditioning at inference (ControlNet layout, or a fixed canvas "
            "template) rather than in the adapter. If a structural adapter is "
            "still wanted, use a much lower rank and alpha for conv layers than "
            "for attention, and far more training data."
        ),
        "cost_usd": round(mins / 60 * config.INSTANCE_HOURLY_USD, 2),
        "duration_min": round(mins, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    key = f"{PREFIX}/evaluation/v2-rejection-{time.strftime('%Y%m%d-%H%M%S')}.json"
    s3.put_object(Bucket=bucket, Key=key,
                  Body=json.dumps(report, indent=2).encode("utf-8"),
                  ContentType="application/json")
    uri = f"s3://{bucket}/{key}"
    print(f"rejection report: {uri}")

    metadata = {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "adapter_type": "LoRA-on-UNet-attn_ff_proj",
        "lora_rank": hp.get("rank", "?"),
        "training_steps": hp.get("max-steps", "?"),
        "train_images": "44-procedurally-generated-10-subjects",
        "experiment_goal": "fix-composition-by-adapting-conv-and-projection-layers",
        "outcome": "REJECTED-mode-collapse",
        "style_score_v2": f"{vc.get('v2_style_score_mean', 0):.3f}",
        "style_score_v1": f"{vc.get('v1_style_score_mean', 0):.3f}",
        "coverage_error_v2": f"{vc.get('v2_coverage_error_mean', 0):.3f}",
        "coverage_error_v1": f"{vc.get('v1_coverage_error_mean', 0):.3f}",
        "failure_mode": "near-identical-output-regardless-of-prompt",
        "do_not_deploy": "true",
        "rejection_report": uri,
        "training_job": args.job_name,
        "registered_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    resp = sm.create_model_package(
        ModelPackageGroupName=GROUP,
        ModelPackageDescription=(
            f"REJECTED. Extended LoRA to conv/projection layers to fix "
            f"composition; caused mode collapse. Style {vc.get('v2_style_score_mean', 0):.3f} "
            f"vs v1 {vc.get('v1_style_score_mean', 0):.3f}. Kept for the record."
        ),
        InferenceSpecification={
            "Containers": [{
                "Image": config.inference_image(),
                "ModelDataUrl": artifact,
                "Environment": {"SAGEMAKER_PROGRAM": "inference.py"},
            }],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [config.INSTANCE_TYPE],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": uri}
            }
        },
        CustomerMetadataProperties={k: sanitise(v) for k, v in metadata.items()},
        ModelApprovalStatus="Rejected",
    )

    print(f"registered: {resp['ModelPackageArn']}")
    print("status    : Rejected (visible, auditable, not deployable by accident)")


if __name__ == "__main__":
    main()
