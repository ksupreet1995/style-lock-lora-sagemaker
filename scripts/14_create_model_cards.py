"""Create SageMaker Model Cards for both LoRA versions.

Model Registry and Model Cards do different jobs, and a governance story
needs both:

  Model Registry  deployment lineage. Versions, approval status, container
                  spec, metric pointers. Machine-facing, gates deployment.
  Model Card      governance narrative. Intended use, limitations, risk
                  rating, ethical considerations. Human-facing, answers
                  "should we use this and what are we accepting?"

The card is the artifact a reviewer or approver actually reads. It is also
where base-model licensing and known limitations belong in prose rather than
as a sanitised metadata key.

v1 gets a card documenting an approved model with a known limitation.
v2 gets a card documenting a rejected experiment, because the reasoning
behind a rejection is institutional knowledge worth keeping.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

import boto3  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REGION = config.region()
CARD_V1 = os.environ.get("MODEL_CARD_V1", "style-lora-v1")
CARD_V2 = os.environ.get("MODEL_CARD_V2", "style-lora-v2-rejected")


def num(v, default=0.0):
    return float(v) if isinstance(v, (int, float)) else default


def v1_content(results: dict, meta: dict) -> dict:
    ab = results.get("ab", {})
    drift = results.get("drift", {})
    novel = results.get("novel", {})
    dial = results.get("dial", {})
    holdout = results.get("holdout", {}).get("summary", {})

    stability = 0.0
    if drift:
        bs = max(drift["base"]) - min(drift["base"])
        ls = max(drift["lora"]) - min(drift["lora"])
        stability = bs / ls

    return {
        "model_overview": {
            "model_name": CARD_V1,
            "model_description": (
                "SDXL LoRA adapter encoding the 'flatgem' house art style for "
                "game inventory icons. Trained on 24 procedurally generated "
                "images with a locked 12-colour palette, uniform outlines, flat "
                "fills and hard offset shadows. LoRA is applied to UNet "
                "attention projections only (to_k, to_q, to_v, to_out.0) at "
                "rank 32 over 1200 optimiser steps; both text encoders remain "
                "frozen. Trigger token: 'flatgem style'. "
                "Purpose: move style from the prompt into the weights so it is "
                "a versioned artifact rather than a per-artist prompt."
            ),
            "model_version": 1,
            "problem_type": "Image generation - style transfer",
            "algorithm_type": "LoRA fine-tuning of latent diffusion UNet",
            "model_creator": os.environ.get("MODEL_CREATOR", "ML team"),
            "model_artifact": [meta.get("artifact", "")],
            "inference_environment": {
                "container_image": [config.inference_image()]
            },
        },
        "intended_uses": {
            "purpose_of_model": (
                "Generate game inventory icons that adhere to a fixed house art "
                "style, and hold that style stable across iterative artist "
                "revisions."
            ),
            "intended_uses": (
                "Concept iteration and asset drafting inside an art pipeline "
                "that already supplies composition control. Style consistency "
                "across many assets produced by different artists. Versioned "
                "'style bible' artifact that an art director can approve and pin "
                "to a project."
            ),
            "factors_affecting_model_efficiency": (
                "LoRA weight: usable range 0.8-1.0, optimum 0.9, degrades "
                "sharply at 1.2 (style score 0.977 at 0.9 falling to 0.657 at "
                "1.2). Subject familiarity: subjects far from the 6 training "
                "categories score lower (weakest novel subject 0.783 against a "
                "~0.96 typical). Composition: the adapter does not control "
                "subject scale or placement and requires external conditioning."
            ),
            "risk_rating": "Low",
            "explanations_for_risk_rating": (
                "Generates stylised game art assets, not decisions about people. "
                "No PII in training data, which is procedurally generated rather "
                "than scraped or model-generated. Base model SDXL 1.0 is licensed "
                "under CreativeML Open RAIL++-M, which permits commercial use. "
                "Primary risk is wasted artist time if the known composition "
                "limitation is not handled by the surrounding pipeline."
            ),
        },
        "training_details": {
            "objective_function": {
                "function": {
                    "function": "Other",
                    "facet": "MSE between predicted and sampled noise (epsilon prediction)",
                }
            },
            "training_observations": (
                "Loss fell from 0.0147 to the 0.003-0.010 range with the "
                "variance normal for diffusion training. Training images score "
                "0.996 on the style metric by construction, which validates the "
                "metric before any generation. Captions deliberately omit all "
                "style words so palette, outline and shadow bind to the trigger "
                "token rather than smearing across common vocabulary."
            ),
            "training_job_details": {
                "training_arn": meta.get("training_job_arn", ""),
                "training_datasets": [
                    "24 procedurally generated 512x512 icons, 6 subjects x 5 "
                    "accent colours, 6 holdout combinations excluded"
                ],
                "training_environment": {
                    "container_image": [config.training_image()]
                },
                "hyper_parameters": [
                    {"name": "rank", "value": "32"},
                    {"name": "lora_alpha", "value": "32"},
                    {"name": "target_modules", "value": "to_k, to_q, to_v, to_out.0"},
                    {"name": "max_steps", "value": "1200"},
                    {"name": "gradient_accumulation", "value": "4"},
                    {"name": "learning_rate", "value": "1e-4"},
                    {"name": "resolution", "value": "1024"},
                    {"name": "base_model", "value": "stabilityai/stable-diffusion-xl-base-1.0"},
                ],
            },
        },
        "evaluation_details": [
            {
                "name": "Style adherence on unseen subject (fixed seed A/B)",
                "evaluation_observation": (
                    "LoRA scores 0.984 against 0.866 for the base model with "
                    "deliberately strong prompt engineering, a 13.6% relative "
                    "improvement on a subject absent from training."
                ),
                "metric_groups": [{
                    "name": "Style score",
                    "metric_data": [
                        {"name": "style_score_lora", "type": "number",
                         "value": num(ab.get("lora"))},
                        {"name": "style_score_base_prompted", "type": "number",
                         "value": num(ab.get("base"))},
                    ],
                }],
            },
            {
                "name": "Iteration stability",
                "evaluation_observation": (
                    "Across 5 passes with accumulating edits the LoRA holds "
                    "style within a 0.039 spread versus 0.194 for the prompted "
                    "baseline, a 4.9x reduction in variance. Note the baseline "
                    "is erratic rather than monotonically decaying, so the "
                    "defensible claim is tighter variance, not steady decay."
                ),
                "metric_groups": [{
                    "name": "Stability",
                    "metric_data": [
                        {"name": "lora_spread", "type": "number",
                         "value": round(max(drift["lora"]) - min(drift["lora"]), 4) if drift else 0.0},
                        {"name": "base_spread", "type": "number",
                         "value": round(max(drift["base"]) - min(drift["base"]), 4) if drift else 0.0},
                        {"name": "stability_ratio", "type": "number",
                         "value": round(stability, 4)},
                    ],
                }],
            },
            {
                "name": "Generalisation to novel subjects",
                "evaluation_observation": (
                    "Four subjects absent from training: spell book 0.965, "
                    "treasure chest 0.958, compass 0.957, bell 0.783. Three of "
                    "four above 0.95 supports genuine style transfer rather than "
                    "memorisation of training images. The bell outlier indicates "
                    "subject coverage is narrow at 24 training images."
                ),
                "metric_groups": [{
                    "name": "Novel subject style scores",
                    "metric_data": [
                        {"name": k.replace("a ", ""), "type": "number", "value": num(v)}
                        for k, v in novel.items()
                    ],
                }],
            },
            {
                "name": "Held-out ground truth evaluation",
                "evaluation_observation": (
                    "Against 6 withheld reference images never seen in training, "
                    "the LoRA scores 0.923 on style rules versus 0.786 for the "
                    "prompted baseline. However mean subject coverage is 0.667 "
                    "against a ground-truth 0.146: the model fills the canvas "
                    "instead of drawing a small centred icon. The prompted "
                    "baseline fails almost identically at 0.678, which "
                    "identifies this as a base-model prior rather than a LoRA "
                    "regression."
                ),
                "metric_groups": [{
                    "name": "Holdout metrics",
                    "metric_data": [
                        {"name": "lora_style_score_mean", "type": "number",
                         "value": num(holdout.get("lora_style_score_mean"))},
                        {"name": "base_style_score_mean", "type": "number",
                         "value": num(holdout.get("base_style_score_mean"))},
                        {"name": "lora_subject_coverage_mean", "type": "number",
                         "value": num(holdout.get("lora_coverage_mean"))},
                        {"name": "ground_truth_coverage_mean", "type": "number",
                         "value": num(holdout.get("gt_coverage_mean"))},
                    ],
                }],
            },
            {
                "name": "LoRA weight sensitivity",
                "evaluation_observation": (
                    "Monotonic improvement to weight 0.9 (0.977) followed by a "
                    "sharp drop at 1.2 (0.657) where the adapter over-saturates. "
                    "Recommended operating range 0.8-1.0."
                ),
                "metric_groups": [{
                    "name": "Style score by LoRA weight",
                    "metric_data": [
                        {"name": f"weight_{k}", "type": "number", "value": num(v)}
                        for k, v in sorted(dial.items(), key=lambda kv: float(kv[0]))
                    ],
                }],
            },
        ],
        "additional_information": {
            "ethical_considerations": (
                "Training data is procedurally generated in code, so no artist "
                "work was used without consent and no PII is present. In "
                "production use on a studio's own art, confirm the studio holds "
                "rights to the training images and that contributing artists "
                "have consented to style replication. Base model SDXL 1.0 is "
                "CreativeML Open RAIL++-M which permits commercial use; verify "
                "current terms before shipping assets. A style LoRA replicates "
                "an aesthetic, not an individual artist's identity, but the "
                "distinction narrows when a style belongs largely to one person."
            ),
            "caveats_and_recommendations": (
                "KNOWN LIMITATION - composition is not controlled. Mean subject "
                "coverage is 0.667 versus a ground-truth 0.146. This is a "
                "base-model prior: the prompted baseline fails identically "
                "(0.678). Inference-side mitigation was tested across 8 "
                "configurations (guidance 2.5-7.0, composition prompts, negative "
                "prompts) and reached only 0.641. Training-side mitigation was "
                "attempted in v2 by extending LoRA to convolution and projection "
                "layers and caused mode collapse; v2 is registered Rejected. "
                "RECOMMENDATION - keep LoRA on attention for style and control "
                "composition at inference with layout conditioning such as "
                "ControlNet or a fixed canvas template. This split is also "
                "architecturally preferable: style should stay constant across a "
                "project while composition varies per asset, so baking "
                "composition into weights would require a separate adapter per "
                "layout. Note that ControlNet was not tested in this work, so "
                "evidence covers what LoRA and prompting cannot do rather than "
                "what conditioning will achieve. "
                "OTHER - trained on only 24 images, so subject coverage is "
                "narrow. Do not exceed LoRA weight 1.0. Style metrics are "
                "specific to this machine-checkable house style and would need "
                "redefining for a real studio style bible."
            ),
            "custom_details": {
                "trigger_token": "flatgem style",
                "recommended_lora_weight": "0.9",
                "training_cost_usd": str(meta.get("cost", "")),
                "training_duration_min": str(meta.get("duration", "")),
                "mlflow_experiment": "sdxl-lora-style-lock",
                "model_registry_version": "1 (Approved)",
            },
        },
    }


def v2_content(results: dict, meta: dict) -> dict:
    vc = results.get("version_comparison", {}).get("summary", {})
    return {
        "model_overview": {
            "model_name": CARD_V2,
            "model_description": (
                "REJECTED EXPERIMENT. Attempt to fix the v1 composition "
                "limitation by extending LoRA from 4 attention modules to 12 "
                "modules including feed-forward, resnet convolution, shortcut "
                "and time-embedding projections, with the dataset doubled to 44 "
                "images across 10 subjects and rank lowered to 16. The attempt "
                "caused mode collapse and was rejected. Retained because the "
                "reasoning is institutional knowledge: anyone who later proposes "
                "adapting convolution layers to control composition can see it "
                "was tried, how it failed, and at what cost."
            ),
            "model_version": 2,
            "problem_type": "Image generation - style transfer",
            "algorithm_type": "LoRA fine-tuning of latent diffusion UNet",
            "model_creator": "Solutions Architecture demo",
            "model_artifact": [meta.get("artifact", "")],
        },
        "intended_uses": {
            "purpose_of_model": (
                "None. This version must not be deployed. It is retained as a "
                "documented negative result."
            ),
            "intended_uses": "Reference only.",
            "factors_affecting_model_efficiency": (
                "Mode collapse is not tunable. Output is near-identical "
                "regardless of prompt at every LoRA weight tested from 0.2 to "
                "1.0, and style is already degraded at 0.2."
            ),
            "risk_rating": "High",
            "explanations_for_risk_rating": (
                "Not a safety risk, but deploying it would silently degrade "
                "output quality: style score 0.641 against v1's 0.923, with "
                "prompt conditioning effectively destroyed. Registered as "
                "Rejected in Model Registry to prevent accidental deployment."
            ),
        },
        "training_details": {
            "objective_function": {
                "function": {
                    "function": "Other",
                    "facet": "MSE between predicted and sampled noise (epsilon prediction)",
                }
            },
            "training_observations": (
                "Loss tracked the v1 run for roughly three quarters of "
                "training (quarters 1-3 mean 0.008) then diverged sharply, "
                "rising from 0.13 to 0.44 over the final 150 steps. Loss did "
                "therefore signal trouble, but only near the end of a "
                "92-minute run and without indicating what had failed. "
                "48.7M trainable parameters "
                "at rank 16 across 12 module types, versus 46.4M at rank 32 "
                "across 4 in v1."
            ),
            "training_job_details": {
                "training_arn": meta.get("training_job_arn", ""),
                "training_datasets": [
                    "44 procedurally generated 512x512 icons, 10 subjects, "
                    "subject scale varied 0.85-1.1, holdout combinations excluded"
                ],
                "hyper_parameters": [
                    {"name": "rank", "value": "16"},
                    {"name": "target_modules", "value": "attn_ff_proj (12 module types)"},
                    {"name": "max_steps", "value": "900"},
                    {"name": "train_images", "value": "44"},
                ],
            },
        },
        "evaluation_details": [{
            "name": "Head-to-head versus v1 on identical holdout",
            "evaluation_observation": (
                "v2 is worse on both axes. Style score fell from 0.923 to 0.641 "
                "and subject-coverage error rose from 0.521 to 0.686. Root cause "
                "is mode collapse: across potion and sword the coverage "
                "difference is at most 0.002 at every LoRA weight from 0.2 to "
                "1.0, meaning output no longer responds to the prompt. Adapting "
                "convolution and projection layers at the same rank and alpha as "
                "attention, on only 44 images, overwhelmed the base model's "
                "structural priors. Three competing explanations were refuted by "
                "measurement: training data lacked scale variation (false, v1 "
                "coverage was already 0.1463 with std 0.047), preprocessing "
                "cropped subjects (false, +0.002 delta), and composition is "
                "fixable at inference (false, 8-config ablation reached 0.641 "
                "against a 0.146 target)."
            ),
            "metric_groups": [{
                "name": "Version comparison",
                "metric_data": [
                    {"name": "v2_style_score_mean", "type": "number",
                     "value": num(vc.get("v2_style_score_mean"))},
                    {"name": "v1_style_score_mean", "type": "number",
                     "value": num(vc.get("v1_style_score_mean"))},
                    {"name": "v2_coverage_error_mean", "type": "number",
                     "value": num(vc.get("v2_coverage_error_mean"))},
                    {"name": "v1_coverage_error_mean", "type": "number",
                     "value": num(vc.get("v1_coverage_error_mean"))},
                    {"name": "ground_truth_coverage_mean", "type": "number",
                     "value": num(vc.get("gt_coverage_mean"))},
                ],
            }],
        }],
        "additional_information": {
            "ethical_considerations": (
                "Same as v1: procedurally generated training data, no PII, no "
                "artist work used without consent."
            ),
            "caveats_and_recommendations": (
                "DO NOT DEPLOY. Recommended path forward is to keep LoRA on "
                "attention for style and solve composition with inference-time "
                "conditioning. If a structural adapter is still wanted, use a "
                "substantially lower rank and alpha for convolution layers than "
                "for attention, and far more training data than 44 images. "
                "A useful lesson for the wider pipeline: loss eventually "
                "diverged, but only in the final minutes of training, and a "
                "rising loss cannot distinguish mode collapse from ordinary "
                "instability. Identifying that prompt conditioning had been "
                "destroyed, that style regressed to 0.641 and that composition "
                "worsened, required evaluation against withheld ground truth. "
                "That is what caught it, for $2.32."
            ),
            "custom_details": {
                "outcome": "Rejected",
                "failure_mode": "mode collapse",
                "training_cost_usd": str(meta.get("cost", "")),
                "model_registry_version": "2 (Rejected)",
            },
        },
    }


def job_meta(sm, job_name: str) -> dict:
    d = sm.describe_training_job(TrainingJobName=job_name)
    mins = (d["TrainingEndTime"] - d["TrainingStartTime"]).total_seconds() / 60
    return {
        "artifact": d["ModelArtifacts"]["S3ModelArtifacts"],
        "training_job_arn": d["TrainingJobArn"],
        "duration": round(mins, 1),
        "cost": round(mins / 60 * config.INSTANCE_HOURLY_USD, 2),
    }


def upsert(sm, name: str, content: dict, status: str):
    body = json.dumps(content)
    try:
        sm.describe_model_card(ModelCardName=name)
        sm.update_model_card(ModelCardName=name, Content=body)
        print(f"updated card: {name}")
    except sm.exceptions.ResourceNotFound:
        sm.create_model_card(
            ModelCardName=name,
            Content=body,
            ModelCardStatus=status,
        )
        print(f"created card: {name}  (status {status})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1-job", required=True)
    p.add_argument("--v2-job", required=True)
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)
    results = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))

    upsert(sm, CARD_V1,
           v1_content(results, job_meta(sm, args.v1_job)), "Approved")
    upsert(sm, CARD_V2,
           v2_content(results, job_meta(sm, args.v2_job)), "Draft")


if __name__ == "__main__":
    main()
