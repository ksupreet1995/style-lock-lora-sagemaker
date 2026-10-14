"""Trace the image generation pipeline into MLflow.

TRACES vs LOGS vs METRICS, since the three are easy to conflate:

  LOGS     unstructured text lines, emitted wherever you put a print.
           "step 450/4800 loss=0.0038". Searchable by string, no structure,
           no relationships between lines. Ours are in CloudWatch.

  METRICS  named numbers attached to a run. style_score=0.984. Aggregatable
           and chartable, but they say nothing about HOW the number was
           produced or how long each stage took.

  TRACES   a structured tree of timed, nested spans for ONE request, with
           inputs and outputs captured at each level. Tells you what called
           what, in what order, with what arguments, and where the time went.

The distinction that matters here: a metric tells you the style score was
0.984. A trace tells you that producing it took 11.2s, of which 9.8s was the
endpoint call and 1.1s was scoring, that the prompt was exactly
"flatgem style, a lantern", and that lora_scale was 0.9. When an artist says
"generation got slow this afternoon", metrics cannot answer it and logs make
you reconstruct the sequence by hand. A trace shows it directly.

MLflow tracing exists primarily for GenAI apps, where a single request fans
out across retrieval, prompt assembly and several model calls. An image
pipeline fits that shape well: prompt build -> endpoint invoke -> decode ->
score.

Usage:
  python scripts/18_trace_generation.py --endpoint sdxl-lora-flatgem-demo
"""

import argparse
import base64
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import boto3  # noqa: E402
import mlflow  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "style"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

TRACKING_ARN = config.tracking_arn()
EXPERIMENT = config.EXPERIMENT
REGION = config.region()

from house_style import TRIGGER, describe_style  # noqa: E402


def _metrics():
    spec = importlib.util.spec_from_file_location(
        "style_metrics", ROOT / "scripts" / "02_style_metrics.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SM = _metrics()


@mlflow.trace(name="build_prompt", span_type="CHAIN")
def build_prompt(subject: str, use_lora: bool) -> dict:
    """Prompt assembly. Captured so you can see exactly what was sent."""
    if use_lora:
        prompt = f"{TRIGGER}, {subject}"
        strategy = "trigger_token"
    else:
        prompt = f"{subject}, {describe_style()}"
        strategy = "prompt_engineering"
    return {
        "prompt": prompt,
        "strategy": strategy,
        "negative_prompt": "photograph, 3d render, gradient, blurry",
        "prompt_chars": len(prompt),
    }


@mlflow.trace(name="invoke_endpoint", span_type="LLM")
def invoke_endpoint(endpoint: str, payload: dict) -> str:
    """The expensive span. Usually 85-90% of total latency."""
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = rt.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    body = json.loads(resp["Body"].read())
    mlflow.update_current_trace(
        tags={"endpoint": endpoint, "model_meta": json.dumps(body["meta"])}
    )
    return body["image_b64"]


@mlflow.trace(name="decode_image", span_type="PARSER")
def decode_image(image_b64: str) -> dict:
    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    return {"width": img.width, "height": img.height, "bytes": len(image_b64)}


@mlflow.trace(name="score_style", span_type="TOOL")
def score_style(image_b64: str) -> dict:
    """Scoring broken out so you can see metric cost separately."""
    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    palette = SM.palette_score(img)
    flatness = SM.flatness_score(img)
    shadow = SM.shadow_present(img)
    return {
        "palette": round(palette, 4),
        "flatness": round(flatness, 4),
        "shadow": round(shadow, 4),
        "style_score": round((palette + flatness + shadow) / 3, 4),
    }


@mlflow.trace(name="generate_styled_asset", span_type="CHAIN")
def generate_styled_asset(endpoint: str, subject: str, use_lora: bool = True,
                          lora_scale: float = 0.9, seed: int = 42) -> dict:
    """Root span. One call per generated asset, with everything nested under."""
    built = build_prompt(subject, use_lora)

    payload = {
        "prompt": built["prompt"],
        "negative_prompt": built["negative_prompt"],
        "use_lora": use_lora,
        "lora_scale": lora_scale,
        "seed": seed,
        "steps": 30,
        "guidance_scale": 7.0,
    }

    image_b64 = invoke_endpoint(endpoint, payload)
    meta = decode_image(image_b64)
    scores = score_style(image_b64)

    mlflow.update_current_trace(
        tags={
            "subject": subject,
            "use_lora": str(use_lora),
            "lora_scale": str(lora_scale),
            "style_score": str(scores["style_score"]),
            "strategy": built["strategy"],
        }
    )

    return {"image_meta": meta, "scores": scores, "prompt": built["prompt"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--subject", default="a lantern")
    p.add_argument("--compare", action="store_true",
                   help="trace both the LoRA and the prompted baseline")
    args = p.parse_args()

    mlflow.set_tracking_uri(TRACKING_ARN)
    mlflow.set_experiment(EXPERIMENT)

    runs = [(True, 0.9)]
    if args.compare:
        runs.append((False, 0.0))

    with mlflow.start_run(run_name="traced-generation"):
        mlflow.set_tag("purpose", "demonstrate MLflow tracing on the image pipeline")
        for use_lora, scale in runs:
            label = "LoRA" if use_lora else "baseline"
            print(f"\ntracing {label} generation of '{args.subject}' ...")
            out = generate_styled_asset(
                args.endpoint, args.subject, use_lora=use_lora, lora_scale=scale
            )
            print(f"  prompt      : {out['prompt'][:70]}")
            print(f"  style_score : {out['scores']['style_score']}")
            print(f"  image       : {out['image_meta']['width']}x"
                  f"{out['image_meta']['height']}")

    # Async export races interpreter shutdown without this.
    mlflow.flush_trace_async_logging()
    print("\nflushed trace queue")

    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    traces = mlflow.search_traces(
        experiment_ids=[exp.experiment_id], max_results=10, return_type="list"
    )
    print(f"traces now on server: {len(traces)}")
    for t in traces[:4]:
        spans = [s.name for s in t.data.spans]
        ms = getattr(t.info, "execution_duration", None)
        print(f"  {t.info.request_id[:12]}  {t.info.state}  "
              f"{ms if ms else '?'}ms  spans={len(spans)}")

    print(f"\nOpen the Traces tab on the 'traced-generation' run.")


if __name__ == "__main__":
    main()
