"""Extract the training loss curve from CloudWatch logs and chart it.

Why this is needed: the SageMaker console only renders a loss chart if the
training job was created with MetricDefinitions — regex patterns telling
SageMaker which log lines are metrics. Our launcher did not set them, so the
console has no curve to draw and FinalMetricDataList is null.

The loss itself was never lost. train_lora.py prints
"step 450/4800 loss=0.0038" every 50 steps, and those lines sit in
CloudWatch. This script parses them, logs the series to MLflow so it is
chartable there, and saves a PNG for the deck.

For future runs, scripts/03_launch_training.py now passes metric_definitions
so the console chart works natively.
"""

import argparse
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import boto3  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REGION = config.region()
TRACKING_ARN = config.tracking_arn()
EXPERIMENT = config.EXPERIMENT
OUT = ROOT / "outputs"

STEP_RE = re.compile(r"step (\d+)/(\d+) loss=([\d.]+)")


def fetch_loss(job_name: str):
    logs = boto3.client("logs", region_name=REGION)
    group = "/aws/sagemaker/TrainingJobs"
    streams = logs.describe_log_streams(
        logGroupName=group, logStreamNamePrefix=job_name, orderBy="LogStreamName"
    )["logStreams"]
    if not streams:
        raise SystemExit(f"no log streams for {job_name}")

    events, token = [], None
    while True:
        kw = {"logGroupName": group, "logStreamName": streams[0]["logStreamName"],
              "startFromHead": True, "limit": 10000}
        if token:
            kw["nextToken"] = token
        r = logs.get_log_events(**kw)
        events.extend(r["events"])
        if r["nextForwardToken"] == token or not r["events"]:
            break
        token = r["nextForwardToken"]

    points = []
    for e in events:
        m = STEP_RE.search(e["message"])
        if m:
            points.append((int(m.group(1)), float(m.group(3))))
    return sorted(points)


def plot(series: dict, path: Path):
    """Chart without matplotlib: draw directly with PIL."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1400, 620
    pad_l, pad_r, pad_t, pad_b = 90, 40, 80, 70
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b

    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def font(sz, bold=False):
        for n in (["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]):
            try:
                return ImageFont.truetype(n, sz)
            except Exception:
                continue
        return ImageFont.load_default()

    all_loss = [v for pts in series.values() for _, v in pts]
    all_step = [s for pts in series.values() for s, _ in pts]
    if not all_loss:
        raise SystemExit("no loss points to plot")

    y_max = max(all_loss) * 1.08
    x_max = max(all_step)

    d.text((pad_l, 22), "SDXL LoRA training loss", fill=(15, 15, 15),
           font=font(28, bold=True))
    d.text((pad_l, 54),
           "v1 stays flat at ~0.010 throughout. v2 tracks it for 3/4 of "
           "training then diverges sharply: 0.13 -> 0.44 over the last 150 "
           "steps.",
           fill=(95, 95, 95), font=font(15))

    # Axes and gridlines.
    d.rectangle([pad_l, pad_t, pad_l + pw, pad_t + ph], outline=(205, 205, 205))
    for i in range(5):
        y = pad_t + ph - int(ph * i / 4)
        val = y_max * i / 4
        d.line([(pad_l, y), (pad_l + pw, y)], fill=(238, 238, 238))
        d.text((10, y - 8), f"{val:.4f}", fill=(120, 120, 120), font=font(13))
    for i in range(6):
        x = pad_l + int(pw * i / 5)
        d.line([(x, pad_t), (x, pad_t + ph)], fill=(244, 244, 244))
        d.text((x - 18, pad_t + ph + 10), f"{int(x_max * i / 5)}",
               fill=(120, 120, 120), font=font(13))

    d.text((pad_l + pw // 2 - 50, H - 28), "training step",
           fill=(90, 90, 90), font=font(15))

    colours = [(30, 90, 190), (195, 60, 40)]
    for idx, (label, pts) in enumerate(series.items()):
        col = colours[idx % len(colours)]
        xy = [(pad_l + int(pw * s / x_max),
               pad_t + ph - int(ph * v / y_max)) for s, v in pts]
        if len(xy) > 1:
            d.line(xy, fill=col, width=2)
        ly = pad_t + 14 + idx * 24
        d.line([(pad_l + pw - 300, ly), (pad_l + pw - 260, ly)], fill=col, width=3)
        d.text((pad_l + pw - 250, ly - 9), label, fill=col, font=font(15))

    img.save(path)
    print(f"  chart -> {path.name} ({W}x{H})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", nargs="+", default=None,
                   help="training job names; defaults to the latest completed job")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args()

    jobs = args.jobs or [config.latest_training_job()]

    series = {}
    for job in jobs:
        pts = fetch_loss(job)
        label = "v2 (attn+conv)" if "v2" in job else "v1 (attention only)"
        series[label] = pts
        losses = [v for _, v in pts]
        print(f"{job}")
        print(f"  points {len(pts)}   first {losses[0]:.4f}   last {losses[-1]:.4f}   "
              f"min {min(losses):.4f}   max {max(losses):.4f}")

    OUT.mkdir(exist_ok=True)
    plot(series, OUT / "10_loss_curve.png")

    if args.no_mlflow:
        return

    import mlflow
    mlflow.set_tracking_uri(TRACKING_ARN)
    mlflow.set_experiment(EXPERIMENT)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)

    for job in jobs:
        run_name = f"train-{job}"
        matches = [r for r in client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=10,
        ) if "mlflow.parentRunId" not in r.data.tags]
        if not matches:
            print(f"  no MLflow run for {job}, skipped")
            continue
        rid = matches[0].info.run_id
        for step, loss in fetch_loss(job):
            client.log_metric(rid, "train_loss", loss, step=step)
        print(f"  logged train_loss series to {run_name[:40]}")

    print("\nChart it in MLflow as 'train_loss' on the Model metrics tab.")


if __name__ == "__main__":
    main()
