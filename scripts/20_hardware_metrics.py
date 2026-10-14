"""Pull real GPU/CPU telemetry from CloudWatch and log it to MLflow.

Why MLflow's own "System metrics" tab is blank, and why enabling it would
have been misleading:

  1. It is opt-in. You need mlflow.enable_system_metrics_logging() or
     MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true, plus psutil, plus pynvml for
     GPU. None of that was set.

  2. More importantly, it samples the machine running the MLflow CLIENT.
     Every run in this experiment was logged from a Windows laptop that read
     results.json and pushed to the tracking server. The training ran
     separately on an ml.g5.2xlarge. So enabling it would have populated the
     tab with laptop CPU and RAM, which is worse than blank: it looks like
     training telemetry but describes the wrong computer entirely.

The authoritative hardware data is published by SageMaker to CloudWatch
under /aws/sagemaker/TrainingJobs. This script reads it and logs it as
stepped MLflow metrics on the correct parent run, so the numbers live
alongside the style scores and are chartable.

Findings worth knowing: GPU averaged 78.8% on v1 and 86.9% on v2 with peaks
at 100%, so the job was genuinely GPU-bound rather than waiting on data.
GPU memory peaked at only 46%, which means the instance was oversized for
the batch size and there is headroom to raise batch size or resolution
without moving to a larger instance.
"""

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import boto3  # noqa: E402
import mlflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()
EXPERIMENT = config.EXPERIMENT

METRICS = [
    "GPUUtilization",
    "GPUMemoryUtilization",
    "CPUUtilization",
    "MemoryUtilization",
    "DiskUtilization",
]


def fetch(cw, job, start, end, metric, period=60):
    resp = cw.get_metric_statistics(
        Namespace="/aws/sagemaker/TrainingJobs",
        MetricName=metric,
        Dimensions=[{"Name": "Host", "Value": f"{job}/algo-1"}],
        StartTime=start - timedelta(minutes=2),
        EndTime=end + timedelta(minutes=2),
        Period=period,
        Statistics=["Average", "Maximum"],
    )
    return sorted(resp["Datapoints"], key=lambda p: p["Timestamp"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="print only, do not write to MLflow")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)
    cw = boto3.client("cloudwatch", region_name=REGION)

    d = sm.describe_training_job(TrainingJobName=args.job_name)
    start, end = d["TrainingStartTime"], d["TrainingEndTime"]
    instance = d["ResourceConfig"]["InstanceType"]

    print(f"job      : {args.job_name}")
    print(f"instance : {instance}")
    print(f"window   : {start:%H:%M} -> {end:%H:%M}\n")

    series = {}
    for m in METRICS:
        pts = fetch(cw, args.job_name, start, end, m)
        if not pts:
            print(f"  {m:<22} no datapoints")
            continue
        avgs = [pt["Average"] for pt in pts]
        series[m] = pts
        print(f"  {m:<22} avg {sum(avgs)/len(avgs):>6.1f}%  "
              f"max {max(pt['Maximum'] for pt in pts):>6.1f}%  "
              f"({len(pts)} samples)")

    if not series:
        raise SystemExit("\nNo CloudWatch data. Metrics expire after 15 months.")

    gpu = series.get("GPUUtilization")
    gmem = series.get("GPUMemoryUtilization")
    if gpu and gmem:
        gpu_avg = sum(p["Average"] for p in gpu) / len(gpu)
        mem_max = max(p["Maximum"] for p in gmem)
        print("\ninterpretation:")
        if gpu_avg > 70:
            print(f"  GPU-bound at {gpu_avg:.0f}% average: compute is the "
                  "bottleneck, not data loading.")
        else:
            print(f"  GPU only {gpu_avg:.0f}% average: likely starved by data "
                  "loading or CPU preprocessing.")
        if mem_max < 60:
            print(f"  GPU memory peaked at {mem_max:.0f}%: room to raise batch "
                  "size or resolution on this same instance.")

    if args.dry_run:
        print("\n(dry run, nothing written)")
        return

    mlflow.set_tracking_uri(config.tracking_arn())
    mlflow.set_experiment(EXPERIMENT)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)

    run_name = f"train-{args.job_name}"
    matches = [r for r in client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=10,
    ) if "mlflow.parentRunId" not in r.data.tags]

    if not matches:
        raise SystemExit(f"no parent run named {run_name}; run 07 or 13 first")

    run_id = matches[0].info.run_id

    # Step index in minutes from training start, so charts read as a timeline.
    for metric, pts in series.items():
        for pt in pts:
            minute = int((pt["Timestamp"] - start).total_seconds() // 60)
            client.log_metric(run_id, f"hw_{metric}_avg", pt["Average"], step=minute)
            client.log_metric(run_id, f"hw_{metric}_max", pt["Maximum"], step=minute)

    gpu_avg = sum(p["Average"] for p in series["GPUUtilization"]) / len(series["GPUUtilization"])
    client.log_metric(run_id, "hw_gpu_util_mean", round(gpu_avg, 2))
    client.log_metric(run_id, "hw_gpu_mem_peak",
                      round(max(p["Maximum"] for p in series["GPUMemoryUtilization"]), 2))
    client.set_tag(run_id, "hw_instance_type", instance)
    client.set_tag(
        run_id, "hw_metrics_source",
        "CloudWatch /aws/sagemaker/TrainingJobs. MLflow's built-in System "
        "metrics tab samples the client machine, which for these runs was a "
        "laptop, not the training instance.",
    )

    print(f"\nlogged {sum(len(v) for v in series.values())} datapoints "
          f"to run {run_id[:12]}")
    print("Chart them as hw_GPUUtilization_avg on the Model metrics tab.")


if __name__ == "__main__":
    main()
