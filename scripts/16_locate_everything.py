"""Print where every artifact this project creates lives in the console.

Useful because two things are easy to miss in the SageMaker UI:

  1. MLflow has two separate backends. "MLflow Apps" (newer, serverless) and
     "Tracking Servers" are different services holding different data. This
     project logs to a tracking server, so the Apps tab will look empty.

  2. The console's "Model evaluations" page is foundation-model evaluation:
     text benchmarks for LLMs. It cannot display custom image-style metrics.
     Those live in MLflow, in Model Registry ModelMetrics, and in Model Cards.

Also a quick health check that resources exist and nothing has drifted.
"""

import json
from pathlib import Path
import os
import sys

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()
GROUP = config.MODEL_PACKAGE_GROUP
TRACKING_SERVER = os.environ.get("MLFLOW_TRACKING_SERVER", "")
EXPERIMENT = config.EXPERIMENT

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

print("=" * 72)
print("1. MLFLOW  ->  console tab 'Tracking Servers', NOT 'MLflow Apps'")
print("=" * 72)

servers = sm.list_mlflow_tracking_servers().get("TrackingServerSummaries", [])
if not servers:
    print("  none in this region")
for s in servers:
    mark = "  <-- configured" if s["TrackingServerName"] == TRACKING_SERVER else ""
    print(f"  {s['TrackingServerName']:<26} {s['TrackingServerStatus']:<9} "
          f"{s['IsActive']}{mark}")

if TRACKING_SERVER:
    d = sm.describe_mlflow_tracking_server(TrackingServerName=TRACKING_SERVER)
    print(f"\n  UI url  : {d['TrackingServerUrl']}")
    print(f"  experiment to open: {EXPERIMENT}")
    print("\n  Get a presigned link with:")
    print("    python scripts/17_mlflow_link.py --open")
else:
    print("\n  MLFLOW_TRACKING_SERVER is not set; skipping MLflow details.")

print("\n" + "=" * 72)
print("2. EVALUATION METRICS  ->  three places, none of them 'Model evaluations'")
print("=" * 72)
print("""
  The console's 'Model evaluations' page is SageMaker Clarify / foundation
  model evaluation: text benchmarks for LLMs. Custom image-style metrics do
  not appear there. Ours are in:

    a) MLflow  : experiment above, nested runs under each training run
    b) Registry: ModelMetrics on each model package version (S3 JSON)
    c) Cards   : evaluation_details sections, human-readable
""")

print("  a) MLflow nested runs:")
print("     eval-ab, eval-drift, eval-novel, eval-dial, eval-holdout")
print("     plus eval-version-comparison under the v2 run")

print("\n  b) Registry ModelMetrics:")
try:
    pkgs = sm.list_model_packages(
        ModelPackageGroupName=GROUP)["ModelPackageSummaryList"]
    if not pkgs:
        print("     no versions registered yet")
    for p in sorted(pkgs, key=lambda x: x["ModelPackageVersion"]):
        det = sm.describe_model_package(ModelPackageName=p["ModelPackageArn"])
        uri = (det.get("ModelMetrics", {}).get("ModelQuality", {})
                  .get("Statistics", {}).get("S3Uri", "none"))
        print(f"     v{p['ModelPackageVersion']}  "
              f"{p['ModelApprovalStatus']:<22} {uri}")
except sm.exceptions.ClientError:
    print(f"     group '{GROUP}' not found; run scripts/10_register_model.py")

print("\n  c) Model Cards:")
cards = [c for c in sm.list_model_cards().get("ModelCardSummaries", [])
         if "lora" in c["ModelCardName"].lower()]
if not cards:
    print("     none yet; run scripts/14_create_model_cards.py")
for c in cards:
    card = sm.describe_model_card(ModelCardName=c["ModelCardName"])
    n = len(json.loads(card["Content"]).get("evaluation_details", []))
    print(f"     {c['ModelCardName']:<34} {c['ModelCardStatus']:<10} "
          f"{n} evaluation section(s)")

print("\n" + "=" * 72)
print("3. EVERYTHING ELSE")
print("=" * 72)

jobs = sm.list_training_jobs(NameContains="sdxl-lora", MaxResults=10,
                             SortBy="CreationTime", SortOrder="Descending")
print("  Training jobs:")
for j in jobs["TrainingJobSummaries"]:
    print(f"     {j['TrainingJobName']:<42} {j['TrainingJobStatus']}")

print("\n  Endpoints (these bill continuously until deleted):")
eps = [e for e in sm.list_endpoints(MaxResults=50)["Endpoints"]
       if "sdxl-lora" in e["EndpointName"]]
if not eps:
    print("     none running")
for e in eps:
    print(f"     {e['EndpointName']:<42} {e['EndpointStatus']}  "
          f"(~${config.INSTANCE_HOURLY_USD:.2f}/hr)")

print("\n  Console deep links:")
base = f"https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}"
print(f"     Tracking servers : {base}#/mlflow-tracking-servers")
print(f"     Model registry   : {base}#/model-registry/{GROUP}")
print(f"     Model cards      : {base}#/model-cards")
print(f"     Training jobs    : {base}#/jobs")
print(f"     Endpoints        : {base}#/endpoints")
