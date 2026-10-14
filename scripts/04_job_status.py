"""Poll a SageMaker training job and tail its CloudWatch logs."""

import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()


def status(job_name: str, tail: int = 40):
    sm = boto3.client("sagemaker", region_name=REGION)
    d = sm.describe_training_job(TrainingJobName=job_name)

    print(f"job      : {job_name}")
    print(f"status   : {d['TrainingJobStatus']}")
    print(f"secondary: {d.get('SecondaryStatus')}")
    if d.get("FailureReason"):
        print(f"failure  : {d['FailureReason']}")
    if d.get("TrainingStartTime"):
        print(f"started  : {d['TrainingStartTime']}")
    if d.get("TrainingEndTime"):
        secs = (d["TrainingEndTime"] - d["TrainingStartTime"]).total_seconds()
        print(f"duration : {secs / 60:.1f} min")
    if d["TrainingJobStatus"] == "Completed":
        print(f"artifact : {d['ModelArtifacts']['S3ModelArtifacts']}")

    logs = boto3.client("logs", region_name=REGION)
    group = "/aws/sagemaker/TrainingJobs"
    try:
        streams = logs.describe_log_streams(
            logGroupName=group, logStreamNamePrefix=job_name, orderBy="LogStreamName"
        )["logStreams"]
        if not streams:
            print("\n(no log streams yet)")
            return
        ev = logs.get_log_events(
            logGroupName=group, logStreamName=streams[0]["logStreamName"], limit=tail
        )["events"]
        print(f"\n--- last {len(ev)} log lines ---")
        for e in ev:
            print(e["message"])
    except logs.exceptions.ResourceNotFoundException:
        print("\n(log group not created yet)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sm = boto3.client("sagemaker", region_name=REGION)
        jobs = sm.list_training_jobs(
            NameContains="sdxl-lora", SortBy="CreationTime", SortOrder="Descending", MaxResults=5
        )["TrainingJobSummaries"]
        if not jobs:
            print("No sdxl-lora training jobs found.")
            sys.exit(0)
        print("Recent jobs:")
        for j in jobs:
            print(f"  {j['TrainingJobName']}  {j['TrainingJobStatus']}")
        print("\nPass a job name to see detail.")
        sys.exit(0)
    status(sys.argv[1])
