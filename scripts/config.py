"""Shared configuration, resolved from the environment.

Nothing account-specific is committed. Every value is either auto-detected
from your AWS session or read from an environment variable.

Required before running anything that touches AWS:

    export AWS_REGION=us-east-2
    export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<your-execution-role>

Optional:

    export MLFLOW_TRACKING_SERVER=<name>     # for the MLflow scripts
    export S3_BUCKET=<bucket>                # defaults to the SageMaker bucket
    export S3_PREFIX=sdxl-lora-demo

Copy .env.example to .env and source it, or set these in your shell.
"""

import os
from functools import lru_cache

DEFAULT_PREFIX = "sdxl-lora-demo"
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "sdxl-lora-style-lock")
MODEL_PACKAGE_GROUP = os.environ.get("MODEL_PACKAGE_GROUP", "sdxl-lora-flatgem-style")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g5.2xlarge")

# On-demand hourly price used only for cost estimates in logs and reports.
# Override if your region or commitment differs.
INSTANCE_HOURLY_USD = float(os.environ.get("INSTANCE_HOURLY_USD", "1.515"))


def region() -> str:
    r = (os.environ.get("AWS_REGION")
         or os.environ.get("AWS_DEFAULT_REGION"))
    if r:
        return r
    import boto3
    r = boto3.Session().region_name
    if not r:
        raise SystemExit(
            "No region configured. Set AWS_REGION or configure the AWS CLI."
        )
    return r


@lru_cache(maxsize=1)
def account_id() -> str:
    """Resolved from the caller's own credentials, never hardcoded."""
    import boto3
    return boto3.client("sts").get_caller_identity()["Account"]


def role_arn() -> str:
    arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    if arn:
        return arn
    # Fall back to the SDK's own resolution, which works inside Studio.
    try:
        import sagemaker
        return sagemaker.get_execution_role()
    except Exception:
        raise SystemExit(
            "No execution role found. Set SAGEMAKER_ROLE_ARN to an IAM role "
            "that SageMaker can assume, e.g.\n"
            "  export SAGEMAKER_ROLE_ARN="
            "arn:aws:iam::<account>:role/service-role/<role-name>"
        )


def bucket() -> str:
    b = os.environ.get("S3_BUCKET")
    if b:
        return b
    import boto3
    import sagemaker
    return sagemaker.Session(
        boto_session=boto3.Session(region_name=region())
    ).default_bucket()


def prefix() -> str:
    return os.environ.get("S3_PREFIX", DEFAULT_PREFIX)


def tracking_arn() -> str:
    """ARN of a SageMaker managed MLflow tracking server."""
    explicit = os.environ.get("MLFLOW_TRACKING_ARN")
    if explicit:
        return explicit
    name = os.environ.get("MLFLOW_TRACKING_SERVER")
    if not name:
        raise SystemExit(
            "No MLflow tracking server configured. Either:\n"
            "  export MLFLOW_TRACKING_SERVER=<server-name>\n"
            "  export MLFLOW_TRACKING_ARN=<full-arn>\n"
            "List available servers with:\n"
            "  aws sagemaker list-mlflow-tracking-servers"
        )
    return (f"arn:aws:sagemaker:{region()}:{account_id()}"
            f":mlflow-tracking-server/{name}")


def training_image() -> str:
    """Resolve the PyTorch DLC URI for this region rather than hardcoding it."""
    override = os.environ.get("TRAINING_IMAGE_URI")
    if override:
        return override
    from sagemaker import image_uris
    return image_uris.retrieve(
        framework="pytorch", region=region(), version="2.5.1",
        py_version="py311", instance_type=INSTANCE_TYPE,
        image_scope="training",
    )


def inference_image() -> str:
    override = os.environ.get("INFERENCE_IMAGE_URI")
    if override:
        return override
    from sagemaker import image_uris
    return image_uris.retrieve(
        framework="pytorch", region=region(), version="2.5.1",
        py_version="py311", instance_type=INSTANCE_TYPE,
        image_scope="inference",
    )


def latest_training_job(name_contains: str = "sdxl-lora") -> str:
    """Most recent completed training job, so job names need not be hardcoded."""
    import boto3
    sm = boto3.client("sagemaker", region_name=region())
    jobs = sm.list_training_jobs(
        NameContains=name_contains, StatusEquals="Completed",
        SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
    )["TrainingJobSummaries"]
    if not jobs:
        raise SystemExit(f"No completed training job matching '{name_contains}'")
    return jobs[0]["TrainingJobName"]
