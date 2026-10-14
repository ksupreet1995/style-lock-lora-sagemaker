"""Deploy the trained LoRA to a real-time SageMaker endpoint.

Implementation note: we repack model.tar.gz ourselves with the inference code
already inside under code/, rather than letting the SDK do it. The SDK's
automatic repack extracts to a local temp directory and can fail on cleanup
on Windows, particularly under synced folders. Packing it ourselves is also
more transparent about what ends up in the container.

COST WARNING: a real-time endpoint bills continuously until deleted.
A single-GPU instance is roughly $1.50/hr, about $36/day if left running.
Run scripts/99_cleanup.py when you are finished.
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

import boto3  # noqa: E402
import sagemaker  # noqa: E402
from sagemaker.pytorch import PyTorchModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Inference needs diffusers/peft, which the base PyTorch DLC does not ship.
INFERENCE_REQS = """diffusers==0.31.0
transformers==4.46.3
peft==0.13.2
safetensors==0.4.5
accelerate==1.1.1
"""


def repack(artifact_uri: str, sess, bucket: str) -> str:
    """Download the trained artifact, inject code/, re-upload."""
    s3 = boto3.client("s3", region_name=config.region())
    _, _, rest = artifact_uri.partition("s3://")
    src_bucket, _, src_key = rest.partition("/")

    # Use the system temp dir, not the repo, to avoid file-sync interference.
    work = Path(tempfile.mkdtemp(prefix="smrepack-"))
    try:
        tar_in = work / "model.tar.gz"
        print(f"downloading {artifact_uri}")
        s3.download_file(src_bucket, src_key, str(tar_in))

        extract = work / "extract"
        extract.mkdir()
        with tarfile.open(tar_in, "r:gz") as t:
            t.extractall(extract)
        print(f"  contents: {[p.name for p in extract.iterdir()]}")

        code = extract / "code"
        code.mkdir(exist_ok=True)
        shutil.copy(ROOT / "src" / "inference.py", code / "inference.py")
        (code / "requirements.txt").write_text(INFERENCE_REQS, encoding="utf-8")

        tar_out = work / "model-with-code.tar.gz"
        with tarfile.open(tar_out, "w:gz") as t:
            for item in extract.iterdir():
                t.add(item, arcname=item.name)
        size_mb = tar_out.stat().st_size / 1e6
        print(f"  repacked: {size_mb:.1f} MB")

        key = f"{config.prefix()}/deploy/model-{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
        s3.upload_file(str(tar_out), bucket, key)
        uri = f"s3://{bucket}/{key}"
        print(f"  uploaded: {uri}")
        return uri
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--job-name", default=None,
                   help="defaults to the most recent completed sdxl-lora job")
    p.add_argument("--endpoint-name", default=None)
    args = p.parse_args()

    region = config.region()
    boto_sess = boto3.Session(region_name=region)
    sess = sagemaker.Session(boto_session=boto_sess)
    sm = boto_sess.client("sagemaker")
    bucket = config.bucket()

    job_name = args.job_name or config.latest_training_job()
    print(f"training job: {job_name}")

    d = sm.describe_training_job(TrainingJobName=job_name)
    if d["TrainingJobStatus"] != "Completed":
        raise SystemExit(f"Job status is {d['TrainingJobStatus']}, expected Completed.")

    artifact = d["ModelArtifacts"]["S3ModelArtifacts"]
    model_data = repack(artifact, sess, bucket)

    endpoint = args.endpoint_name or f"sdxl-lora-{time.strftime('%m%d-%H%M')}"

    model = PyTorchModel(
        model_data=model_data,
        role=config.role_arn(),
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",  # already inside code/
        sagemaker_session=sess,
        env={
            "HF_HOME": "/tmp/hf",
            "SAGEMAKER_MODEL_SERVER_TIMEOUT": "900",
            "TS_DEFAULT_RESPONSE_TIMEOUT": "900",
            "SAGEMAKER_PROGRAM": "inference.py",
        },
    )

    print(f"\nDeploying to {endpoint} on {config.INSTANCE_TYPE} ...")
    print("Expect 10-15 min: container start, pip installs, SDXL base download.")

    model.deploy(
        initial_instance_count=1,
        instance_type=config.INSTANCE_TYPE,
        endpoint_name=endpoint,
        wait=True,
    )

    print(f"\nEndpoint live: {endpoint}")
    print(f"Next:\n  python scripts/06_compare.py --endpoint {endpoint}")
    print(f"\nREMEMBER: python scripts/99_cleanup.py --endpoint {endpoint}")


if __name__ == "__main__":
    main()
