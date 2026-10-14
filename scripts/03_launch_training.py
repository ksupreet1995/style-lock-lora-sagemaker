"""Upload the dataset and launch the SDXL LoRA training job on SageMaker.

Cost note: a single-GPU instance such as ml.g5.2xlarge is roughly $1.50/hr
on demand. Measured here: 1200 optimiser steps with gradient accumulation 4
at 1024px took ~102 minutes, about $2.60 per style experiment. For a faster
rehearsal run try 768px and 600 steps, roughly 25 minutes.

Configuration comes from the environment; see scripts/config.py and
.env.example.

Pass --wait to stream logs, or omit it to return immediately and poll.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

import boto3  # noqa: E402
import sagemaker  # noqa: E402
from sagemaker.estimator import Estimator  # noqa: E402
from sagemaker.inputs import TrainingInput  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--wait", action="store_true", help="stream logs until done")
    args = p.parse_args()

    region = config.region()
    boto_sess = boto3.Session(region_name=region)
    sess = sagemaker.Session(boto_session=boto_sess)
    bucket = config.bucket()
    prefix = config.prefix()
    print(f"region : {region}")
    print(f"bucket : {bucket}")

    train_dir = ROOT / "data" / "train"
    n_images = len(list(train_dir.glob("*.png")))
    if n_images == 0:
        raise SystemExit("No training images. Run scripts/01_generate_dataset.py first.")

    print(f"\nUploading {n_images} images from {train_dir} ...")
    s3_train = sess.upload_data(
        path=str(train_dir),
        bucket=bucket,
        key_prefix=f"{prefix}/data/train",
    )
    print(f"  -> {s3_train}")

    job_name = f"sdxl-lora-{time.strftime('%Y%m%d-%H%M%S')}"

    est = Estimator(
        image_uri=config.training_image(),
        role=config.role_arn(),
        instance_count=1,
        instance_type=config.INSTANCE_TYPE,
        sagemaker_session=sess,
        entry_point="train_lora.py",
        source_dir=str(ROOT / "src"),
        output_path=f"s3://{bucket}/{prefix}/output",
        base_job_name="sdxl-lora",
        max_run=3 * 60 * 60,
        volume_size=120,
        # Without this the console shows no loss chart and
        # FinalMetricDataList is null: SageMaker needs regex telling it which
        # log lines are metrics. train_lora.py prints
        # "step 450/4800 loss=0.0038".
        metric_definitions=[
            {"Name": "train:loss", "Regex": r"loss=([0-9.]+)"},
            {"Name": "train:step", "Regex": r"step (\d+)/"},
        ],
        environment={
            # Cache the HF download on the instance's local NVMe.
            "HF_HOME": "/tmp/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        hyperparameters={
            "max-steps": args.max_steps,
            "rank": args.rank,
            "lora-alpha": args.rank,
            "resolution": args.resolution,
            "learning-rate": args.learning_rate,
            "batch-size": 1,
            "grad-accum": 4,
        },
    )

    print(f"\nLaunching job: {job_name}")
    print(f"  instance : {config.INSTANCE_TYPE}")
    print(f"  steps    : {args.max_steps}   rank: {args.rank}   res: {args.resolution}")

    est.fit(
        {"train": TrainingInput(s3_train, input_mode="File")},
        job_name=job_name,
        wait=args.wait,
    )

    print(f"\nJob submitted: {job_name}")
    print(f"Artifacts will land at: "
          f"s3://{bucket}/{prefix}/output/{job_name}/output/model.tar.gz")
    print(f"\nPoll with:\n  python scripts/04_job_status.py {job_name}")


if __name__ == "__main__":
    main()
