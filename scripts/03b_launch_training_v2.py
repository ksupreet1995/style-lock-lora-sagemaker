"""Launch the v2 SDXL LoRA training job.

Differences from v1, each tied to a measured finding:

  dataset        44 images / 10 subjects (was 24 / 6), with mild scale
                 variation. Not because v1 coverage was wrong (it was 0.1463,
                 matching holdout) but to give the relationship more signal.
  targets        attn_ff_proj instead of attention-only. Base SDXL and the v1
                 LoRA both produced ~0.67 coverage against a 0.146 target, so
                 the large-object prior lives outside attention. This adds
                 feed-forward, resnet conv and projection layers.
  rank           16 instead of 32. More layers are adapted, so a lower rank
                 keeps the parameter count and overfitting risk in check on a
                 small dataset.
  steps          900 instead of 1200, since the dataset nearly doubled and
                 more layers train per step.
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
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--target-modules", default="attn_ff_proj",
                   choices=["attn", "attn_ff", "attn_ff_proj"])
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()

    region = config.region()
    sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    bucket = config.bucket()
    prefix = config.prefix()

    train_dir = ROOT / "data" / "train_v2"
    n = len(list(train_dir.glob("*.png")))
    if n == 0:
        raise SystemExit("No v2 images. Run scripts/01b_generate_dataset_v2.py first.")

    print(f"uploading {n} images ...")
    s3_train = sess.upload_data(path=str(train_dir), bucket=bucket,
                                key_prefix=f"{prefix}/data/train_v2")
    print(f"  -> {s3_train}")

    job_name = f"sdxl-lora-v2-{time.strftime('%Y%m%d-%H%M%S')}"

    est = Estimator(
        image_uri=config.training_image(),
        role=config.role_arn(),
        instance_count=1,
        instance_type=config.INSTANCE_TYPE,
        sagemaker_session=sess,
        entry_point="train_lora.py",
        source_dir=str(ROOT / "src"),
        output_path=f"s3://{bucket}/{prefix}/output",
        base_job_name="sdxl-lora-v2",
        max_run=3 * 60 * 60,
        volume_size=120,
        # Required for the console's loss chart; see 03_launch_training.py.
        metric_definitions=[
            {"Name": "train:loss", "Regex": r"loss=([0-9.]+)"},
            {"Name": "train:step", "Regex": r"step (\d+)/"},
        ],
        environment={
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
            "target-modules": args.target_modules,
            "batch-size": 1,
            "grad-accum": 4,
        },
    )

    print(f"\nlaunching {job_name}")
    print(f"  targets: {args.target_modules}   rank: {args.rank}   steps: {args.max_steps}")

    est.fit({"train": TrainingInput(s3_train, input_mode="File")},
            job_name=job_name, wait=args.wait)

    print(f"\nsubmitted: {job_name}")
    print(f"poll: python scripts/04_job_status.py {job_name}")


if __name__ == "__main__":
    main()
