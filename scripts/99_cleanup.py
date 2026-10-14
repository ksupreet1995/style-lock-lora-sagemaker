"""Tear down billable resources.

Endpoints bill continuously whether used or not. Run this when the demo ends.
"""

import sys
from pathlib import Path
import argparse

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default=None, help="specific endpoint to delete")
    p.add_argument("--all-lora", action="store_true",
                   help="delete every endpoint whose name contains sdxl-lora")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)

    targets = []
    if args.endpoint:
        targets.append(args.endpoint)
    if args.all_lora:
        for e in sm.list_endpoints(MaxResults=100)["Endpoints"]:
            if "sdxl-lora" in e["EndpointName"]:
                targets.append(e["EndpointName"])

    if not targets:
        eps = sm.list_endpoints(MaxResults=100)["Endpoints"]
        if not eps:
            print("No endpoints in this region.")
        else:
            print("Live endpoints (none deleted; pass --endpoint or --all-lora):")
            for e in eps:
                print(f"  {e['EndpointName']}  {e['EndpointStatus']}")
        return

    for name in dict.fromkeys(targets):
        try:
            sm.delete_endpoint(EndpointName=name)
            print(f"deleted endpoint: {name}")
        except Exception as e:
            print(f"endpoint {name}: {type(e).__name__}: {str(e)[:100]}")
        try:
            sm.delete_endpoint_config(EndpointConfigName=name)
            print(f"deleted endpoint config: {name}")
        except Exception:
            pass

    print("\nNote: S3 artifacts and models are cheap and left in place.")


if __name__ == "__main__":
    main()
