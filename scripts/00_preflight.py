"""Preflight checks for the SageMaker SDXL LoRA demo.

Verifies the two things that actually block a fresh account:
  1. GPU instance quota for SageMaker training jobs and endpoints.
  2. An IAM role SageMaker can assume (an IAM *user* is not enough).

Read-only. Creates nothing, costs nothing.
"""

import sys
from pathlib import Path
import json
import time

import boto3
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()

# Service Quotas has a low request rate; retry aggressively.
RETRY_CONFIG = Config(retries={"max_attempts": 10, "mode": "adaptive"})

# Instance types this demo can realistically use, cheapest-capable first.
CANDIDATES = [
    "ml.g5.2xlarge",
    "ml.g5.xlarge",
    "ml.g6.2xlarge",
    "ml.g6.xlarge",
    "ml.g5.4xlarge",
    "ml.g4dn.2xlarge",
    "ml.g4dn.xlarge",
]


def check_quotas(region: str) -> dict:
    sq = boto3.client("service-quotas", region_name=region, config=RETRY_CONFIG)

    quotas = {}
    next_token = None
    while True:
        kwargs = {"ServiceCode": "sagemaker", "MaxResults": 100}
        if next_token:
            kwargs["NextToken"] = next_token
        resp = sq.list_service_quotas(**kwargs)
        for item in resp["Quotas"]:
            quotas[item["QuotaName"]] = item["Value"]
        next_token = resp.get("NextToken")
        if not next_token:
            break
        time.sleep(1.5)  # stay under the rate quota

    print(f"Fetched {len(quotas)} SageMaker quotas in {region}\n")

    results = {"training": {}, "endpoint": {}}
    for inst in CANDIDATES:
        train_key = f"{inst} for training job usage"
        endpoint_key = f"{inst} for endpoint usage"
        results["training"][inst] = quotas.get(train_key)
        results["endpoint"][inst] = quotas.get(endpoint_key)

    print(f"{'instance':<18} {'training':>10} {'endpoint':>10}")
    print("-" * 40)
    for inst in CANDIDATES:
        t = results["training"][inst]
        e = results["endpoint"][inst]
        t_s = "n/a" if t is None else f"{t:.0f}"
        e_s = "n/a" if e is None else f"{e:.0f}"
        print(f"{inst:<18} {t_s:>10} {e_s:>10}")

    return results


def check_roles() -> list:
    iam = boto3.client("iam")
    paginator = iam.get_paginator("list_roles")

    sm_roles = []
    for page in paginator.paginate():
        for role in page["Roles"]:
            doc = role.get("AssumeRolePolicyDocument", {})
            if isinstance(doc, str):
                doc = json.loads(doc)
            for stmt in doc.get("Statement", []):
                principal = stmt.get("Principal", {})
                service = principal.get("Service", [])
                if isinstance(service, str):
                    service = [service]
                if any("sagemaker" in s for s in service):
                    sm_roles.append(role["RoleName"])
                    break

    if sm_roles:
        print("\nRoles assumable by SageMaker:")
        for r in sm_roles:
            print(f"  - {r}")
    else:
        print("\nNo SageMaker-assumable IAM role found.")
        print("A training job needs one. See scripts/01_create_role.py")

    return sm_roles


if __name__ == "__main__":
    print(f"=== Preflight: region {REGION} ===\n")
    ident = boto3.client("sts").get_caller_identity()
    print(f"Account : {ident['Account']}")
    print(f"Identity: {ident['Arn']}\n")

    quotas = check_quotas(REGION)
    roles = check_roles()

    usable_train = [i for i, v in quotas["training"].items() if v and v > 0]
    usable_endpoint = [i for i, v in quotas["endpoint"].items() if v and v > 0]

    print("\n=== Summary ===")
    print(f"Training-capable GPU types : {usable_train or 'NONE - quota increase required'}")
    print(f"Endpoint-capable GPU types : {usable_endpoint or 'NONE - quota increase required'}")
    print(f"SageMaker roles            : {len(roles)}")
