"""Generate a presigned MLflow link that bypasses console authentication.

This is the definitive test for the account-visibility problem. A presigned
URL carries its own auth token, so it does not depend on whichever account
the browser console happens to be signed into. If this opens and shows the
experiment, the data was always there and the console was simply pointed
somewhere else.

Two different timeouts matter here and they are easy to confuse:

  ExpiresInSeconds                  how long the LINK stays clickable.
                                    Range 5-300. We use the 300 maximum.
  SessionExpirationDurationInSeconds  how long the SESSION lasts once the
                                    link has been opened. Range 1800-43200.
                                    We use the 43200 maximum (12 hours).

So: click within 5 minutes, then stay signed in for 12 hours. That is long
enough to cover a rehearsal and the real call without regenerating.

The URL is ~3 KB, which terminals tend to wrap or truncate, so it is written
to a file as well as printed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import boto3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()
SERVER = os.environ.get("MLFLOW_TRACKING_SERVER", "")
EXPERIMENT = config.EXPERIMENT

# Maximums permitted by the API.
LINK_TTL = 300
SESSION_TTL = 43200


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--open", action="store_true",
                   help="launch the default browser immediately")
    p.add_argument("--server", default=SERVER,
                   help="tracking server name; or set MLFLOW_TRACKING_SERVER")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=REGION)

    if not args.server:
        servers = sm.list_mlflow_tracking_servers().get(
            "TrackingServerSummaries", [])
        if not servers:
            raise SystemExit(
                f"No MLflow tracking servers found in {REGION}. Create one in "
                "the SageMaker console under MLflow > Tracking Servers."
            )
        print("Available tracking servers:")
        for s in servers:
            print(f"  {s['TrackingServerName']}  ({s['TrackingServerStatus']})")
        raise SystemExit(
            "\nPass --server <name> or set MLFLOW_TRACKING_SERVER."
        )

    d = sm.describe_mlflow_tracking_server(TrackingServerName=args.server)
    if d["TrackingServerStatus"] not in ("Created", "Updated", "Started"):
        print(f"WARNING server status is {d['TrackingServerStatus']}")

    url = sm.create_presigned_mlflow_tracking_server_url(
        TrackingServerName=args.server,
        ExpiresInSeconds=LINK_TTL,
        SessionExpirationDurationInSeconds=SESSION_TTL,
    )["AuthorizedUrl"]

    out = ROOT / "outputs" / "mlflow_link.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(url, encoding="utf-8")

    base = d["TrackingServerUrl"]
    print(f"server      : {args.server}  ({d['TrackingServerStatus']})")
    print(f"base url    : {base}")
    print(f"link valid  : {LINK_TTL}s to click")
    print(f"session     : {SESSION_TTL // 3600}h once opened")
    print(f"\nwritten to  : {out}")
    print(f"url length  : {len(url)} chars")

    print("\nAfter it opens, find the experiment:")
    print(f"  {EXPERIMENT}")
    print("  One parent run per training job, with nested evaluation runs")
    print("  (eval-ab, eval-drift, eval-novel, eval-dial, eval-holdout).")

    print("\n--- URL ---")
    print(url)

    if args.open:
        try:
            if sys.platform == "win32":
                subprocess.run(["cmd", "/c", "start", "", url], check=False)
            elif sys.platform == "darwin":
                subprocess.run(["open", url], check=False)
            else:
                subprocess.run(["xdg-open", url], check=False)
            print("\nbrowser launched")
        except Exception as e:
            print(f"\ncould not launch browser: {type(e).__name__}")


if __name__ == "__main__":
    main()
