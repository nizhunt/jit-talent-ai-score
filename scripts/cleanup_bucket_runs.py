#!/usr/bin/env python
import argparse

from bucket_storage import build_s3_bucket_client_from_env
from pipeline_handoff import build_run_artifact_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete JD pipeline handoff artifacts from S3-compatible bucket.")
    parser.add_argument("--run-id", default="", help="Pipeline run id to delete (deletes runs/{run_id}/...).")
    parser.add_argument(
        "--prefix",
        default="",
        help="Custom prefix to delete. Use this if you need non-standard cleanup scope.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_id and not args.prefix:
        raise SystemExit("Provide either --run-id or --prefix.")

    bucket = build_s3_bucket_client_from_env()
    if args.run_id:
        delete_prefix = build_run_artifact_keys(args.run_id)["root"]
    else:
        delete_prefix = str(args.prefix).strip().strip("/")
    if not delete_prefix:
        raise SystemExit("Resolved empty prefix; refusing delete.")

    deleted_count = bucket.delete_prefix(delete_prefix)
    print(f"Deleted {deleted_count} objects under prefix: {delete_prefix}")


if __name__ == "__main__":
    main()
