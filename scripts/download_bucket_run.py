#!/usr/bin/env python
import argparse
import os

from bucket_storage import build_s3_bucket_client_from_env
from pipeline_handoff import build_run_artifact_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all bucket artifacts for a pipeline run.")
    parser.add_argument("--run-id", required=True, help="Run id to download.")
    parser.add_argument(
        "--output-dir",
        default="bucket-runs",
        help="Local destination root directory (default: ./bucket-runs).",
    )
    return parser.parse_args()


def _relative_key_for_run(key: str, run_id: str) -> str:
    marker = f"/{run_id}/"
    normalized = str(key or "").strip().lstrip("/")
    idx = normalized.find(marker)
    if idx < 0:
        return os.path.basename(normalized)
    rel = normalized[idx + len(marker) :]
    return rel or os.path.basename(normalized)


def main() -> None:
    args = parse_args()
    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("run_id cannot be empty.")

    bucket = build_s3_bucket_client_from_env()
    artifact_keys = build_run_artifact_keys(run_id)
    root = artifact_keys["root"]
    keys = bucket.list_keys(root)
    if not keys:
        raise SystemExit(f"No objects found for run_id={run_id} under prefix={root}")

    output_root = os.path.abspath(str(args.output_dir))
    run_output_dir = os.path.join(output_root, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    downloaded = 0
    for key in sorted(keys):
        rel = _relative_key_for_run(key, run_id=run_id)
        destination = os.path.join(run_output_dir, rel)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        info = bucket.download_file(key, destination)
        downloaded += 1
        print(f"downloaded key={key} -> {info['path']} ({info['size_bytes']} bytes)")

    print(f"Run download complete: run_id={run_id} files={downloaded} output_dir={run_output_dir}")


if __name__ == "__main__":
    main()
