#!/usr/bin/env python
import argparse

from dotenv import load_dotenv

from bucket_storage import build_s3_bucket_client_from_env
from pipeline_handoff import build_run_artifact_keys
from queueing import enqueue_jd_score_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Requeue score-stage job from existing run artifacts.")
    parser.add_argument("--run-id", required=True, help="Existing run id with handoff artifacts in bucket.")
    parser.add_argument("--channel-id", default="", help="Override channel id (defaults to meta.json value).")
    parser.add_argument("--jd-name", default="", help="Override JD name (defaults to meta.json value).")
    parser.add_argument(
        "--jd-test-mode",
        choices=["auto", "true", "false"],
        default="auto",
        help="Use auto to inherit meta; or force true/false.",
    )
    return parser.parse_args()


def _resolve_jd_test_mode(raw: str, fallback: bool) -> bool:
    if raw == "auto":
        return bool(fallback)
    return raw == "true"


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("run_id cannot be empty.")

    bucket = build_s3_bucket_client_from_env()
    artifact_keys = build_run_artifact_keys(run_id)
    meta_key = artifact_keys["meta"]
    if not bucket.exists(meta_key):
        raise SystemExit(f"Missing meta artifact for run_id={run_id}: {meta_key}")
    meta_payload = bucket.download_json(meta_key)
    if not isinstance(meta_payload, dict):
        raise SystemExit(f"Invalid meta artifact payload for run_id={run_id}: expected object.")

    channel_id = str(args.channel_id or meta_payload.get("channel_id") or "").strip()
    if not channel_id:
        raise SystemExit("channel_id is required. Provide --channel-id or ensure meta.json includes channel_id.")

    jd_name = str(args.jd_name or meta_payload.get("jd_name") or "").strip()
    jd_test_mode = _resolve_jd_test_mode(
        raw=str(args.jd_test_mode),
        fallback=bool(meta_payload.get("jd_test_mode") or False),
    )

    payload = {
        "run_id": run_id,
        "channel_id": channel_id,
        "jd_name": jd_name,
        "jd_test_mode": jd_test_mode,
        "event_id": str(meta_payload.get("event_id") or ""),
        "artifact_root_key": artifact_keys["root"],
    }
    job = enqueue_jd_score_job(payload)
    print(f"Requeued score job: run_id={run_id} job_id={job.id} queue_payload_keys={sorted(payload.keys())}")


if __name__ == "__main__":
    main()
