#!/usr/bin/env python
import argparse
import time
from typing import Any, Dict, List

from bucket_storage import build_s3_bucket_client_from_env
from pipeline_handoff import ARTIFACT_META_JSON, build_run_artifact_keys, get_runs_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete JD pipeline handoff artifacts from S3-compatible bucket.")
    parser.add_argument("--run-id", default="", help="Pipeline run id to delete (deletes runs/{run_id}/...).")
    parser.add_argument(
        "--prefix",
        default="",
        help="Custom prefix to delete. Use this if you need non-standard cleanup scope.",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=0.0,
        help="Delete all runs older than this age (hours). Uses expires_after_unix from meta when present.",
    )
    parser.add_argument(
        "--runs-prefix",
        default=get_runs_prefix(),
        help="Root prefix used for run storage (default from PIPELINE_RUNS_PREFIX or 'runs').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting.")
    return parser.parse_args()


def _discover_run_roots(bucket: Any, runs_prefix: str) -> List[str]:
    keys = bucket.list_keys(runs_prefix)
    roots = set()
    marker = f"{runs_prefix.strip().strip('/')}/"
    meta_suffix = f"/{ARTIFACT_META_JSON}"
    for key in keys:
        normalized = str(key or "").strip().strip("/")
        if not normalized:
            continue
        if normalized.endswith(meta_suffix):
            roots.add(normalized[: -len(meta_suffix)])
            continue
        idx = normalized.find(marker)
        if idx < 0:
            continue
        tail = normalized[idx + len(marker) :]
        run_id = tail.split("/", 1)[0].strip()
        if run_id:
            roots.add(f"{runs_prefix.strip().strip('/')}/{run_id}")
    return sorted(roots)


def _safe_read_meta(bucket: Any, root: str) -> Dict[str, Any]:
    meta_key = f"{root}/{ARTIFACT_META_JSON}"
    try:
        if not bucket.exists(meta_key):
            return {}
        payload = bucket.download_json(meta_key)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        print(f"[warn] failed reading meta for root={root}: {exc}")
    return {}


def _eligible_for_deletion(meta: Dict[str, Any], now_unix: float, older_than_hours: float) -> bool:
    expires_after = float(meta.get("expires_after_unix") or 0)
    if expires_after > 0:
        return now_unix >= expires_after

    if older_than_hours <= 0:
        return False

    reference_ts = float(
        meta.get("score_stage_completed_at_unix")
        or meta.get("source_stage_completed_at_unix")
        or meta.get("source_stage_started_at_unix")
        or 0
    )
    if reference_ts <= 0:
        return False
    return (now_unix - reference_ts) >= (older_than_hours * 3600.0)


def main() -> None:
    args = parse_args()
    if not args.run_id and not args.prefix and args.older_than_hours <= 0:
        raise SystemExit("Provide --run-id or --prefix or --older-than-hours.")

    bucket = build_s3_bucket_client_from_env()
    if args.run_id:
        delete_prefix = build_run_artifact_keys(args.run_id)["root"]
        if args.dry_run:
            print(f"[dry-run] would delete prefix: {delete_prefix}")
            return
        deleted_count = bucket.delete_prefix(delete_prefix)
        print(f"Deleted {deleted_count} objects under prefix: {delete_prefix}")
        return

    if args.prefix:
        delete_prefix = str(args.prefix).strip().strip("/")
        if not delete_prefix:
            raise SystemExit("Resolved empty prefix; refusing delete.")
        if args.dry_run:
            print(f"[dry-run] would delete prefix: {delete_prefix}")
            return
        deleted_count = bucket.delete_prefix(delete_prefix)
        print(f"Deleted {deleted_count} objects under prefix: {delete_prefix}")
        return

    runs_prefix = str(args.runs_prefix).strip().strip("/")
    now_unix = time.time()
    roots = _discover_run_roots(bucket=bucket, runs_prefix=runs_prefix)
    if not roots:
        print(f"No runs found under prefix: {runs_prefix}")
        return

    total_deleted = 0
    eligible = 0
    for root in roots:
        meta = _safe_read_meta(bucket=bucket, root=root)
        if not _eligible_for_deletion(meta=meta, now_unix=now_unix, older_than_hours=float(args.older_than_hours)):
            continue

        eligible += 1
        run_id = root.rsplit("/", 1)[-1]
        status = str(meta.get("pipeline_status") or "unknown")
        if args.dry_run:
            print(f"[dry-run] would delete run_id={run_id} root={root} status={status}")
            continue
        deleted = bucket.delete_prefix(root)
        total_deleted += deleted
        print(f"deleted run_id={run_id} root={root} status={status} objects={deleted}")

    if args.dry_run:
        print(f"[dry-run] eligible_runs={eligible} scanned_runs={len(roots)}")
    else:
        print(f"Cleanup complete: deleted_objects={total_deleted} eligible_runs={eligible} scanned_runs={len(roots)}")


if __name__ == "__main__":
    main()
