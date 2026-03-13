#!/usr/bin/env python
import argparse
import datetime as dt
from typing import Any, Dict, List

from bucket_storage import build_s3_bucket_client_from_env
from pipeline_handoff import ARTIFACT_META_JSON, get_runs_prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List JD pipeline runs found in bucket storage.")
    parser.add_argument(
        "--runs-prefix",
        default=get_runs_prefix(),
        help="Root prefix used for run storage (default from PIPELINE_RUNS_PREFIX or 'runs').",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows to print.")
    return parser.parse_args()


def _discover_run_roots(bucket: Any, runs_prefix: str) -> List[str]:
    keys = bucket.list_keys(runs_prefix)
    roots = set()
    meta_suffix = f"/{ARTIFACT_META_JSON}"
    marker = f"{runs_prefix.strip().strip('/')}/"
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


def _safe_download_meta(bucket: Any, root: str) -> Dict[str, Any]:
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


def _fmt_ts(value: Any) -> str:
    try:
        ts = float(value or 0)
    except (TypeError, ValueError):
        return "-"
    if ts <= 0:
        return "-"
    return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%SZ")


def main() -> None:
    args = parse_args()
    bucket = build_s3_bucket_client_from_env()
    roots = _discover_run_roots(bucket=bucket, runs_prefix=str(args.runs_prefix))
    if not roots:
        print(f"No runs found under prefix: {args.runs_prefix}")
        return

    rows: List[Dict[str, Any]] = []
    for root in roots:
        meta = _safe_download_meta(bucket=bucket, root=root)
        run_id = root.rsplit("/", 1)[-1]
        rows.append(
            {
                "run_id": run_id,
                "status": str(meta.get("pipeline_status") or "unknown"),
                "source_status": str(meta.get("source_stage_status") or "-"),
                "score_status": str(meta.get("score_stage_status") or "-"),
                "expires_at": _fmt_ts(meta.get("expires_after_unix")),
                "rows_after_dedup": int(meta.get("rows_after_dedup") or 0),
                "jd_name": str(meta.get("jd_name") or ""),
                "updated_at": _fmt_ts(
                    meta.get("score_stage_completed_at_unix")
                    or meta.get("score_stage_failed_at_unix")
                    or meta.get("source_stage_completed_at_unix")
                    or meta.get("source_stage_failed_at_unix")
                    or meta.get("source_stage_started_at_unix")
                ),
            }
        )

    rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
    print("run_id\tstatus\tsource\tscore\texpires_at\trows_after_dedup\tjd_name")
    for row in rows[: max(1, int(args.limit))]:
        print(
            f"{row['run_id']}\t{row['status']}\t{row['source_status']}\t{row['score_status']}\t"
            f"{row['expires_at']}\t{row['rows_after_dedup']}\t{row['jd_name']}"
        )


if __name__ == "__main__":
    main()
