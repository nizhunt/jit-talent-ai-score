import argparse
import hashlib
import os
import re
import signal
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from rq import Worker

from bucket_storage import S3BucketClient, build_s3_bucket_client_from_env
from dashboard_logger import log_enrichment_dashboard_row, log_jd_processing_dashboard_row
from pipeline_handoff import ARTIFACT_META_JSON, build_run_artifact_keys, get_runs_prefix, required_handoff_keys
from process_candidates import (
    parse_args,
    post_slack_message,
    run_score_stage_from_handoff,
    run_source_stage_from_jd_text,
)
from queueing import (
    clear_event_seen,
    enqueue_jd_score_job,
    get_admin_queue_name,
    get_jd_queue_name,
    get_jd_score_queue_name,
    get_jd_source_queue_name,
    get_redis_connection,
    get_reply_queue_name,
)
from thread_reply_enrichment import post_thread_reply_update, run_thread_reply_enrichment_pipeline


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _is_truthy(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _slugify_identifier(value: str, *, default: str = "jd", max_length: int = 48) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").strip().lower()).strip("-")
    if not cleaned:
        return default
    return cleaned[:max_length].strip("-") or default


def _jd_prefix(jd_name: Optional[str]) -> str:
    name = (jd_name or "").strip()
    if not name:
        return ""
    return f"[{name}] "


def _stable_run_id_from_event(event_id: str) -> str:
    digest = hashlib.sha1(event_id.encode("utf-8")).hexdigest()[:12]
    return digest


def _resolve_run_id(run_id: Optional[str], event_id: Optional[str]) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "", str(run_id or "")).strip("-_")
    if normalized:
        return normalized[:48]
    if event_id:
        return _stable_run_id_from_event(event_id)
    return uuid.uuid4().hex[:12]


def _score_job_id(run_id: str) -> str:
    # RQ job ids cannot contain ":" in this runtime.
    return f"jd-score-{run_id}"


def _build_bucket_client() -> S3BucketClient:
    return build_s3_bucket_client_from_env()


def _artifact_retention_hours() -> float:
    raw = os.getenv("PIPELINE_ARTIFACT_RETENTION_HOURS")
    if raw is None:
        return 24.0
    try:
        value = float(raw)
    except ValueError:
        print(f"[warn] invalid PIPELINE_ARTIFACT_RETENTION_HOURS={raw!r}; using 24")
        return 24.0
    if value <= 0:
        print(f"[warn] invalid PIPELINE_ARTIFACT_RETENTION_HOURS={raw!r}; using 24")
        return 24.0
    return value


def _best_effort_update_meta(
    *,
    bucket: S3BucketClient,
    artifact_keys: Dict[str, str],
    updates: Dict[str, Any],
) -> None:
    try:
        existing = _load_existing_meta_if_present(bucket=bucket, artifact_keys=artifact_keys)
        merged = dict(existing)
        merged.update(updates)
        bucket.upload_json(artifact_keys["meta"], merged)
    except Exception as exc:
        print(f"[warn] run meta update failed for key={artifact_keys['meta']}: {exc}")


def _maybe_upload_file(
    *,
    bucket: S3BucketClient,
    key: str,
    path: str,
    content_type: Optional[str] = None,
) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        bucket.upload_file(key, path, content_type=content_type)
    except Exception as exc:
        print(f"[warn] failed to upload artifact key={key} from path={path}: {exc}")


def _admin_user_allowed(user_id: str) -> bool:
    allowed_raw = os.getenv("SLACK_ADMIN_USER_IDS", "").strip()
    if not allowed_raw:
        return True
    allowed = {part.strip() for part in allowed_raw.split(",") if part.strip()}
    return bool(user_id and user_id in allowed)


def _discover_run_roots(bucket: S3BucketClient, runs_prefix: str) -> List[str]:
    keys = bucket.list_keys(runs_prefix)
    roots = set()
    prefix_marker = f"{runs_prefix.strip().strip('/')}/"
    meta_suffix = f"/{ARTIFACT_META_JSON}"
    for key in keys:
        normalized = str(key or "").strip().strip("/")
        if not normalized:
            continue
        if normalized.endswith(meta_suffix):
            roots.add(normalized[: -len(meta_suffix)])
            continue
        idx = normalized.find(prefix_marker)
        if idx < 0:
            continue
        tail = normalized[idx + len(prefix_marker) :]
        run_id = tail.split("/", 1)[0].strip()
        if run_id:
            roots.add(f"{runs_prefix.strip().strip('/')}/{run_id}")
    return sorted(roots)


def _safe_read_run_meta(bucket: S3BucketClient, root: str) -> Dict[str, Any]:
    meta_key = f"{root}/{ARTIFACT_META_JSON}"
    try:
        if not bucket.exists(meta_key):
            return {}
        payload = bucket.download_json(meta_key)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        print(f"[warn] failed reading run meta key={meta_key}: {exc}")
    return {}


def _format_unix_utc(value: Any) -> str:
    try:
        ts = float(value or 0)
    except (TypeError, ValueError):
        return "-"
    if ts <= 0:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(ts))


def _latest_run_timestamp(meta: Dict[str, Any]) -> float:
    for key in [
        "score_stage_completed_at_unix",
        "score_stage_failed_at_unix",
        "source_stage_completed_at_unix",
        "source_stage_failed_at_unix",
        "source_stage_started_at_unix",
        "pipeline_started_at_unix",
    ]:
        try:
            value = float(meta.get(key) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 0.0


def _eligible_for_cleanup(meta: Dict[str, Any], now_unix: float, older_than_hours: float) -> bool:
    expires_after = float(meta.get("expires_after_unix") or 0)
    if expires_after > 0:
        return now_unix >= expires_after
    if older_than_hours <= 0:
        return False
    ref = _latest_run_timestamp(meta)
    if ref <= 0:
        return False
    return (now_unix - ref) >= (older_than_hours * 3600.0)


def _slack_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _build_enrichment_summary_blocks(
    *,
    result: Dict[str, Any],
    threshold: float,
    source_name: str,
    source_url: str,
    campaign_name: str,
    campaign_analytics_url: str,
    lead_skipped: List[str],
    lead_errors: List[str],
    dashboard_warning: str,
) -> List[Dict[str, Any]]:
    has_campaign = bool(result.get("campaign_id"))
    title = "Thread enrichment complete." if has_campaign else "Thread enrichment completed with no campaign created."
    blocks: List[Dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{_slack_escape(title)}*"}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Source*\n{_slack_escape(source_name or 'N/A')}"},
                {"type": "mrkdwn", "text": f"*JD Name*\n{_slack_escape(str(result.get('jd_name') or 'N/A'))}"},
                {"type": "mrkdwn", "text": f"*Threshold*\n{threshold:g}"},
                {
                    "type": "mrkdwn",
                    "text": f"*Rows with score+LinkedIn parsed*\n{int(result.get('rows_with_score_and_linkedin') or 0)}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Candidates meeting threshold*\n{int(result.get('rows_meeting_threshold') or 0)}",
                },
                {"type": "mrkdwn", "text": f"*SaleSQL emails*\n{int(result.get('salesql_emails_found') or 0)}"},
                {"type": "mrkdwn", "text": f"*Reoon passed*\n{int(result.get('reoon_passed') or 0)}"},
                {"type": "mrkdwn", "text": f"*BounceBan deliverable*\n{int(result.get('bounceban_deliverable') or 0)}"},
            ],
        },
    ]

    if source_url:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Source URL*\n<{source_url}|Open source sheet>"}})

    if has_campaign:
        blocks.append(
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Instantly campaign name*\n{_slack_escape(campaign_name or 'N/A')}"},
                    {"type": "mrkdwn", "text": f"*Leads added*\n{int(result.get('leads_added') or 0)}"},
                    {"type": "mrkdwn", "text": f"*Leads skipped*\n{len(lead_skipped)}"},
                    {"type": "mrkdwn", "text": f"*Lead add errors*\n{len(lead_errors)}"},
                ],
            }
        )
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Instantly campaign*\n<{campaign_analytics_url}|Open campaign analytics>"},
            }
        )

    if lead_skipped:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Skip sample*\n`{_slack_escape(lead_skipped[0])}`"}})

    note = str(result.get("note") or "").strip()
    if note:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Note*\n{_slack_escape(note)}"}})

    if dashboard_warning:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"*Dashboard*: {_slack_escape(dashboard_warning)}"}]})

    return blocks


def build_pipeline_args(
    channel_id: str,
    jd_name: Optional[str] = None,
    jd_test_mode: bool = False,
    run_id: Optional[str] = None,
) -> argparse.Namespace:
    args = parse_args([])
    args.channel_id = channel_id
    args.jd_name = (jd_name or "").strip()
    args.jd_test_mode = bool(jd_test_mode)
    if args.jd_test_mode:
        args.num_results_per_query = 100
    args.debug = False
    args.stop_after = None

    project_root = os.path.dirname(os.path.abspath(__file__))
    args.exa_query_prompt_path = os.path.join(project_root, "prompts", "exa-querry-creator-prompt.md")
    args.scorer_prompt_path = os.path.join(project_root, "prompts", "scorer_prompt.md")
    args.jd_widening_prompts_dir = os.path.join(project_root, "prompts", "jd-widening-prompts")

    resolved_run_id = _resolve_run_id(run_id, event_id=None)
    jd_slug = _slugify_identifier(args.jd_name, default="jd")
    run_dir = os.path.join("/tmp", f"jit-talent-worker-{jd_slug}-{resolved_run_id}")
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir
    args.run_id = resolved_run_id

    args.jd_path = os.path.join(run_dir, "jd.md")
    if args.jd_name:
        args.candidates_csv = os.path.join(run_dir, f"candidates-{jd_slug}.csv")
        args.dedup_csv = os.path.join(run_dir, f"candidates-dedup-{jd_slug}.csv")
        args.scored_csv = os.path.join(run_dir, f"candidates-scored-{jd_slug}.csv")
    else:
        args.candidates_csv = os.path.join(run_dir, "candidates.csv")
        args.dedup_csv = os.path.join(run_dir, "candidates-dedup.csv")
        args.scored_csv = os.path.join(run_dir, "candidates-scored.csv")
    args.debug_dir = os.path.join(run_dir, "debug")
    return args


def cleanup_run_dir(run_dir: Optional[str]) -> None:
    if not run_dir:
        return
    try:
        shutil.rmtree(run_dir)
    except FileNotFoundError:
        return
    except Exception as exc:
        print(f"[warn] failed to clean up run dir '{run_dir}': {exc}")


def _load_existing_meta_if_present(bucket: S3BucketClient, artifact_keys: Dict[str, str]) -> Dict[str, Any]:
    meta_key = artifact_keys["meta"]
    try:
        if not bucket.exists(meta_key):
            return {}
        payload = bucket.download_json(meta_key)
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        print(f"[warn] failed to read existing meta from bucket key={meta_key}: {exc}")
    return {}


def _upload_source_handoff_artifacts(
    *,
    bucket: S3BucketClient,
    artifact_keys: Dict[str, str],
    args: argparse.Namespace,
    jd_text: str,
    source_result: Dict[str, Any],
    channel_id: str,
    jd_name: Optional[str],
    jd_test_mode: bool,
    event_id: Optional[str],
    message_ts: Optional[str],
    pipeline_started_at_unix: float,
) -> Dict[str, Any]:
    source_completed_at_unix = time.time()
    retention_hours = _artifact_retention_hours()
    expires_after_unix = source_completed_at_unix + (retention_hours * 3600.0)

    existing_meta = _load_existing_meta_if_present(bucket=bucket, artifact_keys=artifact_keys)
    persisted_started_at = float(existing_meta.get("pipeline_started_at_unix") or pipeline_started_at_unix)

    queries = source_result.get("queries") or []
    if not isinstance(queries, list):
        queries = []
    query_generation_usage = source_result.get("query_generation_usage") or {}
    if not isinstance(query_generation_usage, dict):
        query_generation_usage = {}

    meta: Dict[str, Any] = {
        "run_id": getattr(args, "run_id", ""),
        "channel_id": channel_id,
        "jd_name": (jd_name or "").strip(),
        "jd_test_mode": bool(jd_test_mode),
        "event_id": event_id or "",
        "message_ts": message_ts or "",
        "pipeline_started_at_unix": persisted_started_at,
        "source_stage_status": "completed",
        "source_stage_completed_at_unix": source_completed_at_unix,
        "pipeline_status": "source_completed_score_queued",
        "retention_hours": retention_hours,
        "expires_after_unix": expires_after_unix,
        "queries_count": int(source_result.get("queries_count") or len(queries)),
        "total_results": int(source_result.get("total_results") or 0),
        "rows_before_dedup": int(source_result.get("rows_before_dedup") or 0),
        "rows_after_dedup": int(source_result.get("rows_after_dedup") or 0),
        "query_generation_usage": query_generation_usage,
        "artifact_keys": artifact_keys,
        "source_attempt_count": int(existing_meta.get("source_attempt_count") or 0) + 1,
    }

    print(
        f"[pipeline] run_id={meta['run_id']} uploading_handoff "
        f"keys={','.join(required_handoff_keys(artifact_keys))}"
    )
    bucket.upload_text(artifact_keys["jd_text"], jd_text)
    bucket.upload_file(artifact_keys["candidates_csv"], args.candidates_csv, content_type="text/csv")
    bucket.upload_file(artifact_keys["dedup_csv"], args.dedup_csv, content_type="text/csv")
    bucket.upload_json(artifact_keys["jd_context_by_prompt"], source_result.get("jd_context_by_prompt_file") or {})
    bucket.upload_json(artifact_keys["queries"], queries)
    bucket.upload_json(artifact_keys["meta"], meta)
    return meta


def _assert_required_handoff_artifacts(bucket: S3BucketClient, artifact_keys: Dict[str, str]) -> None:
    missing: List[str] = []
    for key in required_handoff_keys(artifact_keys):
        if not bucket.exists(key):
            missing.append(key)
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required handoff artifacts for run: {missing_list}")


def _download_score_handoff_artifacts(
    *,
    bucket: S3BucketClient,
    artifact_keys: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    _assert_required_handoff_artifacts(bucket=bucket, artifact_keys=artifact_keys)
    bucket.download_file(artifact_keys["jd_text"], args.jd_path)
    bucket.download_file(artifact_keys["dedup_csv"], args.dedup_csv)
    with open(args.jd_path, "r", encoding="utf-8") as handle:
        jd_text = handle.read()
    jd_context = bucket.download_json(artifact_keys["jd_context_by_prompt"])
    queries = bucket.download_json(artifact_keys["queries"])
    meta = bucket.download_json(artifact_keys["meta"])
    if not isinstance(jd_context, dict):
        raise RuntimeError("Invalid jd_context_by_prompt artifact: expected JSON object.")
    if not isinstance(queries, list):
        raise RuntimeError("Invalid queries artifact: expected JSON array.")
    if not isinstance(meta, dict):
        raise RuntimeError("Invalid meta artifact: expected JSON object.")
    return {
        "jd_text": jd_text,
        "jd_context_by_prompt_file": jd_context,
        "queries": queries,
        "meta": meta,
    }


def _post_final_summary_once(
    *,
    bucket: S3BucketClient,
    artifact_keys: Dict[str, str],
    slack_token: str,
    channel_id: str,
    run_id: str,
    result_message_text: str,
    result_message_blocks: List[Dict[str, Any]],
) -> str:
    marker_key = artifact_keys["final_summary_posted"]
    if bucket.exists(marker_key):
        print(f"[pipeline] run_id={run_id} final_summary_already_posted key={marker_key}; skipping")
        return "already_posted"

    post_response = post_slack_message(
        slack_token=slack_token,
        channel_id=channel_id,
        text=result_message_text,
        blocks=result_message_blocks,
        unfurl_links=False,
        unfurl_media=False,
    )
    try:
        bucket.upload_json(
            marker_key,
            {
                "run_id": run_id,
                "posted_at_unix": time.time(),
                "channel_id": channel_id,
                "slack_ts": str(post_response.get("ts") or ""),
            },
        )
        print(f"[pipeline] run_id={run_id} final_summary_posted key={marker_key}")
        return "posted"
    except Exception as exc:
        print(f"[warn] run_id={run_id} failed to persist final summary marker key={marker_key}: {exc}")
        return "posted_marker_failed"


def _notify_failure(
    channel_id: str,
    error_msg: str,
    event_id: Optional[str] = None,
    jd_name: Optional[str] = None,
) -> None:
    """Best-effort: post failure to Slack and clear event dedup so JD can be retried."""
    try:
        slack_token = os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_USER_TOKEN")
        if slack_token and channel_id:
            short_error = (error_msg[:300] + "...") if len(error_msg) > 300 else error_msg
            post_slack_message(
                slack_token=slack_token,
                channel_id=channel_id,
                text=f"{_jd_prefix(jd_name)}⚠️ Pipeline failed: {short_error}\nPlease re-post the JD to retry.",
            )
    except Exception as exc:
        print(f"[warn] failed to post failure message to Slack: {exc}")

    if event_id:
        try:
            clear_event_seen(event_id)
        except Exception as exc:
            print(f"[warn] failed to clear event dedup marker: {exc}")


def _notify_thread_failure(
    channel_id: str,
    thread_ts: str,
    error_msg: str,
    event_id: Optional[str] = None,
) -> None:
    try:
        slack_token = os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_USER_TOKEN")
        if slack_token and channel_id and thread_ts:
            short_error = (error_msg[:300] + "...") if len(error_msg) > 300 else error_msg
            post_thread_reply_update(
                slack_token=slack_token,
                channel_id=channel_id,
                thread_ts=thread_ts,
                text=f"Thread enrichment failed: {short_error}\nReply with the threshold again to retry.",
            )
    except Exception as exc:
        print(f"[warn] failed to post thread failure message to Slack: {exc}")

    if event_id:
        try:
            clear_event_seen(event_id)
        except Exception as exc:
            print(f"[warn] failed to clear event dedup marker: {exc}")


def process_jd_source_job(
    *,
    jd_text: str,
    message_ts: Optional[str],
    channel_id: str,
    jd_name: Optional[str] = None,
    jd_test_mode: bool = False,
    event_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()
    _ = message_ts  # reserved for compatibility with existing enqueued payloads
    run_dir: Optional[str] = None
    args: Optional[argparse.Namespace] = None
    resolved_run_id = _resolve_run_id(run_id, event_id)
    pipeline_started_at_unix = time.time()
    artifact_keys = build_run_artifact_keys(resolved_run_id)

    openai_api_key = require_env("OPENAI_API_KEY")
    exa_api_key = require_env("EXA_API_KEY")
    slack_token = os.getenv("SLACK_BOT_TOKEN") or require_env("SLACK_USER_TOKEN")
    bucket = _build_bucket_client()

    print(f"[pipeline] run_id={resolved_run_id} source_started")
    _best_effort_update_meta(
        bucket=bucket,
        artifact_keys=artifact_keys,
        updates={
            "run_id": resolved_run_id,
            "channel_id": channel_id,
            "jd_name": (jd_name or "").strip(),
            "jd_test_mode": bool(jd_test_mode),
            "event_id": event_id or "",
            "message_ts": message_ts or "",
            "pipeline_started_at_unix": pipeline_started_at_unix,
            "source_stage_status": "running",
            "source_stage_started_at_unix": pipeline_started_at_unix,
            "pipeline_status": "source_running",
        },
    )

    post_slack_message(
        slack_token=slack_token,
        channel_id=channel_id,
        text=(
            f"{_jd_prefix(jd_name)}Thanks for sharing the JD. Processing has started."
            + (" (quick mode: direct JD query, max 100 results)." if jd_test_mode else "")
        ),
    )

    try:
        client = OpenAI(api_key=openai_api_key)
        args = build_pipeline_args(
            channel_id=channel_id,
            jd_name=jd_name,
            jd_test_mode=jd_test_mode,
            run_id=resolved_run_id,
        )
        run_dir = getattr(args, "run_dir", None)
        source_result = run_source_stage_from_jd_text(
            jd_text=jd_text,
            args=args,
            client=client,
            exa_api_key=exa_api_key,
        )

        meta = _upload_source_handoff_artifacts(
            bucket=bucket,
            artifact_keys=artifact_keys,
            args=args,
            jd_text=jd_text,
            source_result=source_result,
            channel_id=channel_id,
            jd_name=jd_name,
            jd_test_mode=jd_test_mode,
            event_id=event_id,
            message_ts=message_ts,
            pipeline_started_at_unix=pipeline_started_at_unix,
        )

        score_payload = {
            "run_id": resolved_run_id,
            "channel_id": channel_id,
            "jd_name": (jd_name or "").strip(),
            "jd_test_mode": bool(jd_test_mode),
            "event_id": event_id,
            "artifact_root_key": artifact_keys["root"],
        }
        print(f"[queue] run_id={resolved_run_id} score_queued payload_keys={','.join(sorted(score_payload.keys()))}")
        score_job = enqueue_jd_score_job(score_payload, job_id=_score_job_id(resolved_run_id))

        return {
            "ok": True,
            "event_id": event_id,
            "jd_name": jd_name,
            "jd_test_mode": bool(jd_test_mode),
            "run_id": resolved_run_id,
            "artifact_keys": artifact_keys,
            "meta": meta,
            "score_job_id": score_job.id,
            "status": "source_completed_score_queued",
        }
    except Exception as exc:
        if args is not None:
            maybe_candidates_path = str(getattr(args, "candidates_csv", "") or "")
            maybe_dedup_path = str(getattr(args, "dedup_csv", "") or "")
            _maybe_upload_file(
                bucket=bucket,
                key=artifact_keys["candidates_csv"],
                path=maybe_candidates_path,
                content_type="text/csv",
            )
            _maybe_upload_file(
                bucket=bucket,
                key=artifact_keys["dedup_csv"],
                path=maybe_dedup_path,
                content_type="text/csv",
            )
        _best_effort_update_meta(
            bucket=bucket,
            artifact_keys=artifact_keys,
            updates={
                "source_stage_status": "failed",
                "source_stage_failed_at_unix": time.time(),
                "source_stage_error": str(exc),
                "pipeline_status": "source_failed",
            },
        )
        _notify_failure(
            channel_id=channel_id,
            error_msg=str(exc),
            event_id=event_id,
            jd_name=jd_name,
        )
        raise  # Re-raise so RQ marks the job as failed
    finally:
        cleanup_run_dir(run_dir)


def process_jd_score_job(
    *,
    run_id: str,
    channel_id: str,
    jd_name: Optional[str] = None,
    jd_test_mode: bool = False,
    event_id: Optional[str] = None,
    artifact_root_key: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()
    run_dir: Optional[str] = None
    args: Optional[argparse.Namespace] = None
    resolved_run_id = _resolve_run_id(run_id, event_id)

    openai_api_key = require_env("OPENAI_API_KEY")
    slack_token = os.getenv("SLACK_BOT_TOKEN") or require_env("SLACK_USER_TOKEN")
    bucket = _build_bucket_client()
    artifact_keys = build_run_artifact_keys(resolved_run_id)
    if artifact_root_key and artifact_root_key != artifact_keys["root"]:
        print(
            f"[warn] run_id={resolved_run_id} artifact_root_key mismatch "
            f"payload={artifact_root_key} derived={artifact_keys['root']}; using derived key"
        )

    print(f"[pipeline] run_id={resolved_run_id} score_started")
    resolved_jd_name = (jd_name or "").strip()

    try:
        client = OpenAI(api_key=openai_api_key)
        args = build_pipeline_args(
            channel_id=channel_id,
            jd_name=resolved_jd_name,
            jd_test_mode=jd_test_mode,
            run_id=resolved_run_id,
        )
        run_dir = getattr(args, "run_dir", None)

        handoff = _download_score_handoff_artifacts(
            bucket=bucket,
            artifact_keys=artifact_keys,
            args=args,
        )
        meta = handoff["meta"]
        if not resolved_jd_name:
            resolved_jd_name = str(meta.get("jd_name") or "").strip()
            args.jd_name = resolved_jd_name

        _best_effort_update_meta(
            bucket=bucket,
            artifact_keys=artifact_keys,
            updates={
                "score_stage_status": "running",
                "score_stage_started_at_unix": time.time(),
                "score_attempt_count": int(meta.get("score_attempt_count") or 0) + 1,
                "pipeline_status": "score_running",
            },
        )

        source_stage_result = {
            "jd_name": resolved_jd_name,
            "jd_text": handoff.get("jd_text") or "",
            "jd_test_mode": bool(meta.get("jd_test_mode", jd_test_mode)),
            "queries": handoff.get("queries") or [],
            "queries_count": int(meta.get("queries_count") or len(handoff.get("queries") or [])),
            "total_results": int(meta.get("total_results") or 0),
            "rows_before_dedup": int(meta.get("rows_before_dedup") or 0),
            "rows_after_dedup": int(meta.get("rows_after_dedup") or 0),
            "query_generation_usage": meta.get("query_generation_usage") or {},
            "jd_context_by_prompt_file": handoff.get("jd_context_by_prompt_file") or {},
        }

        result = run_score_stage_from_handoff(
            args=args,
            client=client,
            slack_token=slack_token,
            source_stage_result=source_stage_result,
            pipeline_started_at_unix=float(meta.get("pipeline_started_at_unix") or 0) or None,
            post_final_message=False,
        )

        final_post_status = _post_final_summary_once(
            bucket=bucket,
            artifact_keys=artifact_keys,
            slack_token=slack_token,
            channel_id=channel_id,
            run_id=resolved_run_id,
            result_message_text=str(result.get("result_message_text") or ""),
            result_message_blocks=result.get("result_message_blocks") or [],
        )

        _maybe_upload_file(
            bucket=bucket,
            key=artifact_keys["scored_csv"],
            path=str(getattr(args, "scored_csv", "") or ""),
            content_type="text/csv",
        )
        _maybe_upload_file(
            bucket=bucket,
            key=artifact_keys["sheet_ready_csv"],
            path=str(result.get("sheet_ready_csv") or ""),
            content_type="text/csv",
        )

        try:
            log_jd_processing_dashboard_row(
                jd_name=resolved_jd_name,
                candidate_sheet_url=str(result.get("google_sheet_url") or ""),
                total_profiles_found=int(result.get("rows_before_dedup") or 0),
                profiles_after_dedup=int(result.get("rows_after_dedup") or 0),
                score_counts_by_score=result.get("score_counts_by_score") or {},
            )
        except Exception as dashboard_exc:
            print(f"[warn] dashboard JD logging failed: {dashboard_exc}")

        updated_meta = dict(meta)
        updated_meta["score_stage_status"] = "completed"
        updated_meta["score_attempt_count"] = int(meta.get("score_attempt_count") or 0) + 1
        updated_meta["score_stage_completed_at_unix"] = time.time()
        updated_meta["final_summary_post_status"] = final_post_status
        updated_meta["google_sheet_url"] = str(result.get("google_sheet_url") or "")
        updated_meta["pipeline_status"] = "completed"
        bucket.upload_json(artifact_keys["meta"], updated_meta)

        return {
            "ok": True,
            "event_id": event_id,
            "jd_name": resolved_jd_name,
            "jd_test_mode": bool(jd_test_mode),
            "run_id": resolved_run_id,
            "result": result,
            "final_summary_post_status": final_post_status,
        }
    except Exception as exc:
        if args is not None:
            _maybe_upload_file(
                bucket=bucket,
                key=artifact_keys["scored_partial_csv"],
                path=str(getattr(args, "scored_csv", "") or ""),
                content_type="text/csv",
            )
        _best_effort_update_meta(
            bucket=bucket,
            artifact_keys=artifact_keys,
            updates={
                "score_stage_status": "failed",
                "score_stage_failed_at_unix": time.time(),
                "score_stage_error": str(exc),
                "pipeline_status": "score_failed",
            },
        )
        _notify_failure(
            channel_id=channel_id,
            error_msg=str(exc),
            event_id=event_id,
            jd_name=resolved_jd_name or jd_name,
        )
        raise
    finally:
        cleanup_run_dir(run_dir)


def process_jd_pipeline_job(
    *,
    jd_text: str,
    message_ts: Optional[str],
    channel_id: str,
    jd_name: Optional[str] = None,
    jd_test_mode: bool = False,
    event_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    # Backward-compatible entrypoint. Legacy callers enqueue this and now land in source stage.
    return process_jd_source_job(
        jd_text=jd_text,
        message_ts=message_ts,
        channel_id=channel_id,
        jd_name=jd_name,
        jd_test_mode=jd_test_mode,
        event_id=event_id,
        run_id=run_id,
    )


def process_jd_admin_job(
    *,
    channel_id: str,
    message_ts: Optional[str],
    user_id: str,
    command: Dict[str, Any],
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()
    slack_token = os.getenv("SLACK_BOT_TOKEN") or require_env("SLACK_USER_TOKEN")

    def _reply(text: str) -> None:
        post_slack_message(
            slack_token=slack_token,
            channel_id=channel_id,
            thread_ts=message_ts,
            text=text,
        )

    if not _admin_user_allowed(user_id):
        _reply("You are not allowed to run admin commands in this workspace.")
        return {"ok": False, "event_id": event_id, "error": "admin_not_allowed"}

    bucket = _build_bucket_client()
    action = str(command.get("action") or "").strip()
    runs_prefix = get_runs_prefix()

    try:
        if action == "list_runs":
            limit = int(command.get("limit") or 20)
            roots = _discover_run_roots(bucket=bucket, runs_prefix=runs_prefix)
            rows: List[Dict[str, Any]] = []
            for root in roots:
                meta = _safe_read_run_meta(bucket=bucket, root=root)
                run_id = root.rsplit("/", 1)[-1]
                rows.append(
                    {
                        "run_id": run_id,
                        "status": str(meta.get("pipeline_status") or "unknown"),
                        "source": str(meta.get("source_stage_status") or "-"),
                        "score": str(meta.get("score_stage_status") or "-"),
                        "rows_after_dedup": int(meta.get("rows_after_dedup") or 0),
                        "updated_at_unix": _latest_run_timestamp(meta),
                        "updated_at": _format_unix_utc(_latest_run_timestamp(meta)),
                    }
                )
            rows.sort(key=lambda item: float(item.get("updated_at_unix") or 0), reverse=True)
            rows = rows[: max(1, min(limit, 100))]
            if not rows:
                _reply("No pipeline runs found in bucket.")
                return {"ok": True, "event_id": event_id, "action": action, "rows": 0}

            lines = [f"Runs (latest {len(rows)}):"]
            for item in rows:
                lines.append(
                    f"- {item['run_id']} | status={item['status']} | source={item['source']} | "
                    f"score={item['score']} | rows_after_dedup={item['rows_after_dedup']} | updated={item['updated_at']}"
                )
            lines.append("")
            lines.append("Commands:")
            lines.append("- `# JD-Run <run_id>`")
            lines.append("- `# JD-Retry <run_id>`")
            _reply("\n".join(lines))
            return {"ok": True, "event_id": event_id, "action": action, "rows": len(rows)}

        if action == "show_run":
            run_id = str(command.get("run_id") or "").strip()
            if not run_id:
                _reply("Usage: `# JD-Run <run_id>`")
                return {"ok": False, "event_id": event_id, "error": "missing_run_id"}
            artifact_keys = build_run_artifact_keys(run_id)
            meta = _safe_read_run_meta(bucket=bucket, root=artifact_keys["root"])
            if not meta:
                _reply(f"Run not found in bucket: `{run_id}`")
                return {"ok": False, "event_id": event_id, "error": "run_not_found", "run_id": run_id}
            expires_at = _format_unix_utc(meta.get("expires_after_unix"))
            lines = [
                f"Run `{run_id}`",
                f"- status: {meta.get('pipeline_status') or 'unknown'}",
                f"- source: {meta.get('source_stage_status') or '-'}",
                f"- score: {meta.get('score_stage_status') or '-'}",
                f"- rows_after_dedup: {int(meta.get('rows_after_dedup') or 0)}",
                f"- expires_at: {expires_at}",
                f"- sheet: {meta.get('google_sheet_url') or '(not available)'}",
                "",
                f"Retry score: `# JD-Retry {run_id}`",
            ]
            _reply("\n".join(lines))
            return {"ok": True, "event_id": event_id, "action": action, "run_id": run_id}

        if action == "retry_score":
            run_id = str(command.get("run_id") or "").strip()
            if not run_id:
                _reply("Usage: `# JD-Retry <run_id>`")
                return {"ok": False, "event_id": event_id, "error": "missing_run_id"}
            artifact_keys = build_run_artifact_keys(run_id)
            meta = _safe_read_run_meta(bucket=bucket, root=artifact_keys["root"])
            if not meta:
                _reply(f"Cannot retry: run not found in bucket: `{run_id}`")
                return {"ok": False, "event_id": event_id, "error": "run_not_found", "run_id": run_id}

            missing = [key for key in required_handoff_keys(artifact_keys) if not bucket.exists(key)]
            if missing:
                _reply(
                    "Cannot retry score; required handoff artifacts are missing:\n"
                    + "\n".join(f"- `{key}`" for key in missing)
                )
                return {"ok": False, "event_id": event_id, "error": "missing_handoff_artifacts", "run_id": run_id}

            payload = {
                "run_id": run_id,
                "channel_id": str(meta.get("channel_id") or channel_id),
                "jd_name": str(meta.get("jd_name") or ""),
                "jd_test_mode": bool(meta.get("jd_test_mode") or False),
                "event_id": str(meta.get("event_id") or event_id or ""),
                "artifact_root_key": artifact_keys["root"],
            }
            job = enqueue_jd_score_job(payload)
            _best_effort_update_meta(
                bucket=bucket,
                artifact_keys=artifact_keys,
                updates={
                    "pipeline_status": "score_requeued",
                    "score_requeued_at_unix": time.time(),
                    "score_requeued_by_user": user_id,
                    "score_requeued_job_id": job.id,
                },
            )
            _reply(f"Requeued score stage for run `{run_id}`.\nJob id: `{job.id}`")
            return {"ok": True, "event_id": event_id, "action": action, "run_id": run_id, "job_id": job.id}

        if action == "cleanup_runs":
            hours = float(command.get("hours") or 24.0)
            dry_run = bool(command.get("dry_run", True))
            now_unix = time.time()
            roots = _discover_run_roots(bucket=bucket, runs_prefix=runs_prefix)
            eligible: List[Dict[str, Any]] = []
            total_deleted = 0

            for root in roots:
                meta = _safe_read_run_meta(bucket=bucket, root=root)
                if not _eligible_for_cleanup(meta=meta, now_unix=now_unix, older_than_hours=hours):
                    continue
                run_id = root.rsplit("/", 1)[-1]
                status = str(meta.get("pipeline_status") or "unknown")
                if dry_run:
                    eligible.append({"run_id": run_id, "status": status, "root": root})
                    continue
                deleted = bucket.delete_prefix(root)
                total_deleted += deleted
                eligible.append({"run_id": run_id, "status": status, "root": root, "deleted": deleted})

            if dry_run:
                lines = [f"Cleanup dry-run (older than {hours:g}h): eligible runs={len(eligible)}"]
                for item in eligible[:20]:
                    lines.append(f"- {item['run_id']} | status={item['status']}")
                lines.append("")
                lines.append(f"To execute cleanup, send: `# JD-Cleanup {hours:g} confirm`")
                _reply("\n".join(lines))
                return {"ok": True, "event_id": event_id, "action": action, "eligible_runs": len(eligible), "dry_run": True}

            _reply(
                f"Cleanup complete for runs older than {hours:g}h.\n"
                f"Deleted runs: {len(eligible)}\nDeleted objects: {total_deleted}"
            )
            return {
                "ok": True,
                "event_id": event_id,
                "action": action,
                "eligible_runs": len(eligible),
                "deleted_objects": total_deleted,
                "dry_run": False,
            }

        _reply(
            "Unknown admin command. Supported:\n"
            "- `# JD-Runs [limit]`\n"
            "- `# JD-Run <run_id>`\n"
            "- `# JD-Retry <run_id>`\n"
            "- `# JD-Cleanup [hours]` (dry-run)\n"
            "- `# JD-Cleanup [hours] confirm`"
        )
        return {"ok": False, "event_id": event_id, "error": "unknown_action", "action": action}
    except Exception as exc:
        _reply(f"Admin command failed: {exc}")
        return {"ok": False, "event_id": event_id, "error": str(exc), "action": action}


def process_thread_reply_enrichment_job(
    *,
    channel_id: str,
    thread_ts: str,
    reply_ts: Optional[str],
    threshold: float,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()
    _ = reply_ts  # reserved for compatibility with existing enqueued payloads

    slack_token = os.getenv("SLACK_BOT_TOKEN") or require_env("SLACK_USER_TOKEN")
    verbose_updates = _is_truthy(os.getenv("THREAD_ENRICHMENT_VERBOSE_UPDATES"), default=False)
    post_thread_reply_update(
        slack_token=slack_token,
        channel_id=channel_id,
        thread_ts=thread_ts,
        text=f"Received threshold `{threshold:g}`. Starting enrichment workflow...",
    )

    try:
        result = run_thread_reply_enrichment_pipeline(
            slack_token=slack_token,
            channel_id=channel_id,
            thread_ts=thread_ts,
            threshold=threshold,
            post_updates=verbose_updates,
        )
    except Exception as exc:
        _notify_thread_failure(
            channel_id=channel_id,
            thread_ts=thread_ts,
            error_msg=str(exc),
            event_id=event_id,
        )
        raise

    if result.get("ignored") == "not_result_message_thread":
        post_thread_reply_update(
            slack_token=slack_token,
            channel_id=channel_id,
            thread_ts=thread_ts,
            text=(
                "Ignored: this thread is not a pipeline result message thread.\n"
                "Reply with a threshold in a thread where the root message is the scored result."
            ),
        )
        return {"ok": True, "event_id": event_id, "result": result}

    lead_skipped = result.get("lead_skipped") or []
    dashboard_warning = ""
    try:
        log_enrichment_dashboard_row(
            jd_name=str(result.get("jd_name") or ""),
            candidate_sheet_url=str(result.get("source_url") or ""),
            minimum_score_for_contact=threshold,
            candidates_entered_enrichment=int(result.get("rows_meeting_threshold") or 0),
            emails_found_salesql=int(result.get("salesql_emails_found") or 0),
            emails_passed_reoon=int(result.get("reoon_passed") or 0),
            emails_passed_bounceban=int(result.get("bounceban_deliverable") or 0),
            pre_exist_in_instantly=len(lead_skipped),
            net_leads_enrolled_instantly=int(result.get("leads_added") or 0),
            instantly_campaign_name=str(result.get("campaign_name") or ""),
            instantly_campaign_id=str(result.get("campaign_id") or ""),
            notes=str(result.get("note") or ""),
        )
        if not (os.getenv("DASHBOARD_GOOGLE_SHEET_URL") or "").strip():
            dashboard_warning = "Dashboard logging skipped: DASHBOARD_GOOGLE_SHEET_URL is not set."
    except Exception as exc:
        print(f"[warn] dashboard enrichment logging failed: {exc}")
        dashboard_warning = f"Dashboard logging failed: {exc}"

    lead_errors = result.get("lead_errors") or []
    campaign_id = result.get("campaign_id")
    campaign_name = result.get("campaign_name")
    campaign_analytics_url = ""
    source_name = result.get("source_name") or "N/A"
    source_url = result.get("source_url") or ""
    source_url_line = f"Source URL: {source_url}\n" if source_url else ""
    if campaign_id:
        campaign_analytics_url = f"https://app.instantly.ai/app/campaign/{campaign_id}/analytics"
        summary_text = (
            f"Thread enrichment complete.\n"
            f"Source: {source_name}\n"
            f"{source_url_line}"
            f"JD Name: {result.get('jd_name') or 'N/A'}\n"
            f"Threshold: {threshold:g}\n"
            f"Rows with score+LinkedIn parsed: {result.get('rows_with_score_and_linkedin')}\n"
            f"Candidates meeting threshold: {result.get('rows_meeting_threshold')}\n"
            f"SaleSQL emails: {result.get('salesql_emails_found')}\n"
            f"Reoon passed: {result.get('reoon_passed')}\n"
            f"BounceBan deliverable: {result.get('bounceban_deliverable')}\n"
            f"Instantly campaign name: {campaign_name or 'N/A'}\n"
            f"Instantly campaign: {campaign_analytics_url}\n"
            f"Leads added: {result.get('leads_added')}\n"
            f"Leads skipped: {len(lead_skipped)}\n"
            f"Lead add errors: {len(lead_errors)}"
        )
        if lead_skipped:
            summary_text = f"{summary_text}\nSkip sample: {lead_skipped[0]}"
    else:
        summary_text = (
            f"Thread enrichment completed with no campaign created.\n"
            f"Source: {source_name}\n"
            f"{source_url_line}"
            f"JD Name: {result.get('jd_name') or 'N/A'}\n"
            f"Threshold: {threshold:g}\n"
            f"Rows with score+LinkedIn parsed: {result.get('rows_with_score_and_linkedin')}\n"
            f"Candidates meeting threshold: {result.get('rows_meeting_threshold')}\n"
            f"SaleSQL emails: {result.get('salesql_emails_found')}\n"
            f"Reoon passed: {result.get('reoon_passed')}\n"
            f"BounceBan deliverable: {result.get('bounceban_deliverable')}"
        )
        note = (result.get("note") or "").strip()
        if note:
            summary_text = f"{summary_text}\nNote: {note}"
    if dashboard_warning:
        summary_text = f"{summary_text}\nDashboard: {dashboard_warning}"
    summary_blocks = _build_enrichment_summary_blocks(
        result=result,
        threshold=threshold,
        source_name=str(source_name),
        source_url=str(source_url),
        campaign_name=str(campaign_name or ""),
        campaign_analytics_url=campaign_analytics_url,
        lead_skipped=lead_skipped,
        lead_errors=lead_errors,
        dashboard_warning=dashboard_warning,
    )
    post_thread_reply_update(
        slack_token=slack_token,
        channel_id=channel_id,
        thread_ts=thread_ts,
        text=summary_text,
        blocks=summary_blocks,
        unfurl_links=False,
        unfurl_media=False,
    )
    return {"ok": True, "event_id": event_id, "result": result}


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ worker for JD source/score and reply-enrichment queues")
    parser.add_argument("--burst", action="store_true", help="Exit after current queue is drained")
    parser.add_argument(
        "--with-scheduler",
        action="store_true",
        help="Enable RQ scheduler. Disabled by default to reduce Redis command volume.",
    )
    parser.add_argument(
        "--queue-type",
        choices=["source", "score", "jd", "reply", "admin", "all"],
        default=os.getenv("RQ_WORKER_QUEUE_TYPE", "source"),
        help="Queue set to consume: source, score, jd(legacy source alias), reply, admin, or all (default: source).",
    )
    return parser.parse_args()


def get_worker_queue_names(queue_type: str) -> List[str]:
    if queue_type == "source":
        return [get_jd_source_queue_name()]
    if queue_type == "score":
        return [get_jd_score_queue_name()]
    if queue_type == "jd":
        # Legacy alias kept for existing Railway services/scripts.
        return [get_jd_queue_name()]
    if queue_type == "reply":
        return [get_reply_queue_name()]
    if queue_type == "admin":
        return [get_admin_queue_name()]
    return [get_jd_source_queue_name(), get_jd_score_queue_name(), get_reply_queue_name(), get_admin_queue_name()]


def attach_fail_fast_exception_handler(worker: Worker) -> None:
    stop_on_failure = _is_truthy(os.getenv("RQ_STOP_ON_JOB_FAILURE"), default=True)
    if not stop_on_failure:
        return

    def _stop_worker_after_failure(job, exc_type, exc_value, traceback):  # type: ignore[no-untyped-def]
        job_id = getattr(job, "id", "unknown")
        print(
            f"[fatal] job {job_id} failed with {getattr(exc_type, '__name__', 'Exception')}: {exc_value}. "
            "Stopping worker before dequeuing the next job."
        )
        try:
            worker.request_stop(signal.SIGTERM, None)
        except Exception as stop_exc:
            print(f"[warn] failed to request worker stop after job failure: {stop_exc}")
        return True

    worker.push_exc_handler(_stop_worker_after_failure)


def main() -> None:
    load_dotenv()
    args = parse_worker_args()

    conn = get_redis_connection()
    queue_names = get_worker_queue_names(args.queue_type)
    worker = Worker(queue_names, connection=conn)
    attach_fail_fast_exception_handler(worker)
    with_scheduler = args.with_scheduler or _is_truthy(os.getenv("RQ_WITH_SCHEDULER"), default=False)
    worker.work(with_scheduler=with_scheduler, burst=args.burst)


if __name__ == "__main__":
    main()
