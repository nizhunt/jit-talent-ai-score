import argparse
import os
import re
import signal
import shutil
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from rq import Worker

from dashboard_logger import log_enrichment_dashboard_row, log_jd_processing_dashboard_row
from process_candidates import parse_args, post_slack_message, run_pipeline_from_jd_text
from queueing import clear_event_seen, get_jd_queue_name, get_redis_connection, get_reply_queue_name
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

    run_id = uuid.uuid4().hex[:12]
    jd_slug = _slugify_identifier(args.jd_name, default="jd")
    run_dir = os.path.join("/tmp", f"jit-talent-worker-{jd_slug}-{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

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


def process_jd_pipeline_job(
    *,
    jd_text: str,
    message_ts: Optional[str],
    channel_id: str,
    jd_name: Optional[str] = None,
    jd_test_mode: bool = False,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()
    _ = message_ts  # reserved for compatibility with existing enqueued payloads
    run_dir: Optional[str] = None

    openai_api_key = require_env("OPENAI_API_KEY")
    exa_api_key = require_env("EXA_API_KEY")
    slack_token = os.getenv("SLACK_BOT_TOKEN") or require_env("SLACK_USER_TOKEN")

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
        args = build_pipeline_args(channel_id=channel_id, jd_name=jd_name, jd_test_mode=jd_test_mode)
        run_dir = getattr(args, "run_dir", None)
        result = run_pipeline_from_jd_text(
            jd_text=jd_text,
            args=args,
            client=client,
            exa_api_key=exa_api_key,
            slack_token=slack_token,
        )
        try:
            log_jd_processing_dashboard_row(
                jd_name=jd_name or "",
                candidate_sheet_url=str(result.get("google_sheet_url") or ""),
                total_profiles_found=int(result.get("rows_before_dedup") or 0),
                profiles_after_dedup=int(result.get("rows_after_dedup") or 0),
                score_counts_by_score=result.get("score_counts_by_score") or {},
            )
        except Exception as exc:
            print(f"[warn] dashboard JD logging failed: {exc}")
        return {
            "ok": True,
            "event_id": event_id,
            "jd_name": jd_name,
            "jd_test_mode": bool(jd_test_mode),
            "result": result,
        }
    except Exception as exc:
        _notify_failure(channel_id=channel_id, error_msg=str(exc), event_id=event_id, jd_name=jd_name)
        raise  # Re-raise so RQ marks the job as failed
    finally:
        cleanup_run_dir(run_dir)


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
    parser = argparse.ArgumentParser(description="RQ worker for JD and reply-enrichment queues")
    parser.add_argument("--burst", action="store_true", help="Exit after current queue is drained")
    parser.add_argument(
        "--with-scheduler",
        action="store_true",
        help="Enable RQ scheduler. Disabled by default to reduce Redis command volume.",
    )
    parser.add_argument(
        "--queue-type",
        choices=["jd", "reply", "all"],
        default=os.getenv("RQ_WORKER_QUEUE_TYPE", "jd"),
        help="Queue set to consume: jd, reply, or all (default: jd).",
    )
    return parser.parse_args()


def get_worker_queue_names(queue_type: str) -> List[str]:
    if queue_type == "jd":
        return [get_jd_queue_name()]
    if queue_type == "reply":
        return [get_reply_queue_name()]
    return [get_jd_queue_name(), get_reply_queue_name()]


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
