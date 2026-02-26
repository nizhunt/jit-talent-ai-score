import argparse
import os
import shutil
import uuid
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from rq import Worker

from process_candidates import parse_args, post_slack_message, run_pipeline_from_jd_text
from queueing import clear_event_seen, get_queue_name, get_redis_connection
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


def build_pipeline_args(channel_id: str) -> argparse.Namespace:
    args = parse_args([])
    args.channel_id = channel_id
    args.debug = False
    args.stop_after = None

    project_root = os.path.dirname(os.path.abspath(__file__))
    args.exa_query_prompt_path = os.path.join(project_root, "prompts", "exa-querry-creator-prompt.md")
    args.scorer_prompt_path = os.path.join(project_root, "prompts", "scorer_prompt.md")
    args.jd_widening_prompts_dir = os.path.join(project_root, "prompts", "jd-widening-prompts")

    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join("/tmp", f"jit-talent-worker-{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

    args.jd_path = os.path.join(run_dir, "jd.md")
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


def _notify_failure(channel_id: str, error_msg: str, event_id: Optional[str] = None) -> None:
    """Best-effort: post failure to Slack and clear event dedup so JD can be retried."""
    try:
        slack_token = os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_USER_TOKEN")
        if slack_token and channel_id:
            short_error = (error_msg[:300] + "...") if len(error_msg) > 300 else error_msg
            post_slack_message(
                slack_token=slack_token,
                channel_id=channel_id,
                text=f"⚠️ Pipeline failed: {short_error}\nPlease re-post the JD to retry.",
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
        text="Thanks for sharing the JD. Processing has started.",
    )

    try:
        client = OpenAI(api_key=openai_api_key)
        args = build_pipeline_args(channel_id=channel_id)
        run_dir = getattr(args, "run_dir", None)
        result = run_pipeline_from_jd_text(
            jd_text=jd_text,
            args=args,
            client=client,
            exa_api_key=exa_api_key,
            slack_token=slack_token,
        )
        return {"ok": True, "event_id": event_id, "result": result}
    except Exception as exc:
        _notify_failure(channel_id=channel_id, error_msg=str(exc), event_id=event_id)
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
            post_updates=True,
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
                "Ignored: this thread is not a pipeline result CSV message thread.\n"
                "Reply with a threshold in a thread where the root message is the scored CSV result."
            ),
        )
        return {"ok": True, "event_id": event_id, "result": result}

    lead_errors = result.get("lead_errors") or []
    campaign_id = result.get("campaign_id")
    if campaign_id:
        summary_text = (
            f"Thread enrichment complete.\n"
            f"CSV: {result.get('csv_filename')}\n"
            f"Threshold: {threshold:g}\n"
            f"Candidates meeting threshold: {result.get('rows_meeting_threshold')}\n"
            f"SaleSQL emails: {result.get('salesql_emails_found')}\n"
            f"Reoon passed: {result.get('reoon_passed')}\n"
            f"BounceBan deliverable: {result.get('bounceban_deliverable')}\n"
            f"Instantly campaign: `{campaign_id}`\n"
            f"Leads added: {result.get('leads_added')}\n"
            f"Lead add errors: {len(lead_errors)}"
        )
    else:
        summary_text = (
            f"Thread enrichment completed with no campaign created.\n"
            f"CSV: {result.get('csv_filename')}\n"
            f"Threshold: {threshold:g}\n"
            f"Candidates meeting threshold: {result.get('rows_meeting_threshold')}\n"
            f"SaleSQL emails: {result.get('salesql_emails_found')}\n"
            f"Reoon passed: {result.get('reoon_passed')}\n"
            f"BounceBan deliverable: {result.get('bounceban_deliverable')}"
        )
    post_thread_reply_update(
        slack_token=slack_token,
        channel_id=channel_id,
        thread_ts=thread_ts,
        text=summary_text,
    )
    return {"ok": True, "event_id": event_id, "result": result}


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ worker for JD pipeline")
    parser.add_argument("--burst", action="store_true", help="Exit after current queue is drained")
    parser.add_argument(
        "--with-scheduler",
        action="store_true",
        help="Enable RQ scheduler. Disabled by default to reduce Redis command volume.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_worker_args()

    conn = get_redis_connection()
    queue_name = get_queue_name()
    worker = Worker([queue_name], connection=conn)
    with_scheduler = args.with_scheduler or _is_truthy(os.getenv("RQ_WITH_SCHEDULER"), default=False)
    worker.work(with_scheduler=with_scheduler, burst=args.burst)


if __name__ == "__main__":
    main()
