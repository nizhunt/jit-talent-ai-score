import hashlib
import hmac
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from queueing import (
    QueueingUnavailableError,
    clear_event_seen,
    enqueue_jd_admin_job,
    enqueue_jd_pipeline_job,
    enqueue_thread_reply_enrichment_job,
    get_queue,
    get_reply_queue_name,
    mark_event_seen,
    require_env,
)
from thread_reply_enrichment import (
    parse_threshold_and_target_from_text,
    parse_threshold_from_text,
    post_thread_reply_update,
)


load_dotenv()

app = FastAPI(title="JIT Talent Slack Webhook")
CHANNEL_ID_DEFAULT = "C0AF5RGPMEW"
PROJECT_ROOT = Path(__file__).resolve().parent
FLOW_UI_ROOT = PROJECT_ROOT / "ui"

# ---------------------------------------------------------------------------
# Flow routes – all flow logic lives in ui/flow/flow_routes.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(FLOW_UI_ROOT / "flow"))
from flow_routes import router as flow_router  # noqa: E402

app.include_router(flow_router)

if FLOW_UI_ROOT.exists():
    app.mount("/ui", StaticFiles(directory=str(FLOW_UI_ROOT)), name="ui")


# ---------------------------------------------------------------------------
# Helpers (Slack-specific, kept in app.py)
# ---------------------------------------------------------------------------


def _is_truthy(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _queue_unavailable_response(stage: str, detail: str) -> JSONResponse:
    payload = {
        "ok": True,
        "status": "degraded",
        "ignored": "queue_unavailable",
        "stage": stage,
        "detail": detail,
    }
    # Ack Slack by default to avoid retry storms while Redis is degraded/out of quota.
    ack_queue_unavailable = _is_truthy(os.getenv("SLACK_ACK_QUEUE_UNAVAILABLE"), default=True)
    if ack_queue_unavailable:
        return JSONResponse(payload, status_code=200)
    return JSONResponse(payload, status_code=503)


def extract_jd_text(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[1:]).strip()


def parse_jd_message(text: str) -> Optional[Dict[str, str]]:
    lines = text.splitlines()
    if not lines:
        return None

    header = lines[0].strip()
    quick_match = re.match(
        r"(?i)^#\s*jd(?:\s*-\s*|\s+)(?:quick|test)\b(?:\s*[:\-|]\s*|\s+)?(.*)$",
        header,
    )
    if quick_match:
        jd_name = (quick_match.group(1) or "").strip()
        jd_text = extract_jd_text(text)
        return {"jd_name": jd_name, "jd_text": jd_text, "jd_test_mode": "true"}

    match = re.match(r"(?i)^#\s*jd\b(?:\s*[:\-|]\s*|\s+)?(.*)$", header)
    if match is None:
        return None

    jd_name = (match.group(1) or "").strip()
    jd_text = extract_jd_text(text)
    return {"jd_name": jd_name, "jd_text": jd_text, "jd_test_mode": "false"}


def parse_admin_command(text: str) -> Optional[Dict[str, Any]]:
    lines = text.splitlines()
    if not lines:
        return None
    header = lines[0].strip()

    list_match = re.match(r"(?i)^#\s*jd(?:\s*-\s*|\s+)runs(?:\s+(\d+))?\s*$", header)
    if list_match:
        limit_raw = list_match.group(1)
        limit = int(limit_raw) if limit_raw else 20
        limit = max(1, min(limit, 100))
        return {"action": "list_runs", "limit": limit}

    run_match = re.match(r"(?i)^#\s*jd(?:\s*-\s*|\s+)run\s+([A-Za-z0-9_-]+)\s*$", header)
    if run_match:
        return {"action": "show_run", "run_id": run_match.group(1)}

    retry_match = re.match(r"(?i)^#\s*jd(?:\s*-\s*|\s+)retry\s+([A-Za-z0-9_-]+)\s*$", header)
    if retry_match:
        return {"action": "retry_score", "run_id": retry_match.group(1)}

    cleanup_match = re.match(
        r"(?i)^#\s*jd(?:\s*-\s*|\s+)cleanup(?:\s+([0-9]+(?:\.[0-9]+)?))?(?:\s+(confirm))?\s*$",
        header,
    )
    if cleanup_match:
        hours_raw = cleanup_match.group(1)
        hours = float(hours_raw) if hours_raw else 24.0
        if hours <= 0:
            hours = 24.0
        confirm_token = (cleanup_match.group(2) or "").strip().lower()
        dry_run = confirm_token != "confirm"
        return {"action": "cleanup_runs", "hours": hours, "dry_run": dry_run}

    sync_analytics_match = re.match(
        r"(?i)^#\s*jd(?:\s*-\s*|\s+)sync(?:\s*-\s*|\s+)analytics\s*$",
        header,
    )
    if sync_analytics_match:
        return {"action": "sync_instantly_analytics"}

    return None


def verify_slack_signature(signing_secret: str, timestamp: str, signature: str, body: bytes) -> bool:
    if not timestamp or not signature:
        return False

    try:
        request_ts = int(timestamp)
    except ValueError:
        return False

    if abs(time.time() - request_ts) > 60 * 5:
        return False

    base = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    expected = "v0=" + hmac.new(signing_secret.encode("utf-8"), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "jit-talent-slack-events", "mode": "enqueue"})


@app.get("/healthz")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.post("/slack/events")
async def slack_events(request: Request):
    primary_channel_id = os.getenv("SLACK_CHANNEL_ID", CHANNEL_ID_DEFAULT)
    extra_channel_ids_raw = os.getenv("SLACK_EXTRA_CHANNEL_IDS", "")
    allowed_channel_ids = {primary_channel_id}
    for cid in extra_channel_ids_raw.split(","):
        cid = cid.strip()
        if cid:
            allowed_channel_ids.add(cid)

    body = await request.body()

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid_json: {exc}") from exc

    request_type = payload.get("type")
    if request_type == "url_verification":
        return PlainTextResponse(payload.get("challenge", ""))

    signing_secret = require_env("SLACK_SIGNING_SECRET")
    signature = request.headers.get("X-Slack-Signature", "")
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    if not verify_slack_signature(signing_secret, timestamp, signature, body):
        raise HTTPException(status_code=401, detail="invalid_signature")

    if request_type != "event_callback":
        return JSONResponse({"ok": True, "ignored": "unsupported_type"})

    event_id = payload.get("event_id", "")
    if not event_id:
        return JSONResponse({"ok": True, "ignored": "missing_event_id"})

    event: Dict = payload.get("event") or {}
    if event.get("type") != "message":
        return JSONResponse({"ok": True, "ignored": "non_message"})

    if event.get("subtype") is not None or event.get("bot_id"):
        return JSONResponse({"ok": True, "ignored": "non_user_message"})

    channel_id = event.get("channel", "")
    if channel_id not in allowed_channel_ids:
        return JSONResponse({"ok": True, "ignored": "wrong_channel"})

    text = event.get("text", "") or ""
    message_ts = event.get("ts")
    thread_ts = event.get("thread_ts")
    is_thread_reply = bool(thread_ts and message_ts and thread_ts != message_ts)

    workflow_name: str
    enqueue_fn: Any
    job_payload: Dict[str, Any]

    admin_command = parse_admin_command(text)
    if admin_command is not None:
        workflow_name = "jd_admin"
        enqueue_fn = enqueue_jd_admin_job
        job_payload = {
            "channel_id": channel_id,
            "message_ts": message_ts,
            "user_id": event.get("user", ""),
            "event_id": event_id,
            "command": admin_command,
        }
    else:
        jd_payload = parse_jd_message(text)
        if jd_payload is not None:
            jd_text = jd_payload["jd_text"]
            jd_name = jd_payload.get("jd_name", "")
            jd_test_mode = str(jd_payload.get("jd_test_mode", "false")).strip().lower() == "true"
            if not jd_text:
                return JSONResponse({"ok": True, "ignored": "empty_jd"})
            workflow_name = "jd_pipeline"
            enqueue_fn = enqueue_jd_pipeline_job
            job_payload = {
                "jd_text": jd_text,
                "jd_name": jd_name,
                "jd_test_mode": jd_test_mode,
                "message_ts": message_ts,
                "channel_id": channel_id,
                "event_id": event_id,
            }
        elif is_thread_reply:
            parsed = parse_threshold_and_target_from_text(text)
            if parsed is None:
                return JSONResponse({"ok": True, "ignored": "thread_reply_missing_threshold"})
            threshold = parsed["threshold"]
            target = parsed["target"]
            workflow_name = f"thread_reply_{target}"
            enqueue_fn = enqueue_thread_reply_enrichment_job
            job_payload = {
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "reply_ts": message_ts,
                "threshold": threshold,
                "target": target,
                "event_id": event_id,
            }
            calendly_url = parsed.get("calendly_url")
            if calendly_url:
                job_payload["calendly_url"] = calendly_url
        else:
            return JSONResponse({"ok": True, "ignored": "not_supported_message"})

    dedup_enabled = _is_truthy(os.getenv("SLACK_EVENT_DEDUP_ENABLED"), default=True)
    event_marked_seen = False
    if dedup_enabled:
        # Cross-instance idempotency: only first webhook delivery for an event id is accepted.
        try:
            is_first_delivery = mark_event_seen(event_id)
        except QueueingUnavailableError as exc:
            return _queue_unavailable_response(stage="dedup", detail=str(exc))

        if not is_first_delivery:
            return JSONResponse({"ok": True, "ignored": "duplicate_event"})
        event_marked_seen = True

    try:
        job = enqueue_fn(job_payload)
    except QueueingUnavailableError as exc:
        if event_marked_seen:
            clear_event_seen(event_id)
        return _queue_unavailable_response(stage="enqueue", detail=str(exc))
    except Exception as exc:
        if event_marked_seen:
            clear_event_seen(event_id)
        raise HTTPException(status_code=503, detail=f"queue_enqueue_failed: {exc}") from exc

    # If an enrichment job is queued behind other jobs, notify the user of its position.
    if workflow_name.startswith("thread_reply_") and thread_ts:
        try:
            position = get_queue(get_reply_queue_name()).count
            if position > 1:
                slack_token = os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_USER_TOKEN") or ""
                if slack_token:
                    post_thread_reply_update(
                        slack_token=slack_token,
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        text=f"Your request `{text.strip()}` is queued at position {position}. You'll be notified when processing starts.",
                    )
        except Exception:
            pass  # best-effort; don't fail the webhook response

    return JSONResponse({"ok": True, "status": "queued", "workflow": workflow_name, "job_id": job.id})
