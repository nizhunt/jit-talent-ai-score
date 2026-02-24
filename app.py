import hashlib
import hmac
import os
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
    enqueue_jd_pipeline_job,
    mark_event_seen,
    require_env,
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


def is_jd_message(text: str) -> bool:
    lines = text.splitlines()
    if not lines:
        return False
    return lines[0].strip().lower() == "# jd"


def extract_jd_text(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[1:]).strip()


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
    expected_channel_id = os.getenv("SLACK_CHANNEL_ID", CHANNEL_ID_DEFAULT)
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
    if channel_id != expected_channel_id:
        return JSONResponse({"ok": True, "ignored": "wrong_channel"})

    text = event.get("text", "") or ""
    if not is_jd_message(text):
        return JSONResponse({"ok": True, "ignored": "not_jd_format"})

    jd_text = extract_jd_text(text)
    if not jd_text:
        return JSONResponse({"ok": True, "ignored": "empty_jd"})

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

    message_ts = event.get("ts")
    try:
        job = enqueue_jd_pipeline_job(
            {
                "jd_text": jd_text,
                "message_ts": message_ts,
                "channel_id": channel_id,
                "event_id": event_id,
            }
        )
    except QueueingUnavailableError as exc:
        if event_marked_seen:
            clear_event_seen(event_id)
        return _queue_unavailable_response(stage="enqueue", detail=str(exc))
    except Exception as exc:
        if event_marked_seen:
            clear_event_seen(event_id)
        raise HTTPException(status_code=503, detail=f"queue_enqueue_failed: {exc}") from exc

    return JSONResponse({"ok": True, "status": "queued", "job_id": job.id})
