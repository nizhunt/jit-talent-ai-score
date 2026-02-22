import hashlib
import hmac
import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
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
FLOW_LEGACY_FILE_NAME = "flow-definition.json"
FLOW_CURRENT_FILE_NAME = "flow-current.json"
FLOW_PROPOSED_FILE_NAME = "flow-proposed.json"
FLOW_SLOT_CURRENT = "current"
FLOW_SLOT_PROPOSED = "proposed"
FLOW_VALID_SLOTS = {FLOW_SLOT_CURRENT, FLOW_SLOT_PROPOSED}
FLOW_UI_ROOT = PROJECT_ROOT / "ui"
FLOW_UI_DIR = FLOW_UI_ROOT / "flow"

if FLOW_UI_ROOT.exists():
    app.mount("/ui", StaticFiles(directory=str(FLOW_UI_ROOT)), name="ui")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_flow_definition() -> Dict[str, Any]:
    return {
        "version": 1,
        "title": "JIT Talent AI Score Pipeline",
        "description": (
            "Visual map of the Slack -> queue -> worker pipeline. "
            "Use the proposed flow file to plan logic changes."
        ),
        "updated_at": _utc_now_iso(),
        "nodes": [
            {
                "id": "slack_event",
                "type": "trigger",
                "label": "Slack Event Received",
                "description": "POST /slack/events accepts incoming Slack payload.",
                "status": "current",
                "x": 40,
                "y": 160,
            },
            {
                "id": "verify_signature",
                "type": "decision",
                "label": "Verify Slack Signature",
                "description": "Reject invalid signature or stale timestamp.",
                "status": "current",
                "x": 340,
                "y": 160,
            },
            {
                "id": "validate_message",
                "type": "decision",
                "label": "Is #JD Message in Target Channel?",
                "description": "Filters bot messages, wrong channel, non-JD format, empty JD.",
                "status": "current",
                "x": 660,
                "y": 160,
            },
            {
                "id": "deduplicate_event",
                "type": "decision",
                "label": "Deduplicate Slack Event",
                "description": "Cross-instance dedup by event id before enqueue.",
                "status": "current",
                "x": 980,
                "y": 160,
            },
            {
                "id": "enqueue_job",
                "type": "action",
                "label": "Enqueue JD Pipeline Job",
                "description": "Pushes jd_text + metadata to Redis/RQ queue.",
                "status": "current",
                "x": 1280,
                "y": 160,
            },
            {
                "id": "worker_pickup",
                "type": "action",
                "label": "Worker Picks Up Job",
                "description": "worker.py pulls queue item and starts processing.",
                "status": "current",
                "x": 1560,
                "y": 320,
            },
            {
                "id": "generate_queries",
                "type": "action",
                "label": "Generate 15 Exa Queries",
                "description": "OpenAI generates query list from JD text.",
                "status": "current",
                "x": 1820,
                "y": 160,
            },
            {
                "id": "exa_search",
                "type": "action",
                "label": "Run Exa Fanout Search",
                "description": "Fetches candidate profiles from Exa per query.",
                "status": "current",
                "x": 2120,
                "y": 160,
            },
            {
                "id": "build_csv",
                "type": "data",
                "label": "Build Candidate CSV",
                "description": "Flatten Exa results and write candidates.csv.",
                "status": "current",
                "x": 2390,
                "y": 160,
            },
            {
                "id": "dedup_candidates",
                "type": "action",
                "label": "Deduplicate Candidates",
                "description": "Dedupes by normalized LinkedIn/name/location signals.",
                "status": "current",
                "x": 2660,
                "y": 160,
            },
            {
                "id": "score_candidates",
                "type": "action",
                "label": "Score Candidates with AI",
                "description": "Applies scoring prompt, posts progress updates to Slack.",
                "status": "current",
                "x": 2920,
                "y": 160,
            },
            {
                "id": "upload_scored_csv",
                "type": "output",
                "label": "Upload Scored CSV to Slack",
                "description": "Sends final scored file and summary updates to channel.",
                "status": "current",
                "x": 3190,
                "y": 160,
            },
            {
                "id": "post_cost_summary",
                "type": "output",
                "label": "Post Cost Summary",
                "description": "Computes and posts estimated per-candidate cost.",
                "status": "current",
                "x": 3190,
                "y": 360,
            },
        ],
        "edges": [
            {"id": "e1", "source": "slack_event", "target": "verify_signature", "label": "incoming event"},
            {"id": "e2", "source": "verify_signature", "target": "validate_message", "label": "valid signature"},
            {"id": "e3", "source": "validate_message", "target": "deduplicate_event", "label": "valid JD event"},
            {"id": "e4", "source": "deduplicate_event", "target": "enqueue_job", "label": "first delivery"},
            {"id": "e5", "source": "enqueue_job", "target": "worker_pickup", "label": "rq queue"},
            {"id": "e6", "source": "worker_pickup", "target": "generate_queries", "label": "start worker flow"},
            {"id": "e7", "source": "generate_queries", "target": "exa_search", "label": "15 queries"},
            {"id": "e8", "source": "exa_search", "target": "build_csv", "label": "results"},
            {"id": "e9", "source": "build_csv", "target": "dedup_candidates", "label": "candidates.csv"},
            {"id": "e10", "source": "dedup_candidates", "target": "score_candidates", "label": "dedup csv"},
            {"id": "e11", "source": "score_candidates", "target": "upload_scored_csv", "label": "scored csv"},
            {"id": "e12", "source": "score_candidates", "target": "post_cost_summary", "label": "token usage + costs"},
        ],
        "notes": [
            {
                "id": "n1",
                "text": "Edit flow-proposed.json while planning, then sync to flow-current.json after implementation.",
            }
        ],
    }


def _normalize_flow_slot(raw_slot: str) -> str:
    slot = (raw_slot or FLOW_SLOT_PROPOSED).strip().lower()
    if slot not in FLOW_VALID_SLOTS:
        raise ValueError(f"invalid slot: {raw_slot}")
    return slot


def _dedupe_paths(candidates: List[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen: Set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate.expanduser())
    return deduped


def _flow_slot_file_name(slot: str) -> str:
    if slot == FLOW_SLOT_CURRENT:
        return FLOW_CURRENT_FILE_NAME
    return FLOW_PROPOSED_FILE_NAME


def _flow_slot_env_var(slot: str) -> str:
    if slot == FLOW_SLOT_CURRENT:
        return "FLOW_CURRENT_STORAGE_PATH"
    return "FLOW_PROPOSED_STORAGE_PATH"


def _legacy_flow_candidates() -> List[Path]:
    candidates: List[Path] = []
    legacy_configured = os.getenv("FLOW_STORAGE_PATH")
    if legacy_configured:
        candidates.append(Path(legacy_configured))
    candidates.append(PROJECT_ROOT / FLOW_LEGACY_FILE_NAME)
    candidates.append(Path("/tmp") / FLOW_LEGACY_FILE_NAME)
    return _dedupe_paths(candidates)


def _flow_slot_write_candidates(slot: str) -> List[Path]:
    slot = _normalize_flow_slot(slot)
    candidates: List[Path] = []

    configured = os.getenv(_flow_slot_env_var(slot))
    if configured:
        candidates.append(Path(configured))

    # Backward compatibility: if only FLOW_STORAGE_PATH is set, use it as proposed storage.
    legacy_configured = os.getenv("FLOW_STORAGE_PATH")
    if slot == FLOW_SLOT_PROPOSED and legacy_configured:
        candidates.append(Path(legacy_configured))

    file_name = _flow_slot_file_name(slot)
    candidates.append(PROJECT_ROOT / file_name)
    candidates.append(Path("/tmp") / file_name)
    return _dedupe_paths(candidates)


def _flow_slot_read_candidates(slot: str) -> List[Path]:
    slot = _normalize_flow_slot(slot)
    candidates: List[Path] = []
    candidates.extend(_flow_slot_write_candidates(slot))
    if slot == FLOW_SLOT_CURRENT:
        candidates.extend(_legacy_flow_candidates())
    if slot == FLOW_SLOT_PROPOSED:
        # If proposed flow does not exist yet, bootstrap from current/legacy.
        candidates.extend(_flow_slot_write_candidates(FLOW_SLOT_CURRENT))
        candidates.extend(_legacy_flow_candidates())
    return _dedupe_paths(candidates)


def _validate_and_normalize_flow(flow: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(flow, dict):
        raise ValueError("flow must be an object")

    normalized = deepcopy(flow)
    nodes = normalized.get("nodes")
    edges = normalized.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("flow.nodes and flow.edges must be arrays")

    node_ids: Set[str] = set()
    clean_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("each node must be an object")
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            raise ValueError("each node requires a non-empty id")
        if node_id in node_ids:
            raise ValueError(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        clean_nodes.append(
            {
                "id": node_id,
                "type": str(node.get("type", "action")).strip() or "action",
                "label": str(node.get("label", node_id)).strip() or node_id,
                "description": str(node.get("description", "")).strip(),
                "status": str(node.get("status", "current")).strip() or "current",
                "x": int(node.get("x", 100)),
                "y": int(node.get("y", 100)),
            }
        )

    clean_edges: List[Dict[str, Any]] = []
    for idx, edge in enumerate(edges, start=1):
        if not isinstance(edge, dict):
            raise ValueError("each edge must be an object")
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            raise ValueError("each edge requires source and target")
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"edge references unknown node: {source} -> {target}")
        edge_id = str(edge.get("id", f"e{idx}")).strip() or f"e{idx}"
        clean_edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "label": str(edge.get("label", "")).strip(),
                "source_port": str(edge.get("source_port", "output_1")).strip() or "output_1",
                "target_port": str(edge.get("target_port", "input_1")).strip() or "input_1",
            }
        )

    notes: List[Dict[str, Any]] = []
    raw_notes = normalized.get("notes", [])
    if isinstance(raw_notes, list):
        for idx, note in enumerate(raw_notes, start=1):
            if isinstance(note, dict):
                note_id = str(note.get("id", f"n{idx}")).strip() or f"n{idx}"
                text = str(note.get("text", "")).strip()
                if text:
                    notes.append({"id": note_id, "text": text})
            elif isinstance(note, str):
                text = note.strip()
                if text:
                    notes.append({"id": f"n{idx}", "text": text})

    return {
        "version": int(normalized.get("version") or 1),
        "title": str(normalized.get("title", "Untitled Flow")).strip() or "Untitled Flow",
        "description": str(normalized.get("description", "")).strip(),
        "updated_at": _utc_now_iso(),
        "nodes": clean_nodes,
        "edges": clean_edges,
        "notes": notes,
    }


def _read_flow_from_disk(slot: str) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    for path in _flow_slot_read_candidates(slot):
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            return parsed, path
        except Exception:
            continue
    return None, None


def _write_flow_to_disk(slot: str, flow: Dict[str, Any]) -> Path:
    payload = json.dumps(flow, indent=2)
    last_error: Optional[Exception] = None
    for path in _flow_slot_write_candidates(slot):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
            return path
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"failed_to_persist_flow:{slot}: {last_error}")


def _load_flow_definition(slot: str) -> Tuple[Dict[str, Any], str]:
    slot = _normalize_flow_slot(slot)
    loaded, loaded_path = _read_flow_from_disk(slot)
    if loaded is None:
        if slot == FLOW_SLOT_PROPOSED:
            current_flow, _ = _load_flow_definition(FLOW_SLOT_CURRENT)
            proposed_from_current = deepcopy(current_flow)
            proposed_from_current["updated_at"] = _utc_now_iso()
            return proposed_from_current, "derived_from_current_default"
        default_flow = _default_flow_definition()
        return default_flow, "default"
    try:
        return _validate_and_normalize_flow(loaded), str(loaded_path)
    except ValueError:
        if slot == FLOW_SLOT_PROPOSED:
            current_flow, _ = _load_flow_definition(FLOW_SLOT_CURRENT)
            proposed_from_current = deepcopy(current_flow)
            proposed_from_current["updated_at"] = _utc_now_iso()
            return proposed_from_current, "derived_from_current_invalid_saved_flow"
        default_flow = _default_flow_definition()
        return default_flow, "default_invalid_saved_flow"


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


@app.get("/flow")
async def flow_editor() -> FileResponse:
    index_path = FLOW_UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="flow_ui_not_found")
    return FileResponse(index_path)


@app.get("/api/flow")
async def get_flow(slot: str = Query(default=FLOW_SLOT_PROPOSED)) -> JSONResponse:
    try:
        normalized_slot = _normalize_flow_slot(slot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    flow, source = _load_flow_definition(normalized_slot)
    return JSONResponse({"ok": True, "slot": normalized_slot, "flow": flow, "source": source})


@app.put("/api/flow")
async def save_flow(payload: Dict[str, Any] = Body(...), slot: str = Query(default=FLOW_SLOT_PROPOSED)) -> JSONResponse:
    if not isinstance(payload, dict) or "flow" not in payload:
        raise HTTPException(status_code=400, detail="expected body: {\"flow\": {...}}")

    try:
        normalized_slot = _normalize_flow_slot(slot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        flow = _validate_and_normalize_flow(payload["flow"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid_flow: {exc}") from exc

    try:
        saved_path = _write_flow_to_disk(normalized_slot, flow)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse({"ok": True, "slot": normalized_slot, "flow": flow, "saved_to": str(saved_path)})


@app.post("/api/flow/reset")
async def reset_flow(slot: str = Query(default=FLOW_SLOT_PROPOSED)) -> JSONResponse:
    try:
        normalized_slot = _normalize_flow_slot(slot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if normalized_slot == FLOW_SLOT_PROPOSED:
        flow, reset_from = _load_flow_definition(FLOW_SLOT_CURRENT)
        flow = deepcopy(flow)
        flow["updated_at"] = _utc_now_iso()
    else:
        flow = _default_flow_definition()
        reset_from = "default"

    try:
        saved_path = _write_flow_to_disk(normalized_slot, flow)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "ok": True,
            "slot": normalized_slot,
            "reset_from": reset_from,
            "flow": flow,
            "saved_to": str(saved_path),
        }
    )


@app.post("/api/flow/copy-current-to-proposed")
async def copy_current_to_proposed() -> JSONResponse:
    current_flow, current_source = _load_flow_definition(FLOW_SLOT_CURRENT)
    proposed_flow = deepcopy(current_flow)
    proposed_flow["updated_at"] = _utc_now_iso()
    try:
        saved_path = _write_flow_to_disk(FLOW_SLOT_PROPOSED, proposed_flow)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "ok": True,
            "copied_from": FLOW_SLOT_CURRENT,
            "source_detail": current_source,
            "saved_to": str(saved_path),
            "flow": proposed_flow,
        }
    )


@app.post("/api/flow/promote-proposed-to-current")
async def promote_proposed_to_current() -> JSONResponse:
    proposed_flow, proposed_source = _load_flow_definition(FLOW_SLOT_PROPOSED)
    promoted_flow = deepcopy(proposed_flow)
    promoted_flow["updated_at"] = _utc_now_iso()
    try:
        current_saved_path = _write_flow_to_disk(FLOW_SLOT_CURRENT, promoted_flow)
        proposed_saved_path = _write_flow_to_disk(FLOW_SLOT_PROPOSED, promoted_flow)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "ok": True,
            "promoted_from": FLOW_SLOT_PROPOSED,
            "source_detail": proposed_source,
            "current_saved_to": str(current_saved_path),
            "proposed_saved_to": str(proposed_saved_path),
            "flow": promoted_flow,
        }
    )


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
