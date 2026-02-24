"""
Flow definition CRUD logic and API routes.

All flow-related helpers, validation, file I/O, and endpoints live here.
Flow JSON files (flow-current.json, flow-proposed.json, flow-definition.json)
are stored alongside this module in ui/flow/.
"""

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

FLOW_MODULE_DIR = Path(__file__).resolve().parent          # ui/flow/
FLOW_UI_ROOT = FLOW_MODULE_DIR.parent                      # ui/

FLOW_LEGACY_FILE_NAME = "flow-definition.json"
FLOW_CURRENT_FILE_NAME = "flow-current.json"
FLOW_PROPOSED_FILE_NAME = "flow-proposed.json"

FLOW_SLOT_CURRENT = "current"
FLOW_SLOT_PROPOSED = "proposed"
FLOW_VALID_SLOTS = {FLOW_SLOT_CURRENT, FLOW_SLOT_PROPOSED}

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_flow_definition() -> Dict[str, Any]:
    return {
        "version": 2,
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
                "id": "notify_start",
                "type": "action",
                "label": "Notify Slack: Processing Started",
                "description": (
                    "Posts 'Thanks for sharing the JD. Processing has started.' "
                    "to the Slack channel before the pipeline begins."
                ),
                "status": "current",
                "x": 1820,
                "y": 320,
            },
            {
                "id": "widen_jd",
                "type": "action",
                "label": "Widen JD (6 Meta Prompts)",
                "description": (
                    "BATCH LOOP START — For each of the 6 JD-widening prompts "
                    "(original, wider-location, wider-titles, wider-yoe, "
                    "wider-companies, lenient-skills): applies the meta prompt "
                    "to produce a structured search profile."
                ),
                "status": "current",
                "x": 1820,
                "y": 160,
            },
            {
                "id": "generate_queries",
                "type": "action",
                "label": "Generate 10 Exa Queries (per batch)",
                "description": (
                    "Inside the batch loop: OpenAI generates 10 Exa search queries "
                    "from the widened profile. Runs once per widening prompt "
                    "(6 × 10 = 60 queries total)."
                ),
                "status": "current",
                "x": 2120,
                "y": 160,
            },
            {
                "id": "exa_search",
                "type": "action",
                "label": "Run Exa Fanout Search (per batch)",
                "description": (
                    "Inside the batch loop: fetches up to 100 candidate profiles "
                    "per query from Exa. Each batch searches 10 queries "
                    "(up to 1000 results per batch)."
                ),
                "status": "current",
                "x": 2420,
                "y": 160,
            },
            {
                "id": "build_csv",
                "type": "data",
                "label": "Build Batch CSV",
                "description": (
                    "Inside the batch loop: flattens Exa results and writes a "
                    "per-batch CSV to the exa-batches/ directory. BATCH LOOP END "
                    "— repeats from Widen JD for next prompt."
                ),
                "status": "current",
                "x": 2700,
                "y": 160,
            },
            {
                "id": "combine_batches",
                "type": "data",
                "label": "Combine Batch CSVs",
                "description": (
                    "After all 6 batches complete: concatenates all per-batch CSVs "
                    "into the final candidates.csv. Frees batch data from memory."
                ),
                "status": "current",
                "x": 2970,
                "y": 160,
            },
            {
                "id": "dedup_candidates",
                "type": "action",
                "label": "Deduplicate Candidates",
                "description": (
                    "Dedupes by normalized LinkedIn URL, profile URL, "
                    "text hash, or name+location signals."
                ),
                "status": "current",
                "x": 3240,
                "y": 160,
            },
            {
                "id": "score_candidates",
                "type": "action",
                "label": "Score Candidates with AI",
                "description": (
                    "Applies scoring prompt to each candidate against the original JD. "
                    "Posts progress updates to Slack at 25%, 50%, 75%."
                ),
                "status": "current",
                "x": 3510,
                "y": 160,
            },
            {
                "id": "write_sheet_csv",
                "type": "data",
                "label": "Write Sheet-Ready CSV",
                "description": (
                    "Drops the raw_json column from the scored DataFrame "
                    "and writes a clean sheet-ready CSV for sharing."
                ),
                "status": "current",
                "x": 3780,
                "y": 160,
            },
            {
                "id": "upload_scored_csv",
                "type": "output",
                "label": "Upload Scored CSV to Slack",
                "description": (
                    "Uploads the sheet-ready CSV file to the Slack channel with "
                    "a summary comment (rows scored, dedup stats, cost, time)."
                ),
                "status": "current",
                "x": 4050,
                "y": 160,
            },
            {
                "id": "post_cost_summary",
                "type": "output",
                "label": "Post Cost Summary",
                "description": (
                    "Computes and returns estimated per-candidate cost "
                    "based on Exa and OpenAI usage."
                ),
                "status": "current",
                "x": 4050,
                "y": 360,
            },
        ],
        "edges": [
            {"id": "e1", "source": "slack_event", "target": "verify_signature", "label": "incoming event"},
            {"id": "e2", "source": "verify_signature", "target": "validate_message", "label": "valid signature"},
            {"id": "e3", "source": "validate_message", "target": "deduplicate_event", "label": "valid JD event"},
            {"id": "e4", "source": "deduplicate_event", "target": "enqueue_job", "label": "first delivery"},
            {"id": "e5", "source": "enqueue_job", "target": "worker_pickup", "label": "rq queue"},
            {"id": "e5b", "source": "worker_pickup", "target": "notify_start", "label": "start worker"},
            {"id": "e6", "source": "notify_start", "target": "widen_jd", "label": "ack sent"},
            {"id": "e6b", "source": "widen_jd", "target": "generate_queries", "label": "structured profile"},
            {"id": "e7", "source": "generate_queries", "target": "exa_search", "label": "10 queries"},
            {"id": "e8", "source": "exa_search", "target": "build_csv", "label": "~1000 results"},
            {"id": "e8b", "source": "build_csv", "target": "widen_jd", "label": "next batch (loop ×6)"},
            {"id": "e9", "source": "build_csv", "target": "combine_batches", "label": "all batches done"},
            {"id": "e10", "source": "combine_batches", "target": "dedup_candidates", "label": "candidates.csv"},
            {"id": "e11", "source": "dedup_candidates", "target": "score_candidates", "label": "dedup csv"},
            {"id": "e12", "source": "score_candidates", "target": "write_sheet_csv", "label": "scored csv"},
            {"id": "e13", "source": "write_sheet_csv", "target": "upload_scored_csv", "label": "sheet-ready csv"},
            {"id": "e14", "source": "score_candidates", "target": "post_cost_summary", "label": "token usage + costs"},
        ],
        "notes": [
            {
                "id": "n1",
                "text": "Edit flow-proposed.json while planning, then sync to flow-current.json after implementation.",
            },
            {
                "id": "n2",
                "text": (
                    "Steps widen_jd → generate_queries → exa_search → build_csv run in a batch loop, "
                    "once per widening prompt (×6). After all batches, results are combined."
                ),
            },
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
    candidates.append(FLOW_MODULE_DIR / FLOW_LEGACY_FILE_NAME)
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
    candidates.append(FLOW_MODULE_DIR / file_name)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/flow")
async def flow_editor() -> FileResponse:
    index_path = FLOW_MODULE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="flow_ui_not_found")
    return FileResponse(index_path)


@router.get("/api/flow")
async def get_flow(slot: str = Query(default=FLOW_SLOT_PROPOSED)) -> JSONResponse:
    try:
        normalized_slot = _normalize_flow_slot(slot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    flow, source = _load_flow_definition(normalized_slot)
    return JSONResponse({"ok": True, "slot": normalized_slot, "flow": flow, "source": source})


@router.put("/api/flow")
async def save_flow(payload: Dict[str, Any] = Body(...), slot: str = Query(default=FLOW_SLOT_PROPOSED)) -> JSONResponse:
    if not isinstance(payload, dict) or "flow" not in payload:
        raise HTTPException(status_code=400, detail='expected body: {"flow": {...}}')

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


@router.post("/api/flow/reset")
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


@router.post("/api/flow/copy-current-to-proposed")
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


@router.post("/api/flow/promote-proposed-to-current")
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
