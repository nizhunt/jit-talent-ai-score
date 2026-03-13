import os
from typing import Dict, List


DEFAULT_RUNS_PREFIX = "runs"

# Stable handoff artifact names for cross-job source -> score transfer.
ARTIFACT_JD_TEXT = "jd.txt"
ARTIFACT_DEDUP_CSV = "dedup.csv"
ARTIFACT_JD_CONTEXT_BY_PROMPT_JSON = "jd_context_by_prompt.json"
ARTIFACT_QUERIES_JSON = "queries.json"
ARTIFACT_META_JSON = "meta.json"
ARTIFACT_FINAL_SUMMARY_POSTED_JSON = "final_summary_posted.json"


def get_runs_prefix() -> str:
    raw = (os.getenv("PIPELINE_RUNS_PREFIX") or DEFAULT_RUNS_PREFIX).strip().strip("/")
    return raw or DEFAULT_RUNS_PREFIX


def build_run_artifact_keys(run_id: str, runs_prefix: str = "") -> Dict[str, str]:
    normalized_run_id = str(run_id or "").strip().strip("/")
    if not normalized_run_id:
        raise ValueError("run_id is required to build artifact keys.")

    prefix = (runs_prefix or get_runs_prefix()).strip().strip("/")
    root = f"{prefix}/{normalized_run_id}" if prefix else normalized_run_id
    return {
        "root": root,
        "jd_text": f"{root}/{ARTIFACT_JD_TEXT}",
        "dedup_csv": f"{root}/{ARTIFACT_DEDUP_CSV}",
        "jd_context_by_prompt": f"{root}/{ARTIFACT_JD_CONTEXT_BY_PROMPT_JSON}",
        "queries": f"{root}/{ARTIFACT_QUERIES_JSON}",
        "meta": f"{root}/{ARTIFACT_META_JSON}",
        "final_summary_posted": f"{root}/{ARTIFACT_FINAL_SUMMARY_POSTED_JSON}",
    }


def required_handoff_keys(artifact_keys: Dict[str, str]) -> List[str]:
    return [
        artifact_keys["jd_text"],
        artifact_keys["dedup_csv"],
        artifact_keys["jd_context_by_prompt"],
        artifact_keys["queries"],
        artifact_keys["meta"],
    ]
