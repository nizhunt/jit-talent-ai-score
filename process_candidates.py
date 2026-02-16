import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

try:
    from exa_py import Exa
except ImportError:
    Exa = None


CHANNEL_ID_DEFAULT = "C0AF5RGPMEW"
SLACK_HISTORY_URL = "https://slack.com/api/conversations.history"
SLACK_UPLOAD_URL = "https://slack.com/api/files.upload"
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
SLACK_GET_UPLOAD_URL_EXTERNAL = "https://slack.com/api/files.getUploadURLExternal"
SLACK_COMPLETE_UPLOAD_EXTERNAL = "https://slack.com/api/files.completeUploadExternal"

STAGES = [
    "fetch_jd",
    "generate_queries",
    "exa_search",
    "csv",
    "dedup",
    "score",
    "post",
]

BASE_CSV_FIRST_COLUMNS = ["name", "linkedin", "location", "text"]
SCORED_CSV_FIRST_COLUMNS = ["ai-score", "ai-reason"]

QUERY_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "exa_query_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 15,
                    "maxItems": 15,
                }
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
    },
}

SCORE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "candidate_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Suitability score from 0 to 10",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reasoning for the score",
                },
            },
            "required": ["score", "reason"],
            "additionalProperties": False,
        },
    },
}


class PipelineStop(Exception):
    pass


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found in environment variables.")
    return value


def debug_pause(debug: bool, stage: str) -> None:
    if not debug:
        return

    print(f"\\n[debug] ready for stage '{stage}'. Type 'continue' to run it, or 'quit' to stop.")
    while True:
        command = input("debug> ").strip().lower()
        if command in {"continue", "c"}:
            return
        if command in {"quit", "q", "exit"}:
            raise PipelineStop("Stopped by user in debug mode.")
        print("Please type 'continue' or 'quit'.")


def maybe_stop(stop_after: Optional[str], stage: str) -> None:
    if stop_after and stop_after == stage:
        raise PipelineStop(f"Stopped after stage '{stage}' via --stop-after.")


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def fetch_latest_jd_from_slack(
    slack_token: str,
    channel_id: str,
    max_messages: int = 500,
) -> Tuple[str, Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {slack_token}"}
    cursor: Optional[str] = None
    fetched = 0

    while True:
        params: Dict[str, Any] = {
            "channel": channel_id,
            "limit": min(200, max_messages - fetched),
            "inclusive": True,
        }
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            SLACK_HISTORY_URL,
            headers=headers,
            params=params,
            timeout=60,
        )
        response.raise_for_status()

        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Slack conversations.history failed: {payload.get('error')}")

        messages = payload.get("messages", [])
        fetched += len(messages)

        for message in messages:
            text = (message.get("text") or "").strip()
            if not text:
                continue

            lines = text.splitlines()
            if not lines:
                continue

            if lines[0].strip().lower() == "# jd":
                jd_text = "\\n".join(lines[1:]).strip()
                if jd_text:
                    return jd_text, message

        metadata = payload.get("response_metadata") or {}
        cursor = metadata.get("next_cursor")
        if not payload.get("has_more") or not cursor or fetched >= max_messages:
            break

    raise RuntimeError(
        "No JD message found. Expected a message in the channel with first line exactly '# JD'."
    )


def generate_exa_queries(
    client: OpenAI,
    jd_text: str,
    exa_query_prompt_template: str,
    model: str,
) -> Tuple[List[str], Dict[str, int]]:
    prompt = (
        f"{exa_query_prompt_template.strip()}\\n\\n"
        "Job description:\\n"
        f"{jd_text.strip()}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Generate high-quality, diverse Exa search prompts as valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        response_format=QUERY_RESPONSE_SCHEMA,
    )

    raw = (response.choices[0].message.content or "").strip()
    data = json.loads(raw)
    queries = [q.strip() for q in data["queries"] if q and q.strip()]

    if len(queries) != 15:
        raise RuntimeError(f"Expected 15 queries, got {len(queries)}")

    usage = extract_usage_tokens(response)
    return queries, usage


def save_debug_queries(debug: bool, debug_dir: str, queries: List[str]) -> Optional[str]:
    if not debug:
        return None

    os.makedirs(debug_dir, exist_ok=True)
    filename = datetime.now().strftime("exa-queries-%Y%m%d-%H%M%S.txt")
    path = os.path.join(debug_dir, filename)

    lines = []
    for idx, query in enumerate(queries, start=1):
        lines.append(f"{idx}. {query}")
        lines.append("")

    write_text_file(path, "\\n".join(lines).strip() + "\\n")
    return path


def write_queries_log_file(output_dir: str, queries: List[str]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = datetime.now().strftime("exa-queries-%Y%m%d-%H%M%S.txt")
    path = os.path.join(output_dir, filename)

    lines: List[str] = []
    for idx, query in enumerate(queries, start=1):
        lines.append(f"{idx}. {query}")
        lines.append("")

    write_text_file(path, "\\n".join(lines).strip() + "\\n")
    return path


def to_plain_object(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_plain_object(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_object(v) for v in value]
    if isinstance(value, tuple):
        return [to_plain_object(v) for v in value]
    if hasattr(value, "model_dump"):
        return to_plain_object(value.model_dump())
    if hasattr(value, "dict"):
        return to_plain_object(value.dict())
    if hasattr(value, "__dict__"):
        return to_plain_object(dict(value.__dict__))
    return value


def exa_search_one(exa: Any, query: str, num_results: int = 100) -> Dict[str, Any]:
    result = exa.search(
        query,
        category="people",
        num_results=num_results,
        type="auto",
        contents={
            "text": {
                "verbosity": "full",
            }
        },
    )

    result = to_plain_object(result)
    if isinstance(result, dict):
        return result

    raise RuntimeError("Unexpected Exa response type; could not convert to dict.")


def run_exa_fanout(
    exa_api_key: str,
    queries: List[str],
    num_results_per_query: int = 100,
) -> Dict[str, Any]:
    if Exa is None:
        raise RuntimeError(
            "exa_py is not installed. Install it with: pip install exa-py"
        )

    exa = Exa(exa_api_key)
    all_items: List[Dict[str, Any]] = []
    query_summaries: List[Dict[str, Any]] = []

    for idx, query in enumerate(queries, start=1):
        result = exa_search_one(
            exa=exa,
            query=query,
            num_results=num_results_per_query,
        )
        raw_items = result.get("results", [])
        items: List[Dict[str, Any]] = []
        for raw_item in raw_items if isinstance(raw_items, list) else []:
            item = to_plain_object(raw_item)
            if isinstance(item, dict):
                items.append(item)

        for item in items:
            item["_source_query_index"] = idx
            item["_source_query"] = query
            all_items.append(item)

        query_summaries.append(
            {
                "query_index": idx,
                "query": query,
                "result_count": len(items),
            }
        )
        print(f"Exa query {idx}/15: received {len(items)} results")

    return {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "query_summaries": query_summaries,
        "total_results": len(all_items),
        "results": all_items,
    }


def normalize_whitespace(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\\s+", " ", str(value)).strip()


def normalize_url(value: Any) -> str:
    url = normalize_whitespace(value)
    if not url:
        return ""

    try:
        parts = urlsplit(url)
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/")
        if not netloc:
            return url.lower().rstrip("/")
        return f"{netloc}{path}".lower()
    except Exception:
        return url.lower().rstrip("/")


def pick_linkedin(result: Dict[str, Any]) -> str:
    direct_keys = ["linkedin", "linkedinUrl", "linkedin_url", "profileUrl", "profile_url"]
    for key in direct_keys:
        value = normalize_whitespace(result.get(key))
        if "linkedin.com" in value.lower():
            return value

    url = normalize_whitespace(result.get("url"))
    if "linkedin.com" in url.lower():
        return url

    socials = result.get("socials")
    if isinstance(socials, list):
        for item in socials:
            if isinstance(item, dict):
                val = normalize_whitespace(item.get("url") or item.get("value"))
            else:
                val = normalize_whitespace(item)
            if "linkedin.com" in val.lower():
                return val

    return ""


def flatten_exa_results_to_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in results:
        highlights = item.get("highlights")
        highlights_text = ""
        if isinstance(highlights, list):
            highlights_text = " | ".join(
                normalize_whitespace(x if not isinstance(x, dict) else x.get("text"))
                for x in highlights
            ).strip(" |")

        text = normalize_whitespace(item.get("text"))
        if not text:
            text = highlights_text

        row = {
            "source_query_index": item.get("_source_query_index"),
            "source_query": item.get("_source_query"),
            "id": normalize_whitespace(item.get("id")),
            "name": normalize_whitespace(item.get("name") or item.get("author")),
            "title": normalize_whitespace(item.get("title")),
            "company": normalize_whitespace(item.get("company")),
            "location": normalize_whitespace(item.get("location")),
            "linkedin": pick_linkedin(item),
            "url": normalize_whitespace(item.get("url")),
            "published_date": normalize_whitespace(item.get("publishedDate")),
            "score": item.get("score"),
            "highlights": highlights_text,
            "text": text,
            "raw_json": json.dumps(item, ensure_ascii=False),
        }
        rows.append(row)

    return rows


def write_consolidated_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def reorder_columns(df: pd.DataFrame, first_columns: List[str]) -> pd.DataFrame:
    ordered_first = [col for col in first_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered_first]
    return df[ordered_first + remaining]


def write_results_csv(path: str, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = reorder_columns(df, BASE_CSV_FIRST_COLUMNS)
    df.to_csv(path, index=False)
    return df


def deduplicate_csv(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if df.empty:
        df.to_csv(output_csv, index=False)
        return df

    empty_text_hash = hashlib.md5("".encode("utf-8")).hexdigest()

    df["_linkedin_norm"] = df["linkedin"].apply(normalize_url) if "linkedin" in df.columns else ""
    df["_url_norm"] = df["url"].apply(normalize_url) if "url" in df.columns else ""
    df["_name_norm"] = df["name"].apply(lambda x: normalize_whitespace(x).lower()) if "name" in df.columns else ""
    df["_location_norm"] = (
        df["location"].apply(lambda x: normalize_whitespace(x).lower()) if "location" in df.columns else ""
    )
    text_series = df["text"].apply(normalize_whitespace) if "text" in df.columns else ""
    if not isinstance(text_series, pd.Series):
        text_series = pd.Series([""] * len(df))

    df["_text_hash"] = text_series.apply(lambda x: hashlib.md5(x.encode("utf-8")).hexdigest())

    dedup_keys: List[str] = []
    for i, row in df.iterrows():
        linkedin_norm = row.get("_linkedin_norm", "")
        url_norm = row.get("_url_norm", "")
        text_hash = row.get("_text_hash", "")
        name_norm = row.get("_name_norm", "")
        location_norm = row.get("_location_norm", "")

        if linkedin_norm:
            key = f"li:{linkedin_norm}"
        elif url_norm:
            key = f"url:{url_norm}"
        elif text_hash and text_hash != empty_text_hash:
            key = f"txt:{text_hash}"
        elif name_norm:
            key = f"name:{name_norm}|loc:{location_norm}"
        else:
            key = f"row:{i}"
        dedup_keys.append(key)

    df["_dedup_key"] = dedup_keys
    dedup_df = df.drop_duplicates(subset=["_dedup_key"], keep="first").copy()

    cols_to_drop = [
        "_linkedin_norm",
        "_url_norm",
        "_name_norm",
        "_location_norm",
        "_text_hash",
        "_dedup_key",
    ]
    dedup_df.drop(columns=[c for c in cols_to_drop if c in dedup_df.columns], inplace=True)
    dedup_df = reorder_columns(dedup_df, BASE_CSV_FIRST_COLUMNS)
    dedup_df.to_csv(output_csv, index=False)
    return dedup_df


def get_ai_score(
    client: OpenAI,
    candidate_text: str,
    jd_text: str,
    scorer_prompt_template: str,
    model: str,
) -> Tuple[Any, str, Dict[str, int]]:
    if not candidate_text or pd.isna(candidate_text):
        return 0, "No candidate data available", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    scorer_prompt = scorer_prompt_template.replace("[PASTE LINKEDIN DATA]", str(candidate_text))
    scorer_prompt = scorer_prompt.replace("[PASTE JD TEXT]", jd_text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert recruiter. Evaluate the candidate against the "
                        "job description. Return JSON with keys 'score' and 'reason'."
                    ),
                },
                {"role": "user", "content": scorer_prompt},
            ],
            temperature=0.0,
            response_format=SCORE_RESPONSE_SCHEMA,
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return int(data["score"]), data["reason"], extract_usage_tokens(response)
    except Exception as exc:
        print(f"  [warn] scoring error: {exc}")
        return "Error", str(exc), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def extract_usage_tokens(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def score_candidates_csv(
    client: OpenAI,
    input_csv: str,
    output_csv: str,
    jd_text: str,
    scorer_prompt_template: str,
    model: str,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    notify_every: int = 50,
    initial_seconds_per_candidate: float = 2.5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(input_csv)
    if "ai-score" not in df.columns:
        df["ai-score"] = None
    if "ai-reason" not in df.columns:
        df["ai-reason"] = None

    pending = df["ai-score"].isna()
    pending_count = int(pending.sum())
    print(f"Scoring {pending_count} candidates...")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "start",
                "pending_count": pending_count,
                "eta_seconds": pending_count * max(0.1, initial_seconds_per_candidate),
                "seconds_per_candidate": max(0.1, initial_seconds_per_candidate),
            }
        )

    progress = tqdm(total=pending_count, desc="Scoring", unit="candidate")
    started_at = time.monotonic()
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for idx in df.index:
        if not pending[idx]:
            continue

        text = normalize_whitespace(df.at[idx, "text"]) if "text" in df.columns else ""
        if not text and "raw_json" in df.columns:
            text = normalize_whitespace(df.at[idx, "raw_json"])

        score, reason, usage = get_ai_score(
            client=client,
            candidate_text=text,
            jd_text=jd_text,
            scorer_prompt_template=scorer_prompt_template,
            model=model,
        )
        usage_totals["input_tokens"] += int(usage.get("input_tokens", 0))
        usage_totals["output_tokens"] += int(usage.get("output_tokens", 0))
        usage_totals["total_tokens"] += int(usage.get("total_tokens", 0))

        df.at[idx, "ai-score"] = score
        df.at[idx, "ai-reason"] = reason
        progress.update(1)

        processed = int(progress.n)
        if processed > 0 and progress_callback is not None:
            elapsed = time.monotonic() - started_at
            seconds_per_candidate = elapsed / processed
            remaining = max(0, pending_count - processed)
            eta_seconds = remaining * seconds_per_candidate
            if (
                processed == 1
                or (notify_every > 0 and (processed % notify_every) == 0)
                or processed == pending_count
            ):
                progress_callback(
                    {
                        "event": "progress",
                        "processed": processed,
                        "pending_count": pending_count,
                        "seconds_per_candidate": seconds_per_candidate,
                        "eta_seconds": eta_seconds,
                    }
                )

        if (progress.n % 10) == 0:
            reorder_columns(df, SCORED_CSV_FIRST_COLUMNS).to_csv(output_csv, index=False)

    progress.close()
    df = reorder_columns(df, SCORED_CSV_FIRST_COLUMNS)
    df.to_csv(output_csv, index=False)
    return df, usage_totals


def compute_cost_summary(
    *,
    exa_request_count: int,
    exa_pages_count: int,
    openai_input_tokens: int,
    openai_output_tokens: int,
    scored_count: int,
    used_assumptions: bool,
) -> Dict[str, Any]:
    exa_search_per_1000 = float(os.getenv("EXA_SEARCH_COST_PER_1000_REQUESTS", "25"))
    exa_text_per_1000 = float(os.getenv("EXA_TEXT_COST_PER_1000_PAGES", "1"))
    openai_input_per_1m = float(os.getenv("OPENAI_INPUT_COST_PER_1M_TOKENS", "0.15"))
    openai_output_per_1m = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", "0.60"))

    exa_search_cost = (exa_request_count / 1000.0) * exa_search_per_1000
    exa_text_cost = (exa_pages_count / 1000.0) * exa_text_per_1000
    exa_total = exa_search_cost + exa_text_cost

    openai_input_cost = (openai_input_tokens / 1_000_000.0) * openai_input_per_1m
    openai_output_cost = (openai_output_tokens / 1_000_000.0) * openai_output_per_1m
    openai_total = openai_input_cost + openai_output_cost

    total_cost = exa_total + openai_total
    divisor = max(1, scored_count)
    per_scored = total_cost / divisor

    return {
        "exa_total": exa_total,
        "openai_total": openai_total,
        "total_cost": total_cost,
        "per_scored_candidate": per_scored,
        "used_assumptions": used_assumptions,
    }


def upload_file_to_slack(
    slack_token: str,
    channel_id: str,
    file_path: str,
    initial_comment: str,
    content_type: str,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {slack_token}"}
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Step 1: ask Slack for an external upload URL.
    get_upload_resp = requests.post(
        SLACK_GET_UPLOAD_URL_EXTERNAL,
        headers=headers,
        data={
            "filename": filename,
            "length": str(file_size),
        },
        timeout=60,
    )
    get_upload_resp.raise_for_status()
    get_upload_payload = get_upload_resp.json()
    if not get_upload_payload.get("ok"):
        raise RuntimeError(f"Slack getUploadURLExternal failed: {get_upload_payload.get('error')}")

    upload_url = get_upload_payload.get("upload_url")
    file_id = get_upload_payload.get("file_id")
    if not upload_url or not file_id:
        raise RuntimeError("Slack getUploadURLExternal returned missing upload_url or file_id.")

    # Step 2: upload file bytes to the pre-signed URL.
    with open(file_path, "rb") as f:
        upload_resp = requests.post(
            upload_url,
            files={"file": (filename, f, content_type)},
            timeout=120,
        )
    upload_resp.raise_for_status()

    # Step 3: finalize upload and share in channel.
    complete_resp = requests.post(
        SLACK_COMPLETE_UPLOAD_EXTERNAL,
        headers=headers,
        data={
            "files": json.dumps([{"id": file_id, "title": filename}]),
            "channel_id": channel_id,
            "initial_comment": initial_comment,
        },
        timeout=60,
    )
    complete_resp.raise_for_status()
    complete_payload = complete_resp.json()
    if not complete_payload.get("ok"):
        raise RuntimeError(f"Slack completeUploadExternal failed: {complete_payload.get('error')}")

    # Keep backward compatibility with callsites expecting payload['file'].
    if "file" not in complete_payload:
        files = complete_payload.get("files")
        if isinstance(files, list) and files:
            complete_payload["file"] = files[0]

    return complete_payload


def upload_csv_to_slack(
    slack_token: str,
    channel_id: str,
    csv_path: str,
    initial_comment: str,
) -> Dict[str, Any]:
    return upload_file_to_slack(
        slack_token=slack_token,
        channel_id=channel_id,
        file_path=csv_path,
        initial_comment=initial_comment,
        content_type="text/csv",
    )


def format_eta_seconds(eta_seconds: float) -> str:
    seconds = max(0, int(round(eta_seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def post_slack_message(
    slack_token: str,
    channel_id: str,
    text: str,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "channel": channel_id,
        "text": text,
    }
    response = requests.post(
        SLACK_POST_MESSAGE_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Slack chat.postMessage failed: {body.get('error')}")
    return body


def post_pipeline_update(slack_token: str, channel_id: str, text: str) -> None:
    try:
        post_slack_message(
            slack_token=slack_token,
            channel_id=channel_id,
            text=text,
        )
    except Exception as exc:
        print(f"[warn] failed to post progress update to Slack: {exc}")


def run_pipeline_from_jd_text(
    *,
    jd_text: str,
    jd_message_ts: Optional[str],
    args: argparse.Namespace,
    client: OpenAI,
    exa_api_key: str,
    slack_token: str,
) -> Dict[str, Any]:
    exa_query_prompt_template = read_text_file(args.exa_query_prompt_path)
    scorer_prompt_template = read_text_file(args.scorer_prompt_path)

    write_text_file(args.jd_path, jd_text)
    print(f"Wrote JD text to {args.jd_path}")

    debug_pause(args.debug, "generate_queries")
    queries, query_generation_usage = generate_exa_queries(
        client=client,
        jd_text=jd_text,
        exa_query_prompt_template=exa_query_prompt_template,
        model=args.model,
    )
    debug_log_path = save_debug_queries(args.debug, args.debug_dir, queries)
    if debug_log_path:
        print(f"Debug query log saved: {debug_log_path}")
    logs_dir = args.debug_dir if args.debug else os.path.join(os.path.dirname(args.jd_path), "logs")
    queries_log_path = write_queries_log_file(logs_dir, queries)
    try:
        upload_file_to_slack(
            slack_token=slack_token,
            channel_id=args.channel_id,
            file_path=queries_log_path,
            initial_comment=f"{len(queries)} Exa prompts generated for this JD.",
            content_type="text/plain",
        )
    except Exception as exc:
        print(f"[warn] failed to upload Exa prompt log to Slack: {exc}")
    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=f"{len(queries)} Exa prompts created.",
    )
    maybe_stop(args.stop_after, "generate_queries")

    debug_pause(args.debug, "exa_search")
    consolidated = run_exa_fanout(
        exa_api_key=exa_api_key,
        queries=queries,
        num_results_per_query=args.num_results_per_query,
    )
    print(f"Total Exa results fetched: {consolidated.get('total_results')}")
    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=(
            f"{int(consolidated.get('total_results') or 0)} candidates fetched "
            f"from Exa across {len(queries)} prompts."
        ),
    )
    maybe_stop(args.stop_after, "exa_search")

    debug_pause(args.debug, "csv")
    write_consolidated_json(args.results_json, consolidated)
    rows = flatten_exa_results_to_rows(consolidated.get("results", []))
    csv_df = write_results_csv(args.candidates_csv, rows)
    print(f"Wrote {len(csv_df)} rows to {args.candidates_csv}")
    maybe_stop(args.stop_after, "csv")

    debug_pause(args.debug, "dedup")
    dedup_df = deduplicate_csv(args.candidates_csv, args.dedup_csv)
    print(f"Deduplicated: {len(csv_df)} -> {len(dedup_df)} rows. Output: {args.dedup_csv}")
    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=f"{len(dedup_df)} candidates after deduplication (from {len(csv_df)}).",
    )
    maybe_stop(args.stop_after, "dedup")

    debug_pause(args.debug, "score")
    notify_every = max(1, int(os.getenv("SCORE_PROGRESS_NOTIFY_EVERY", "50")))
    initial_spc = max(0.1, float(os.getenv("SCORE_INITIAL_SECONDS_PER_CANDIDATE", "2.5")))

    def on_score_progress(payload: Dict[str, Any]) -> None:
        event_type = payload.get("event")
        if event_type == "start":
            post_pipeline_update(
                slack_token=slack_token,
                channel_id=args.channel_id,
                text=(
                    "Evaluation started. "
                    f"ETA: {format_eta_seconds(float(payload.get('eta_seconds') or 0))} "
                    f"(~{float(payload.get('seconds_per_candidate') or 0):.2f}s/candidate)."
                ),
            )
            return

        if event_type == "progress":
            processed = int(payload.get("processed") or 0)
            total = int(payload.get("pending_count") or 0)
            pct = (processed / total * 100.0) if total else 100.0
            post_pipeline_update(
                slack_token=slack_token,
                channel_id=args.channel_id,
                text=(
                    f"Scoring: {pct:.0f}% ({processed}/{total}). "
                    f"ETA: {format_eta_seconds(float(payload.get('eta_seconds') or 0))} "
                    f"({float(payload.get('seconds_per_candidate') or 0):.2f}s/candidate)."
                ),
            )

    scored_df, scoring_usage = score_candidates_csv(
        client=client,
        input_csv=args.dedup_csv,
        output_csv=args.scored_csv,
        jd_text=jd_text,
        scorer_prompt_template=scorer_prompt_template,
        model=args.model,
        progress_callback=on_score_progress,
        notify_every=notify_every,
        initial_seconds_per_candidate=initial_spc,
    )
    print(f"Scored rows: {len(scored_df)}. Output: {args.scored_csv}")
    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=f"AI scoring completed for {len(scored_df)} candidates.",
    )
    maybe_stop(args.stop_after, "score")

    debug_pause(args.debug, "post")
    slack_upload = upload_csv_to_slack(
        slack_token=slack_token,
        channel_id=args.channel_id,
        csv_path=args.scored_csv,
        initial_comment="Here is the AI-scored candidates CSV for this JD.",
    )
    file_id = (slack_upload.get("file") or {}).get("id")
    print(f"Uploaded scored CSV to Slack channel {args.channel_id}. file_id={file_id}")

    total_input_tokens = int(query_generation_usage.get("input_tokens", 0)) + int(scoring_usage.get("input_tokens", 0))
    total_output_tokens = int(query_generation_usage.get("output_tokens", 0)) + int(scoring_usage.get("output_tokens", 0))
    used_assumptions = False
    if total_input_tokens == 0 and total_output_tokens == 0:
        used_assumptions = True
        estimated_candidates = max(1, len(scored_df))
        total_input_tokens = 2000 + (estimated_candidates * 2000)
        total_output_tokens = 3500 + (estimated_candidates * 120)

    cost_summary = compute_cost_summary(
        exa_request_count=len(queries),
        exa_pages_count=int(consolidated.get("total_results") or 0),
        openai_input_tokens=total_input_tokens,
        openai_output_tokens=total_output_tokens,
        scored_count=len(scored_df),
        used_assumptions=used_assumptions,
    )

    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=(
            f"Estimated cost per scored candidate: ${cost_summary['per_scored_candidate']:.4f} "
            f"(Exa + OpenAI{', using assumptions' if cost_summary['used_assumptions'] else ', using actual usage'})."
        ),
    )
    maybe_stop(args.stop_after, "post")

    return {
        "queries_count": len(queries),
        "total_results": int(consolidated.get("total_results") or 0),
        "rows_before_dedup": int(len(csv_df)),
        "rows_after_dedup": int(len(dedup_df)),
        "rows_scored": int(len(scored_df)),
        "uploaded_file_id": file_id,
        "debug_queries_log": debug_log_path,
        "cost_summary": cost_summary,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JIT Talent AI scoring pipeline")
    parser.add_argument("--channel-id", default=CHANNEL_ID_DEFAULT)
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and step-by-step prompt")
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-messages", type=int, default=500)

    parser.add_argument("--jd-path", default="jd.md")
    parser.add_argument("--exa-query-prompt-path", default="prompts/exa-querry-creator-prompt.md")
    parser.add_argument("--scorer-prompt-path", default="prompts/scorer_prompt.md")

    parser.add_argument("--results-json", default="exa-results.json")
    parser.add_argument("--candidates-csv", default="candidates.csv")
    parser.add_argument("--dedup-csv", default="candidates-dedup.csv")
    parser.add_argument("--scored-csv", default="candidates-scored.csv")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--num-results-per-query", type=int, default=100)
    return parser.parse_args(argv)


def main() -> None:
    load_dotenv()
    args = parse_args()

    openai_api_key = require_env("OPENAI_API_KEY")
    exa_api_key = require_env("EXA_API_KEY")
    slack_token = os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_USER_TOKEN")
    if not slack_token:
        raise ValueError("SLACK_BOT_TOKEN or SLACK_USER_TOKEN not found in environment variables.")

    client = OpenAI(api_key=openai_api_key)

    try:
        debug_pause(args.debug, "fetch_jd")
        jd_text, jd_message = fetch_latest_jd_from_slack(
            slack_token=slack_token,
            channel_id=args.channel_id,
            max_messages=args.max_messages,
        )
        print(f"Fetched JD from Slack message ts={jd_message.get('ts')}")
        maybe_stop(args.stop_after, "fetch_jd")

        run_pipeline_from_jd_text(
            jd_text=jd_text,
            jd_message_ts=jd_message.get("ts"),
            args=args,
            client=client,
            exa_api_key=exa_api_key,
            slack_token=slack_token,
        )

    except PipelineStop as stop_exc:
        print(str(stop_exc))


if __name__ == "__main__":
    main()
