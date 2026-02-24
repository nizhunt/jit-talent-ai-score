import argparse
import gc
import glob
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
SLACK_GET_UPLOAD_URL_EXTERNAL = "https://slack.com/api/files.getUploadURLExternal"
SLACK_COMPLETE_UPLOAD_EXTERNAL = "https://slack.com/api/files.completeUploadExternal"

STAGES = ["fetch_jd", "generate_queries", "exa_search", "csv", "dedup", "score", "post"]

# Conservative concurrency limits (well under API rate caps).
EXA_CONCURRENT_SEARCHES = 5   # Exa limit: 10 QPS
SCORING_CONCURRENT_CALLS = 100  # gpt-5-mini: 30,000 RPM

BASE_CSV_FIRST_COLUMNS = [
    "first_name",
    "last_name",
    "name",
    "linkedin",
    "location",
    "title",
    "current_title",
    "current_company",
    "text",
]
SCORED_CSV_FIRST_COLUMNS = ["ai-score", "ai-reason", "ai-email"]
SHEET_COLUMN_LABEL_OVERRIDES = {
    "first_name": "First Name",
    "last_name": "Last Name",
    "name": "Full Name",
    "text": "Profile Text",
    "raw_json": "Raw JSON",
    "ai-score": "AI Score",
    "ai-reason": "AI Reason",
    "ai-email": "AI Email",
    "linkedin": "LinkedIn",
    "url": "URL",
    "id": "ID",
}

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
                    "minItems": 10,
                    "maxItems": 10,
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
                "email": {
                    "type": "string",
                    "description": "Outreach email for the candidate (blank for score 0-3)",
                },
            },
            "required": ["score", "reason", "email"],
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


def load_widening_prompts(prompts_dir: str) -> List[Tuple[str, str]]:
    """Load all JD-widening prompt files from the given directory, sorted by filename."""
    pattern = os.path.join(prompts_dir, "*.md")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No widening prompt files found in {prompts_dir}")
    prompts: List[Tuple[str, str]] = []
    for filepath in files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        text = read_text_file(filepath)
        prompts.append((name, text))
    return prompts


def widen_jd(
    client: OpenAI,
    jd_text: str,
    widening_prompt: str,
    prompt_name: str,
    model: str,
) -> Tuple[str, Dict[str, int]]:
    """Apply a single JD-widening meta prompt to produce a structured search profile."""
    prompt = widening_prompt.replace("[JD]", jd_text.strip())

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a structured data extraction assistant. Return the structured profile exactly as requested.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    profile_text = (response.choices[0].message.content or "").strip()
    if not profile_text:
        raise RuntimeError(f"Widening prompt '{prompt_name}' returned empty profile.")

    usage = extract_usage_tokens(response)
    print(f"  Widened JD with '{prompt_name}' ({len(profile_text)} chars)")
    return profile_text, usage


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
    structured_profile: str,
    exa_query_prompt_template: str,
    model: str,
    max_retries: int = 3,
) -> Tuple[List[str], Dict[str, int]]:
    prompt = exa_query_prompt_template.replace(
        "[STRUCTURED PROFILE OUTPUT FROM META PROMPT]",
        structured_profile.strip(),
    )

    last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for attempt in range(1, max_retries + 1):
        messages = [
            {
                "role": "system",
                "content": "Generate high-quality, diverse Exa search prompts as valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
        # On retries, ask the model to be concise.
        if attempt > 1:
            messages.append({
                "role": "user",
                "content": "IMPORTANT: Keep each query under 150 words. Be concise.",
            })

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=16384,
            response_format=QUERY_RESPONSE_SCHEMA,
        )

        last_usage = extract_usage_tokens(response)
        finish_reason = getattr(response.choices[0], "finish_reason", None)

        if finish_reason != "length":
            # Normal completion — parse and return.
            raw = (response.choices[0].message.content or "").strip()
            data = json.loads(raw)
            queries = [q.strip() for q in data["queries"] if q and q.strip()]
            if not queries:
                raise RuntimeError("Query generation returned zero queries.")
            if len(queries) != 10:
                print(f"  [warn] Expected 10 queries, got {len(queries)}; proceeding anyway.")
            return queries, last_usage

        # Truncated response — try to salvage partial queries.
        print(f"  [warn] Query generation truncated (attempt {attempt}/{max_retries}).")
        if attempt == max_retries:
            raw = (response.choices[0].message.content or "").strip()
            # Try to extract whatever queries were completed before truncation.
            try:
                # The JSON is incomplete, try to fix it by closing arrays/objects.
                patched = raw.rstrip()
                if not patched.endswith("]}"):
                    # Find the last complete string in the queries array.
                    last_quote = patched.rfind('"')
                    if last_quote > 0:
                        patched = patched[:last_quote + 1] + "]}"
                data = json.loads(patched)
                queries = [q.strip() for q in data.get("queries", []) if q and q.strip()]
                if queries:
                    print(f"  [warn] Salvaged {len(queries)} queries from truncated response.")
                    return queries, last_usage
            except (json.JSONDecodeError, Exception):
                pass
            raise RuntimeError(
                "OpenAI response truncated after all retries during query generation."
            )

    # Should not reach here, but satisfy type checker.
    raise RuntimeError("Query generation failed unexpectedly.")


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
    query_summaries: List[Dict[str, Any]] = [None] * len(queries)  # preserve order
    lock = threading.Lock()

    def _search_one(idx: int, query: str) -> None:
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

        with lock:
            all_items.extend(items)
            query_summaries[idx - 1] = {
                "query_index": idx,
                "query": query,
                "result_count": len(items),
            }
        print(f"Exa query {idx}/{len(queries)}: received {len(items)} results")

    with ThreadPoolExecutor(max_workers=EXA_CONCURRENT_SEARCHES) as executor:
        futures = [
            executor.submit(_search_one, idx, query)
            for idx, query in enumerate(queries, start=1)
        ]
        for future in as_completed(futures):
            future.result()  # Raise any exceptions

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


def pick_first_non_empty(source: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = normalize_whitespace(source.get(key))
        if value:
            return value
    return ""


def split_name(name: str, explicit_first_name: str = "", explicit_last_name: str = "") -> Tuple[str, str]:
    first_name = normalize_whitespace(explicit_first_name)
    last_name = normalize_whitespace(explicit_last_name)
    if first_name or last_name:
        return first_name, last_name

    normalized_name = normalize_whitespace(name)
    if not normalized_name:
        return "", ""

    parts = normalized_name.split(" ")
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def normalize_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def extract_work_history(item: Dict[str, Any]) -> List[Any]:
    for key in ["workHistory", "experiences", "experience", "work_experience", "positions"]:
        value = item.get(key)
        if isinstance(value, list):
            return value
    return []


def extract_education_history(item: Dict[str, Any]) -> List[Any]:
    for key in ["educationHistory", "education", "educations"]:
        value = item.get(key)
        if isinstance(value, list):
            return value
    return []


def flatten_work_history(row: Dict[str, Any], work_history: List[Any]) -> None:
    row["work_history_count"] = len(work_history)
    summary_parts: List[str] = []

    for idx, entry in enumerate(work_history, start=1):
        if isinstance(entry, dict):
            title = pick_first_non_empty(entry, ["title", "position", "role"])
            company = pick_first_non_empty(entry, ["company", "companyName", "organization"])
            location = pick_first_non_empty(entry, ["location", "place"])
            from_date = pick_first_non_empty(entry, ["from", "startDate", "start", "fromDate"])
            to_date = pick_first_non_empty(entry, ["to", "endDate", "end", "toDate"])
            description = pick_first_non_empty(entry, ["description", "summary"])
            company_id = pick_first_non_empty(entry, ["companyId", "company_id"])
        else:
            title = normalize_whitespace(entry)
            company = ""
            location = ""
            from_date = ""
            to_date = ""
            description = ""
            company_id = ""

        if title and company:
            summary_parts.append(f"{title} at {company}")
        elif title:
            summary_parts.append(title)
        elif company:
            summary_parts.append(company)

        row[f"work_{idx}_title"] = title
        row[f"work_{idx}_company"] = company
        row[f"work_{idx}_company_id"] = company_id
        row[f"work_{idx}_location"] = location
        row[f"work_{idx}_from"] = from_date
        row[f"work_{idx}_to"] = to_date
        row[f"work_{idx}_description"] = description

    row["work_history_summary"] = " | ".join(summary_parts)


def flatten_education_history(row: Dict[str, Any], education_history: List[Any]) -> None:
    row["education_history_count"] = len(education_history)
    summary_parts: List[str] = []

    for idx, entry in enumerate(education_history, start=1):
        if isinstance(entry, dict):
            institution = pick_first_non_empty(entry, ["institution", "school", "university", "organization"])
            degree = pick_first_non_empty(entry, ["degree", "qualification"])
            from_date = pick_first_non_empty(entry, ["from", "startDate", "start", "fromDate"])
            to_date = pick_first_non_empty(entry, ["to", "endDate", "end", "toDate"])
            institution_id = pick_first_non_empty(entry, ["institutionId", "institution_id", "schoolId"])
        else:
            institution = normalize_whitespace(entry)
            degree = ""
            from_date = ""
            to_date = ""
            institution_id = ""

        if degree and institution:
            summary_parts.append(f"{degree} at {institution}")
        elif institution:
            summary_parts.append(institution)
        elif degree:
            summary_parts.append(degree)

        row[f"edu_{idx}_institution"] = institution
        row[f"edu_{idx}_institution_id"] = institution_id
        row[f"edu_{idx}_degree"] = degree
        row[f"edu_{idx}_from"] = from_date
        row[f"edu_{idx}_to"] = to_date

    row["education_history_summary"] = " | ".join(summary_parts)


def flatten_string_list(row: Dict[str, Any], values: List[Any], prefix: str) -> None:
    cleaned: List[str] = []
    for item in values:
        if isinstance(item, dict):
            value = pick_first_non_empty(item, ["name", "skill", "text", "label", "value"])
        else:
            value = normalize_whitespace(item)
        if value:
            cleaned.append(value)

    row[f"{prefix}_count"] = len(cleaned)
    row[f"{prefix}_summary"] = " | ".join(cleaned)
    for idx, value in enumerate(cleaned, start=1):
        row[f"{prefix}_{idx}"] = value


def flatten_exa_results_to_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in results:
        highlights = normalize_list(item.get("highlights"))
        highlights_text = ""
        if highlights:
            highlights_text = " | ".join(
                normalize_whitespace(x if not isinstance(x, dict) else x.get("text"))
                for x in highlights
            ).strip(" |")

        text = normalize_whitespace(item.get("text"))
        if not text:
            text = highlights_text

        explicit_first_name = pick_first_non_empty(item, ["firstName", "first_name", "given_name"])
        explicit_last_name = pick_first_non_empty(item, ["lastName", "last_name", "family_name"])
        full_name = normalize_whitespace(item.get("name") or item.get("author"))
        first_name, last_name = split_name(
            name=full_name,
            explicit_first_name=explicit_first_name,
            explicit_last_name=explicit_last_name,
        )
        name = normalize_whitespace(" ".join(part for part in [first_name, last_name] if part)) or full_name

        work_history = extract_work_history(item)
        education_history = extract_education_history(item)
        skills = normalize_list(item.get("skills"))

        row = {
            "source_query_index": item.get("_source_query_index"),
            "source_query": item.get("_source_query"),
            "id": normalize_whitespace(item.get("id")),
            "entity_id": normalize_whitespace(item.get("entityId")),
            "entity_type": normalize_whitespace(item.get("entityType")),
            "entity_version": normalize_whitespace(item.get("entityVersion")),
            "name": name,
            "first_name": first_name,
            "last_name": last_name,
            "title": normalize_whitespace(item.get("title")),
            "current_title": pick_first_non_empty(item, ["current_title", "currentTitle", "title"]),
            "current_company": pick_first_non_empty(item, ["current_company", "currentCompany", "company"]),
            "company": normalize_whitespace(item.get("company")),
            "location": normalize_whitespace(item.get("location")),
            "author": normalize_whitespace(item.get("author")),
            "image": normalize_whitespace(item.get("image")),
            "linkedin": pick_linkedin(item),
            "url": normalize_whitespace(item.get("url")),
            "published_date": pick_first_non_empty(item, ["publishedDate", "published_date"]),
            "score": item.get("score"),
            "highlights": highlights_text,
            "highlight_scores": normalize_whitespace(item.get("highlightScores")),
            "text": text,
            "raw_json": json.dumps(item, ensure_ascii=False),
        }

        flatten_work_history(row, work_history)
        flatten_education_history(row, education_history)
        flatten_string_list(row, skills, "skill")
        flatten_string_list(row, highlights, "highlight")
        rows.append(row)

    return rows


def reorder_columns(df: pd.DataFrame, first_columns: List[str]) -> pd.DataFrame:
    ordered_first = [col for col in first_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered_first]
    return df[ordered_first + remaining]


def prettify_sheet_column_name(column: str) -> str:
    override = SHEET_COLUMN_LABEL_OVERRIDES.get(column)
    if override:
        return override

    acronyms = {
        "ai": "AI",
        "id": "ID",
        "url": "URL",
        "json": "JSON",
        "jd": "JD",
        "llm": "LLM",
    }
    tokens = [part for part in column.replace("-", "_").split("_") if part]
    if not tokens:
        return column

    label_parts: List[str] = []
    for token in tokens:
        if token.isdigit():
            label_parts.append(token)
            continue
        label_parts.append(acronyms.get(token.lower(), token.capitalize()))

    return " ".join(label_parts)


def write_sheet_ready_csv(path: str, df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: prettify_sheet_column_name(str(col)) for col in df.columns}
    sheet_df = df.rename(columns=rename_map)
    sheet_df.to_csv(path, index=False)
    return sheet_df


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
    del df  # Free the original (larger) DataFrame
    gc.collect()
    return dedup_df


def get_ai_score(
    client: OpenAI,
    candidate_text: str,
    jd_text: str,
    scorer_prompt_template: str,
    model: str,
) -> Tuple[Any, str, str, Dict[str, int]]:
    if not candidate_text or pd.isna(candidate_text):
        return 0, "No candidate data available", "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

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
                        "job description. Return JSON with keys 'score', 'reason', and 'email'."
                    ),
                },
                {"role": "user", "content": scorer_prompt},
            ],
            temperature=0.0,
            response_format=SCORE_RESPONSE_SCHEMA,
        )
        raw = (response.choices[0].message.content or "").strip()
        data = json.loads(raw)
        return int(data["score"]), data["reason"], data.get("email", ""), extract_usage_tokens(response)
    except Exception as exc:
        print(f"  [warn] scoring error: {exc}")
        return "Error", str(exc), "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


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
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(input_csv)
    if "ai-score" not in df.columns:
        df["ai-score"] = None
    if "ai-reason" not in df.columns:
        df["ai-reason"] = None
    if "ai-email" not in df.columns:
        df["ai-email"] = None

    pending = df["ai-score"].isna()
    pending_count = int(pending.sum())
    print(f"Scoring {pending_count} candidates ({SCORING_CONCURRENT_CALLS} concurrent)...")

    # Fixed milestones: notify at 25%, 50%, 75% only (3 progress updates).
    milestone_pcts = [25, 50, 75]
    milestone_thresholds = set()
    for pct in milestone_pcts:
        threshold = int(pending_count * pct / 100)
        if 0 < threshold < pending_count:
            milestone_thresholds.add(threshold)

    # Pre-extract candidate text so worker threads don't touch the DataFrame.
    pending_texts: Dict[int, str] = {}
    for idx in df.index:
        if not pending[idx]:
            continue
        text = normalize_whitespace(df.at[idx, "text"]) if "text" in df.columns else ""
        if not text and "raw_json" in df.columns:
            text = normalize_whitespace(df.at[idx, "raw_json"])
        pending_texts[idx] = text

    progress = tqdm(total=pending_count, desc="Scoring", unit="candidate")
    started_at = time.monotonic()
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    lock = threading.Lock()
    completed_count = 0

    def _score_one(idx: int, text: str) -> Tuple[int, Any, str, str, Dict[str, int]]:
        score, reason, email, usage = get_ai_score(
            client=client,
            candidate_text=text,
            jd_text=jd_text,
            scorer_prompt_template=scorer_prompt_template,
            model=model,
        )
        return idx, score, reason, email, usage

    with ThreadPoolExecutor(max_workers=SCORING_CONCURRENT_CALLS) as executor:
        futures = [
            executor.submit(_score_one, idx, text)
            for idx, text in pending_texts.items()
        ]

        for future in as_completed(futures):
            idx, score, reason, email, usage = future.result()

            with lock:
                usage_totals["input_tokens"] += int(usage.get("input_tokens", 0))
                usage_totals["output_tokens"] += int(usage.get("output_tokens", 0))
                usage_totals["total_tokens"] += int(usage.get("total_tokens", 0))

                df.at[idx, "ai-score"] = score
                df.at[idx, "ai-reason"] = reason
                df.at[idx, "ai-email"] = email
                progress.update(1)
                completed_count += 1

                processed = completed_count
                if processed > 0 and progress_callback is not None and processed in milestone_thresholds:
                    elapsed = time.monotonic() - started_at
                    seconds_per_candidate = elapsed / processed
                    remaining = max(0, pending_count - processed)
                    eta_seconds = remaining * seconds_per_candidate
                    pct = processed / pending_count * 100.0
                    progress_callback(
                        {
                            "event": "progress",
                            "processed": processed,
                            "pending_count": pending_count,
                            "seconds_per_candidate": seconds_per_candidate,
                            "eta_seconds": eta_seconds,
                            "pct": pct,
                        }
                    )

                if (completed_count % 10) == 0:
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


def upload_csv_to_slack(
    slack_token: str,
    channel_id: str,
    csv_path: str,
    initial_comment: str,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {slack_token}"}
    filename = os.path.basename(csv_path)
    file_size = os.path.getsize(csv_path)

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
    with open(csv_path, "rb") as f:
        upload_resp = requests.post(
            upload_url,
            files={"file": (filename, f, "text/csv")},
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


def build_score_tally_lines(scored_df: pd.DataFrame) -> List[str]:
    if "ai-score" not in scored_df.columns:
        return []

    numeric_scores = pd.to_numeric(scored_df["ai-score"], errors="coerce").dropna()
    if numeric_scores.empty:
        return []

    tally = numeric_scores.astype(int).value_counts().sort_index(ascending=False)
    return [f"  Score {int(score_val)}: {int(count)}" for score_val, count in tally.items()]


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
    args: argparse.Namespace,
    client: OpenAI,
    exa_api_key: str,
    slack_token: str,
) -> Dict[str, Any]:
    pipeline_start = time.monotonic()
    exa_query_prompt_template = read_text_file(args.exa_query_prompt_path)
    scorer_prompt_template = read_text_file(args.scorer_prompt_path)

    write_text_file(args.jd_path, jd_text)
    print(f"Wrote JD text to {args.jd_path}")

    # --- Stage: widen JD, generate queries, search Exa, and flatten — in batches ---
    debug_pause(args.debug, "generate_queries")

    widening_prompts = load_widening_prompts(args.jd_widening_prompts_dir)
    print(f"Loaded {len(widening_prompts)} JD-widening prompts.")

    all_queries: List[str] = []
    batch_csv_paths: List[str] = []
    total_exa_results = 0
    total_csv_rows = 0
    batch_dir = os.path.join(os.path.dirname(args.candidates_csv), "exa-batches")
    os.makedirs(batch_dir, exist_ok=True)
    total_query_gen_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for wp_idx, (wp_name, wp_text) in enumerate(widening_prompts, start=1):
        print(f"\n[{wp_idx}/{len(widening_prompts)}] Widening JD with '{wp_name}'...")
        profile, widen_usage = widen_jd(
            client=client,
            jd_text=jd_text,
            widening_prompt=wp_text,
            prompt_name=wp_name,
            model=args.model,
        )
        for key in total_query_gen_usage:
            total_query_gen_usage[key] += int(widen_usage.get(key, 0))

        queries, qgen_usage = generate_exa_queries(
            client=client,
            structured_profile=profile,
            exa_query_prompt_template=exa_query_prompt_template,
            model=args.model,
        )
        for key in total_query_gen_usage:
            total_query_gen_usage[key] += int(qgen_usage.get(key, 0))

        print(f"  Generated {len(queries)} exa queries for '{wp_name}'.")
        all_queries.extend(queries)

        # Run Exa searches for this batch, flatten, and write to a temp CSV immediately.
        batch_result = run_exa_fanout(
            exa_api_key=exa_api_key,
            queries=queries,
            num_results_per_query=args.num_results_per_query,
        )
        batch_items = batch_result.get("results", [])
        total_exa_results += len(batch_items)

        batch_rows = flatten_exa_results_to_rows(batch_items)
        batch_csv = os.path.join(batch_dir, f"batch-{wp_idx}.csv")
        pd.DataFrame(batch_rows).to_csv(batch_csv, index=False)
        batch_csv_paths.append(batch_csv)
        total_csv_rows += len(batch_rows)
        print(f"  Exa batch: {len(batch_items)} results -> {len(batch_rows)} rows (written to disk, {total_csv_rows} total)")

        # Free all batch data before the next iteration.
        del batch_result, batch_items, batch_rows
        gc.collect()

    print(f"\nTotal exa queries: {len(all_queries)}, total rows: {total_csv_rows}")

    logs_dir = args.debug_dir if args.debug else os.path.join(os.path.dirname(args.jd_path), "logs")
    queries_log_path = write_queries_log_file(logs_dir, all_queries)
    if args.debug:
        print(f"Debug query log saved: {queries_log_path}")

    maybe_stop(args.stop_after, "generate_queries")
    maybe_stop(args.stop_after, "exa_search")

    debug_pause(args.debug, "csv")
    # Combine batch CSVs into the final candidates CSV (only one DataFrame in memory).
    csv_df = pd.concat(
        [pd.read_csv(p) for p in batch_csv_paths],
        ignore_index=True,
    )
    csv_df = reorder_columns(csv_df, BASE_CSV_FIRST_COLUMNS)
    csv_df.to_csv(args.candidates_csv, index=False)
    rows_before_dedup = len(csv_df)
    del csv_df  # Free before dedup reads the same data from disk
    gc.collect()

    maybe_stop(args.stop_after, "csv")

    debug_pause(args.debug, "dedup")
    dedup_df = deduplicate_csv(args.candidates_csv, args.dedup_csv)
    rows_after_dedup = len(dedup_df)
    del dedup_df  # Free before scoring reads the dedup CSV from disk
    gc.collect()

    maybe_stop(args.stop_after, "dedup")

    debug_pause(args.debug, "score")

    def on_score_progress(payload: Dict[str, Any]) -> None:
        if payload.get("event") != "progress":
            return
        processed = int(payload.get("processed") or 0)
        total = int(payload.get("pending_count") or 0)
        pct = payload.get("pct", (processed / total * 100.0) if total else 100.0)
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
        model=args.scorer_model,
        progress_callback=on_score_progress,
    )

    maybe_stop(args.stop_after, "score")

    debug_pause(args.debug, "post")
    if args.scored_csv.lower().endswith(".csv"):
        sheet_ready_csv = f"{args.scored_csv[:-4]}-sheet-ready.csv"
    else:
        sheet_ready_csv = f"{args.scored_csv}-sheet-ready.csv"

    # Drop the heavy raw_json column before writing the sheet-ready CSV.
    if "raw_json" in scored_df.columns:
        scored_df.drop(columns=["raw_json"], inplace=True)
    write_sheet_ready_csv(sheet_ready_csv, scored_df)

    rows_scored = len(scored_df)
    total_input_tokens = int(total_query_gen_usage.get("input_tokens", 0)) + int(scoring_usage.get("input_tokens", 0))
    total_output_tokens = int(total_query_gen_usage.get("output_tokens", 0)) + int(scoring_usage.get("output_tokens", 0))
    used_assumptions = False
    if total_input_tokens == 0 and total_output_tokens == 0:
        used_assumptions = True
        estimated_candidates = max(1, rows_scored)
        total_input_tokens = 2000 + (estimated_candidates * 2000)
        total_output_tokens = 3500 + (estimated_candidates * 120)

    cost_summary = compute_cost_summary(
        exa_request_count=len(all_queries),
        exa_pages_count=total_exa_results,
        openai_input_tokens=total_input_tokens,
        openai_output_tokens=total_output_tokens,
        scored_count=rows_scored,
        used_assumptions=used_assumptions,
    )

    # Build score tally before freeing the DataFrame.
    score_tally_lines = build_score_tally_lines(scored_df)

    del scored_df  # Free before Slack upload
    gc.collect()

    elapsed = time.monotonic() - pipeline_start
    tally_block = "\n".join(score_tally_lines) if score_tally_lines else "  (no scores)"
    upload_comment = (
        f"AI-scored candidates CSV for this JD\n"
        f"Scored: {rows_scored}\n"
        f"After dedup: {rows_after_dedup} (from {rows_before_dedup} fetched)\n"
        f"Est. cost/candidate: ${cost_summary['per_scored_candidate']:.4f}\n"
        f"Total time: {format_eta_seconds(elapsed)}\n\n"
        f"Score Tally:\n{tally_block}"
    )
    slack_upload = upload_csv_to_slack(
        slack_token=slack_token,
        channel_id=args.channel_id,
        csv_path=sheet_ready_csv,
        initial_comment=upload_comment,
    )
    file_id = (slack_upload.get("file") or {}).get("id")

    maybe_stop(args.stop_after, "post")

    return {
        "queries_count": len(all_queries),
        "total_results": total_exa_results,
        "rows_before_dedup": rows_before_dedup,
        "rows_after_dedup": rows_after_dedup,
        "rows_scored": rows_scored,
        "uploaded_file_id": file_id,
        "sheet_ready_csv": sheet_ready_csv,
        "queries_log_path": queries_log_path,
        "cost_summary": cost_summary,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JIT Talent AI scoring pipeline")
    parser.add_argument("--channel-id", default=CHANNEL_ID_DEFAULT)
    parser.add_argument("--debug", action="store_true", help="Enable debug logs and step-by-step prompt")
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--model", default="gpt-5-mini", help="Model for JD widening and query generation")
    parser.add_argument("--scorer-model", default="gpt-5-mini", help="Model for candidate scoring (supports structured output)")
    parser.add_argument("--max-messages", type=int, default=500)

    parser.add_argument("--jd-path", default="jd.md")
    parser.add_argument("--exa-query-prompt-path", default="prompts/exa-querry-creator-prompt.md")
    parser.add_argument("--scorer-prompt-path", default="prompts/scorer_prompt.md")
    parser.add_argument("--jd-widening-prompts-dir", default="prompts/jd-widening-prompts")

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
            args=args,
            client=client,
            exa_api_key=exa_api_key,
            slack_token=slack_token,
        )

    except PipelineStop as stop_exc:
        print(str(stop_exc))


if __name__ == "__main__":
    main()
