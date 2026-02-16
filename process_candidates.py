import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
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
) -> List[str]:
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

    return queries


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
) -> Tuple[Any, str]:
    if not candidate_text or pd.isna(candidate_text):
        return 0, "No candidate data available"

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
        return int(data["score"]), data["reason"]
    except Exception as exc:
        print(f"  [warn] scoring error: {exc}")
        return "Error", str(exc)


def score_candidates_csv(
    client: OpenAI,
    input_csv: str,
    output_csv: str,
    jd_text: str,
    scorer_prompt_template: str,
    model: str,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "ai-score" not in df.columns:
        df["ai-score"] = None
    if "ai-reason" not in df.columns:
        df["ai-reason"] = None

    pending = df["ai-score"].isna()
    pending_count = int(pending.sum())
    print(f"Scoring {pending_count} candidates...")

    progress = tqdm(total=pending_count, desc="Scoring", unit="candidate")

    for idx in df.index:
        if not pending[idx]:
            continue

        text = normalize_whitespace(df.at[idx, "text"]) if "text" in df.columns else ""
        if not text and "raw_json" in df.columns:
            text = normalize_whitespace(df.at[idx, "raw_json"])

        score, reason = get_ai_score(
            client=client,
            candidate_text=text,
            jd_text=jd_text,
            scorer_prompt_template=scorer_prompt_template,
            model=model,
        )

        df.at[idx, "ai-score"] = score
        df.at[idx, "ai-reason"] = reason
        progress.update(1)

        if (progress.n % 10) == 0:
            reorder_columns(df, SCORED_CSV_FIRST_COLUMNS).to_csv(output_csv, index=False)

    progress.close()
    df = reorder_columns(df, SCORED_CSV_FIRST_COLUMNS)
    df.to_csv(output_csv, index=False)
    return df


def upload_csv_to_slack(
    slack_token: str,
    channel_id: str,
    csv_path: str,
    initial_comment: str,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {slack_token}"}
    data = {
        "channels": channel_id,
        "filename": os.path.basename(csv_path),
        "title": os.path.basename(csv_path),
        "initial_comment": initial_comment,
    }

    with open(csv_path, "rb") as f:
        files = {"file": (os.path.basename(csv_path), f, "text/csv")}
        response = requests.post(
            SLACK_UPLOAD_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=120,
        )

    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(f"Slack file upload failed: {payload.get('error')}")
    return payload


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
    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=(
            "Started JD pipeline"
            + (f" for message ts={jd_message_ts}." if jd_message_ts else ".")
        ),
    )

    debug_pause(args.debug, "generate_queries")
    queries = generate_exa_queries(
        client=client,
        jd_text=jd_text,
        exa_query_prompt_template=exa_query_prompt_template,
        model=args.model,
    )
    debug_log_path = save_debug_queries(args.debug, args.debug_dir, queries)
    if debug_log_path:
        print(f"Debug query log saved: {debug_log_path}")
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
    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=f"Consolidated CSV created with {len(csv_df)} rows.",
    )
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
    scored_df = score_candidates_csv(
        client=client,
        input_csv=args.dedup_csv,
        output_csv=args.scored_csv,
        jd_text=jd_text,
        scorer_prompt_template=scorer_prompt_template,
        model=args.model,
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
        initial_comment=(
            "AI scored candidates CSV generated from # JD message "
            f"(ts={jd_message_ts or 'unknown'})."
        ),
    )
    file_id = (slack_upload.get("file") or {}).get("id")
    print(f"Uploaded scored CSV to Slack channel {args.channel_id}. file_id={file_id}")
    post_pipeline_update(
        slack_token=slack_token,
        channel_id=args.channel_id,
        text=f"Pipeline complete. Scored CSV uploaded (file_id={file_id}).",
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
