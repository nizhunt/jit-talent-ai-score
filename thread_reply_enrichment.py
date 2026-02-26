import csv
import os
import re
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests


SLACK_CONVERSATIONS_REPLIES_URL = "https://slack.com/api/conversations.replies"
SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"

SALESQL_ENRICH_URL = "https://api-public.salesql.com/v1/persons/enrich"
REOON_CREATE_TASK_URL = "https://emailverifier.reoon.com/api/v1/create-bulk-verification-task/"
REOON_GET_TASK_URL = "https://emailverifier.reoon.com/api/v1/get-result-bulk-verification-task/"
BOUNCEBAN_BULK_VERIFY_URL = "https://api.bounceban.com/v1/verify/bulk"
BOUNCEBAN_BULK_STATUS_URL = "https://api.bounceban.com/v1/verify/bulk/status"
BOUNCEBAN_BULK_DUMP_URL = "https://api.bounceban.com/v1/verify/bulk/dump"

INSTANTLY_CAMPAIGN_CREATE_URL = "https://api.instantly.ai/api/v2/campaigns"
INSTANTLY_LEAD_CREATE_URL = "https://api.instantly.ai/api/v2/leads"

RESULT_MESSAGE_PREFIX_DEFAULT = "AI-scored candidates CSV for this JD"


def parse_threshold_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"(?<![\d.])(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        threshold = float(match.group(1))
    except ValueError:
        return None
    if threshold < 0 or threshold > 10:
        return None
    return threshold


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _is_truthy(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _clean_company_name(name: str) -> str:
    if not name or not name.strip():
        return ""

    clean = name.strip()

    legal_suffixes = re.compile(r"\b(GmbH|Ltd|AG|Inc|LLC|Oy|BV|SA|S\.?r\.?l\.?|Srl|Corp|Corporation|Plc)\b", re.IGNORECASE)
    clean = legal_suffixes.sub("", clean)

    if "|" in clean:
        clean = clean.split("|", 1)[0]
    if " - " in clean:
        clean = clean.split(" - ", 1)[0]
    if " – " in clean:
        clean = clean.split(" – ", 1)[0]
    clean = re.sub(r"\s*\(.*?\)\s*", " ", clean)

    clean = re.sub(r"[\U0001F300-\U0001FAFF®™•]", "", clean)

    words = clean.strip().split()
    out_words: List[str] = []
    for word in words:
        is_acronym = len(word) <= 3 and word.isupper()
        is_camel_case = any(c.islower() for c in word) and any(c.isupper() for c in word[1:])
        is_domain = "." in word
        is_ai = word.upper() == "AI"

        if is_ai:
            out_words.append("AI")
            continue
        if is_acronym or is_camel_case or is_domain:
            out_words.append(word)
            continue
        if len(word) > 3 and word.isupper():
            out_words.append(word[:1] + word[1:].lower())
        else:
            out_words.append(word)

    generic_tails = {
        "technologies",
        "solutions",
        "systems",
        "group",
        "holdings",
        "services",
        "media",
        "corporation",
        "company",
        "networks",
        "network",
    }
    if len(out_words) > 1:
        last_word = re.sub(r"[^\w]", "", out_words[-1]).lower()
        if last_word in generic_tails:
            out_words = out_words[:-1]

    clean = " ".join(out_words)
    clean = re.sub(r"[,\-\|\.\s]+$", "", clean).strip()
    return clean


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _normalize_linkedin_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    if raw.startswith("www."):
        raw = f"https://{raw}"
    elif not raw.startswith("http://") and not raw.startswith("https://"):
        raw = f"https://{raw}"

    parsed = urlsplit(raw)
    netloc = parsed.netloc.lower()
    if "linkedin.com" not in netloc:
        return ""

    normalized = urlunsplit(("https", parsed.netloc, parsed.path.rstrip("/"), "", ""))
    return normalized


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _pick_first(row: Dict[str, Any], keys: List[str]) -> str:
    # Exact-key pass first (fast path).
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text

    # Fuzzy pass: case-insensitive and punctuation-insensitive header matching.
    def _normalize_key(raw: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (raw or "").strip().lower())

    normalized_values: Dict[str, str] = {}
    for raw_key, raw_value in row.items():
        if raw_value is None:
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        nkey = _normalize_key(str(raw_key))
        if nkey and nkey not in normalized_values:
            normalized_values[nkey] = text

    for key in keys:
        nkey = _normalize_key(key)
        if not nkey:
            continue
        value = normalized_values.get(nkey, "")
        if value:
            return value
    return ""


def _slack_headers(slack_token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {slack_token}"}


def _slack_post_message(slack_token: str, channel_id: str, text: str, thread_ts: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"channel": channel_id, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    response = requests.post(
        SLACK_POST_MESSAGE_URL,
        headers={
            "Authorization": f"Bearer {slack_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Slack chat.postMessage failed: {body.get('error')}")
    return body


def post_thread_reply_update(slack_token: str, channel_id: str, thread_ts: str, text: str) -> Dict[str, Any]:
    return _slack_post_message(
        slack_token=slack_token,
        channel_id=channel_id,
        thread_ts=thread_ts,
        text=text,
    )


def _get_thread_root_message(slack_token: str, channel_id: str, thread_ts: str) -> Dict[str, Any]:
    response = requests.get(
        SLACK_CONVERSATIONS_REPLIES_URL,
        headers=_slack_headers(slack_token),
        params={
            "channel": channel_id,
            "ts": thread_ts,
            "inclusive": "true",
            "oldest": thread_ts,
            "latest": thread_ts,
            "limit": 1,
        },
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Slack conversations.replies failed: {body.get('error')}")
    messages = body.get("messages") or []
    if not messages:
        raise RuntimeError("Slack thread root message not found.")
    return messages[0]


def _extract_csv_file_info_from_message(message: Dict[str, Any]) -> Dict[str, Any]:
    files = message.get("files") or []
    for item in files:
        mimetype = (item.get("mimetype") or "").lower()
        filetype = (item.get("filetype") or "").lower()
        name = (item.get("name") or "").lower()
        if mimetype == "text/csv" or filetype == "csv" or name.endswith(".csv"):
            return item
    raise RuntimeError("Thread root message does not include a CSV file.")


def _is_result_message_thread_root(message: Dict[str, Any], csv_file: Dict[str, Any]) -> bool:
    text = (message.get("text") or "").strip().lower()
    prefix = os.getenv("RESULT_MESSAGE_PREFIX", RESULT_MESSAGE_PREFIX_DEFAULT).strip().lower()
    strict = _is_truthy(os.getenv("THREAD_RESULT_STRICT"), default=True)

    name = (csv_file.get("name") or "").strip().lower()
    title = (csv_file.get("title") or "").strip().lower()
    filename_hint = "sheet-ready" in name or "sheet-ready" in title or "scored" in name or "scored" in title

    has_prefix = bool(prefix and prefix in text)
    if strict:
        return has_prefix
    return has_prefix or filename_hint


def _download_slack_file(slack_token: str, url: str, destination: Path) -> None:
    response = requests.get(url, headers=_slack_headers(slack_token), timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)


def _load_candidates_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = _to_float(
                _pick_first(
                    row,
                    [
                        "AI Score",
                        "ai-score",
                        "ai_score",
                        "Score",
                        "score",
                    ],
                )
            )
            linkedin = _normalize_linkedin_url(
                _pick_first(
                    row,
                    [
                        "LinkedIn",
                        "linkedin",
                        "Linkedin",
                        "hs_linkedin_url",
                        "URL",
                    ],
                )
            )
            if score is None or not linkedin:
                continue

            candidates.append(
                {
                    "score": score,
                    "linkedin_url": linkedin,
                    "first_name": _pick_first(row, ["First Name", "first_name", "firstname"]),
                    "last_name": _pick_first(row, ["Last Name", "last_name", "lastname"]),
                    "company_name": _pick_first(
                        row,
                        [
                            "Current Company",
                            "current_company",
                            "Company",
                            "company",
                            "Company Name",
                            "company_name",
                        ],
                    ),
                }
            )
    return candidates


def _salesql_enrich(salesql_api_key: str, linkedin_url: str) -> Dict[str, Any]:
    response = requests.get(
        SALESQL_ENRICH_URL,
        params={"linkedin_url": linkedin_url, "api_key": salesql_api_key},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _build_email_metadata_from_salesql(candidates: List[Dict[str, Any]], salesql_api_key: str) -> Dict[str, Dict[str, str]]:
    metadata_by_email: Dict[str, Dict[str, str]] = {}
    seen_linkedin_urls = set()
    for candidate in candidates:
        linkedin_url = candidate["linkedin_url"]
        if linkedin_url in seen_linkedin_urls:
            continue
        seen_linkedin_urls.add(linkedin_url)
        person = _salesql_enrich(salesql_api_key=salesql_api_key, linkedin_url=linkedin_url)

        fallback_first = candidate.get("first_name", "")
        fallback_last = candidate.get("last_name", "")
        fallback_company = candidate.get("company_name", "")

        first_name = (person.get("first_name") or fallback_first or "").strip()
        last_name = (person.get("last_name") or fallback_last or "").strip()
        org = person.get("organization") or {}
        company_name = (org.get("name") or fallback_company or "").strip()

        for email_obj in person.get("emails") or []:
            email_type = (email_obj.get("type") or "").strip().lower()
            if email_type != "direct":
                continue
            is_valid = email_obj.get("is_valid") is True
            if not is_valid:
                continue
            email = _normalize_email(email_obj.get("email", ""))
            if not email:
                continue
            metadata_by_email[email] = {
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "company_name": company_name,
                "linkedin_url": linkedin_url,
            }
    return metadata_by_email


def _submit_reoon_task(reoon_api_key: str, emails: List[str]) -> int:
    response = requests.post(
        REOON_CREATE_TASK_URL,
        json={"key": reoon_api_key, "emails": emails},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "success":
        raise RuntimeError(f"Reoon task creation failed: {body}")
    task_id = body.get("task_id")
    if not task_id:
        raise RuntimeError("Reoon task creation response missing task_id.")
    return int(task_id)


def _poll_reoon_task(reoon_api_key: str, task_id: int) -> Dict[str, Any]:
    max_wait_seconds = int(os.getenv("REOON_MAX_WAIT_SECONDS", "600"))
    poll_interval_seconds = int(os.getenv("REOON_POLL_INTERVAL_SECONDS", "15"))
    deadline = time.monotonic() + max_wait_seconds

    while True:
        response = requests.get(
            REOON_GET_TASK_URL,
            params={"key": reoon_api_key, "task_id": task_id},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        status = (body.get("status") or "").lower()
        if status == "completed":
            return body
        if status not in {"running", "queued", "processing"}:
            raise RuntimeError(f"Unexpected Reoon status: {body}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for Reoon task {task_id}.")
        time.sleep(max(1, poll_interval_seconds))


def _filter_reoon_results(reoon_result: Dict[str, Any], input_emails: List[str]) -> List[str]:
    results = reoon_result.get("results") or {}
    allowed: List[str] = []
    for email in input_emails:
        item = results.get(email) or {}
        is_safe = item.get("is_safe_to_send") is True
        is_catch_all = item.get("is_catch_all") is True
        if is_safe or is_catch_all:
            allowed.append(email)
    return allowed


def _submit_bounceban_task(bounceban_api_key: str, emails: List[str]) -> str:
    response = requests.post(
        BOUNCEBAN_BULK_VERIFY_URL,
        headers={"Authorization": bounceban_api_key, "Content-Type": "application/json"},
        json={"emails": emails},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    task_id = body.get("id")
    if not task_id:
        raise RuntimeError(f"BounceBan task creation failed: {body}")
    return str(task_id)


def _poll_bounceban_completion(bounceban_api_key: str, task_id: str) -> None:
    max_wait_seconds = int(os.getenv("BOUNCEBAN_MAX_WAIT_SECONDS", "600"))
    poll_interval_seconds = int(os.getenv("BOUNCEBAN_POLL_INTERVAL_SECONDS", "15"))
    deadline = time.monotonic() + max_wait_seconds

    while True:
        response = requests.get(
            BOUNCEBAN_BULK_STATUS_URL,
            headers={"Authorization": bounceban_api_key},
            params={"id": task_id},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        status = (body.get("status") or "").lower()
        if status in {"finished", "completed"}:
            return
        if status not in {"pending", "running", "processing"}:
            raise RuntimeError(f"Unexpected BounceBan status: {body}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for BounceBan task {task_id}.")
        time.sleep(max(1, poll_interval_seconds))


def _fetch_bounceban_dump(bounceban_api_key: str, task_id: str) -> Dict[str, Any]:
    response = requests.get(
        BOUNCEBAN_BULK_DUMP_URL,
        headers={"Authorization": bounceban_api_key},
        params={"id": task_id, "retrieve_all": "1"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _filter_bounceban_deliverables(dump_payload: Dict[str, Any], input_emails: List[str]) -> List[str]:
    item_by_email: Dict[str, Dict[str, Any]] = {}
    for item in dump_payload.get("items") or []:
        email = _normalize_email(item.get("email", ""))
        if not email:
            continue
        item_by_email[email] = item

    deliverable: List[str] = []
    for email in input_emails:
        item = item_by_email.get(email) or {}
        if (item.get("result") or "").lower() == "deliverable":
            deliverable.append(email)
    return deliverable


def _create_instantly_campaign(instantly_api_key: str, threshold: float) -> str:
    timezone = os.getenv("INSTANTLY_CAMPAIGN_TIMEZONE", "Asia/Kolkata")
    start_hour = os.getenv("INSTANTLY_CAMPAIGN_START_HOUR", "08:00")
    end_hour = os.getenv("INSTANTLY_CAMPAIGN_END_HOUR", "18:00")
    duration_days = int(os.getenv("INSTANTLY_CAMPAIGN_DURATION_DAYS", "30"))

    start_date = date.today()
    end_date = start_date + timedelta(days=max(1, duration_days))
    campaign_name = f"JIT Outreach Threshold {threshold:g} ({start_date.isoformat()})"

    payload = {
        "name": campaign_name,
        "campaign_schedule": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "schedules": [
                {
                    "name": "Default",
                    "days": {"0": True, "1": True, "2": True, "3": True, "4": True, "5": False, "6": False},
                    "timing": {"from": start_hour, "to": end_hour},
                    "timezone": timezone,
                }
            ],
        },
        "sequences": [
            {
                "steps": [
                    {
                        "type": "email",
                        "delay": 0,
                        "variants": [{"subject": "", "body": ""}],
                    }
                ]
            }
        ],
        "email_list": [],
    }
    response = requests.post(
        INSTANTLY_CAMPAIGN_CREATE_URL,
        headers={"Authorization": f"Bearer {instantly_api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    campaign_id = body.get("id")
    if not campaign_id:
        raise RuntimeError(f"Instantly campaign creation failed: {body}")
    return str(campaign_id)


def _add_lead_to_instantly_campaign(
    instantly_api_key: str,
    campaign_id: str,
    lead: Dict[str, str],
) -> Dict[str, Any]:
    linkedin_url = lead.get("linkedin_url", "")
    payload = {
        "email": lead["email"],
        "campaign": campaign_id,
        "first_name": lead.get("first_name", ""),
        "last_name": lead.get("last_name", ""),
        "company_name": _clean_company_name(lead.get("company_name", "")),
        "website": linkedin_url,
        "personalization": f"LinkedIn: {linkedin_url}" if linkedin_url else "",
    }
    response = requests.post(
        INSTANTLY_LEAD_CREATE_URL,
        headers={"Authorization": f"Bearer {instantly_api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def run_thread_reply_enrichment_pipeline(
    *,
    slack_token: str,
    channel_id: str,
    thread_ts: str,
    threshold: float,
    post_updates: bool = True,
) -> Dict[str, Any]:
    salesql_api_key = _require_env("SALESQL_API_KEY")
    reoon_api_key = _require_env("REOON_API_KEY")
    bounceban_api_key = _require_env("BOUNCEBAN_API_KEY")
    instantly_api_key = _require_env("INSTANTLY_API_KEY")

    def _update(text: str) -> None:
        if post_updates:
            _slack_post_message(slack_token=slack_token, channel_id=channel_id, thread_ts=thread_ts, text=text)

    root_message = _get_thread_root_message(slack_token=slack_token, channel_id=channel_id, thread_ts=thread_ts)
    csv_file = _extract_csv_file_info_from_message(root_message)
    if not _is_result_message_thread_root(root_message, csv_file):
        return {
            "ignored": "not_result_message_thread",
            "threshold": threshold,
            "csv_filename": csv_file.get("name") or "unknown.csv",
            "campaign_id": None,
            "leads_added": 0,
            "lead_errors": [],
        }
    file_url = csv_file.get("url_private_download") or csv_file.get("url_private")
    if not file_url:
        raise RuntimeError("CSV file is missing url_private_download/url_private.")

    with tempfile.TemporaryDirectory(prefix="jit-thread-enrich-") as tmp_dir:
        csv_path = Path(tmp_dir) / (csv_file.get("name") or "scored.csv")
        _download_slack_file(slack_token=slack_token, url=file_url, destination=csv_path)

        all_candidates = _load_candidates_from_csv(csv_path)
        threshold_candidates = [c for c in all_candidates if c["score"] >= threshold]
        threshold_candidates = [c for c in threshold_candidates if c["linkedin_url"]]

        if not threshold_candidates:
            return {
                "threshold": threshold,
                "csv_filename": csv_file.get("name") or "scored.csv",
                "rows_with_score_and_linkedin": len(all_candidates),
                "rows_meeting_threshold": 0,
                "salesql_emails_found": 0,
                "reoon_passed": 0,
                "bounceban_deliverable": 0,
                "campaign_id": None,
                "leads_added": 0,
                "lead_errors": [],
            }

        _update(f"Threshold {threshold:g} accepted {len(threshold_candidates)} candidates. Enriching emails via SaleSQL...")

        metadata_by_email = _build_email_metadata_from_salesql(
            candidates=threshold_candidates,
            salesql_api_key=salesql_api_key,
        )
        all_emails = _dedupe_preserve_order(list(metadata_by_email.keys()))
        if not all_emails:
            return {
                "threshold": threshold,
                "csv_filename": csv_file.get("name") or "scored.csv",
                "rows_with_score_and_linkedin": len(all_candidates),
                "rows_meeting_threshold": len(threshold_candidates),
                "salesql_emails_found": 0,
                "reoon_passed": 0,
                "bounceban_deliverable": 0,
                "campaign_id": None,
                "leads_added": 0,
                "lead_errors": [],
            }

        _update(f"SaleSQL found {len(all_emails)} emails. Sending to Reoon...")
        reoon_task_id = _submit_reoon_task(reoon_api_key=reoon_api_key, emails=all_emails)
        reoon_result = _poll_reoon_task(reoon_api_key=reoon_api_key, task_id=reoon_task_id)
        reoon_passed = _dedupe_preserve_order(_filter_reoon_results(reoon_result=reoon_result, input_emails=all_emails))

        if not reoon_passed:
            return {
                "threshold": threshold,
                "csv_filename": csv_file.get("name") or "scored.csv",
                "rows_with_score_and_linkedin": len(all_candidates),
                "rows_meeting_threshold": len(threshold_candidates),
                "salesql_emails_found": len(all_emails),
                "reoon_passed": 0,
                "bounceban_deliverable": 0,
                "campaign_id": None,
                "leads_added": 0,
                "lead_errors": [],
            }

        _update(f"Reoon passed {len(reoon_passed)} emails. Sending to BounceBan...")
        bounceban_task_id = _submit_bounceban_task(bounceban_api_key=bounceban_api_key, emails=reoon_passed)
        _poll_bounceban_completion(bounceban_api_key=bounceban_api_key, task_id=bounceban_task_id)
        bounceban_dump = _fetch_bounceban_dump(bounceban_api_key=bounceban_api_key, task_id=bounceban_task_id)
        deliverable = _dedupe_preserve_order(
            _filter_bounceban_deliverables(dump_payload=bounceban_dump, input_emails=reoon_passed)
        )

        if not deliverable:
            return {
                "threshold": threshold,
                "csv_filename": csv_file.get("name") or "scored.csv",
                "rows_with_score_and_linkedin": len(all_candidates),
                "rows_meeting_threshold": len(threshold_candidates),
                "salesql_emails_found": len(all_emails),
                "reoon_passed": len(reoon_passed),
                "bounceban_deliverable": 0,
                "campaign_id": None,
                "leads_added": 0,
                "lead_errors": [],
            }

        _update(f"BounceBan deliverable: {len(deliverable)}. Creating Instantly campaign...")
        campaign_id = _create_instantly_campaign(instantly_api_key=instantly_api_key, threshold=threshold)

        lead_errors: List[str] = []
        added_count = 0
        fail_fast = _is_truthy(os.getenv("INSTANTLY_FAIL_FAST"), default=False)
        for email in deliverable:
            lead_meta = metadata_by_email.get(email) or {"email": email}
            try:
                _add_lead_to_instantly_campaign(
                    instantly_api_key=instantly_api_key,
                    campaign_id=campaign_id,
                    lead=lead_meta,
                )
                added_count += 1
            except Exception as exc:
                lead_errors.append(f"{email}: {exc}")
                if fail_fast:
                    raise

        return {
            "threshold": threshold,
            "csv_filename": csv_file.get("name") or "scored.csv",
            "rows_with_score_and_linkedin": len(all_candidates),
            "rows_meeting_threshold": len(threshold_candidates),
            "salesql_emails_found": len(all_emails),
            "reoon_passed": len(reoon_passed),
            "bounceban_deliverable": len(deliverable),
            "campaign_id": campaign_id,
            "leads_added": added_count,
            "lead_errors": lead_errors,
        }
