import html
import os
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from google_sheets import load_rows_from_google_sheet_url

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

HEYREACH_BASE_URL = "https://api.heyreach.io/api/public"
HEYREACH_CREATE_LIST_URL = f"{HEYREACH_BASE_URL}/list/CreateEmptyList"
HEYREACH_ADD_LEADS_URL = f"{HEYREACH_BASE_URL}/list/AddLeadsToListV2"
HEYREACH_USER_LIST_TYPE = "USER_LIST"
INSTANTLY_STEP_ONE_SUBJECT = "New role at a startup!"
INSTANTLY_STEP_ONE_BODY = (
    "Hi {{firstName}},\n\n"
    "{{personalization}}\n\n"
    "Do you have a few minutes this week? Great to share more\n"
    "Best,\n"
    "Dan\n"
    "CEO and Co-founder, Calyptus"
)
INSTANTLY_STEP_TWO_BODY = (
    "Following up here as the team are moving fast, and you have a fantastic profile.\n"
    "Would you be open to chat?\n\n"
    "Love to connect you with Dianmarie, our senior recruiter who is leading the process!\n\n"
    "Best,\n"
    "Dan\n"
    "CEO and Co-founder, Calyptus"
)
INSTANTLY_STEP_THREE_BODY = (
    "Following up here as the team are moving fast, and you have a fantastic profile.\n\n"
    "Would you be open to chat?\n\n"
    "Love to connect you with Dianmarie, our senior recruiter who is leading the process!\n\n"
    "Best,\n"
    "Dan\n"
    "CEO and Co-founder, Calyptus"
)

RESULT_MESSAGE_PREFIX_DEFAULT = "AI-scored candidates sheet for this JD"
GOOGLE_SHEET_URL_IN_TEXT_RE = re.compile(
    r"https://docs\.google\.com/spreadsheets/d/[a-zA-Z0-9-_]+[^\s<>()]*"
)


def parse_threshold_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    # Accept only a bare integer threshold (1-10), optionally surrounded by whitespace.
    match = re.fullmatch(r"\s*(10|[1-9])\s*", text)
    if not match:
        return None
    return float(match.group(1))


def parse_threshold_and_target_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse '<score>-instantly', '<score>-heyreach', or bare '<score>' from thread reply text.

    Returns ``None`` when the text does not match any recognised format, or a dict
    ``{"threshold": float, "target": "instantly" | "heyreach"}`` on success.
    A bare score (no suffix) is no longer accepted — the user must specify a target.

    For heyreach, an optional Calendly URL may follow the command, e.g.::

        8-heyreach https://calendly.com/nicole-calyptus/t007
    """
    if not text:
        return None
    match = re.fullmatch(
        r"\s*(10|[1-9])\s*-\s*(instantly|heyreach)"
        r"(?:\s+(https?://calendly\.com/\S+))?\s*",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    result: Dict[str, Any] = {
        "threshold": float(match.group(1)),
        "target": match.group(2).strip().lower(),
    }
    calendly_url = (match.group(3) or "").strip()
    if calendly_url:
        result["calendly_url"] = calendly_url
    return result


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


def _clean_personalization_snippet(generated_email: str) -> str:
    text = (generated_email or "").strip()
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        tag_match = re.match(r"^line\s*[123]\s*:\s*(.*)$", stripped, flags=re.IGNORECASE)
        if tag_match:
            cleaned_lines.append(tag_match.group(1).strip())
        else:
            cleaned_lines.append(raw_line.rstrip())

    return "\n".join(cleaned_lines).strip()


def _format_instantly_html_text(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    normalized = clean.replace("\r\n", "\n").replace("\r", "\n")
    escaped = html.escape(normalized, quote=False)
    return escaped.replace("\n\n", "<br /><br />").replace("\n", "<br />")


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


def _slack_post_message(
    slack_token: str,
    channel_id: str,
    text: str,
    thread_ts: Optional[str] = None,
    blocks: Optional[List[Dict[str, Any]]] = None,
    unfurl_links: Optional[bool] = None,
    unfurl_media: Optional[bool] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"channel": channel_id, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    if blocks:
        payload["blocks"] = blocks
    if unfurl_links is not None:
        payload["unfurl_links"] = unfurl_links
    if unfurl_media is not None:
        payload["unfurl_media"] = unfurl_media
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


def post_thread_reply_update(
    slack_token: str,
    channel_id: str,
    thread_ts: str,
    text: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    unfurl_links: Optional[bool] = None,
    unfurl_media: Optional[bool] = None,
) -> Dict[str, Any]:
    return _slack_post_message(
        slack_token=slack_token,
        channel_id=channel_id,
        thread_ts=thread_ts,
        text=text,
        blocks=blocks,
        unfurl_links=unfurl_links,
        unfurl_media=unfurl_media,
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


def _extract_jd_name_from_message(message: Dict[str, Any]) -> str:
    for part in _collect_thread_root_text_parts(message):
        jd_name = _extract_jd_name_from_text(part)
        if jd_name:
            return jd_name
    return ""


def _clean_extracted_jd_name(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^[`*_]+|[`*_]+$", "", cleaned).strip()
    if cleaned.lower() in {"n/a", "na", "none", "null", "-", "(none)"}:
        return ""
    return cleaned


def _extract_jd_name_from_text(text: str) -> str:
    if not text:
        return ""

    raw_lines = [line.strip() for line in text.splitlines() if line and line.strip()]
    for idx, line in enumerate(raw_lines):
        explicit_line = re.match(r"(?i)^JD Name\s*:\s*(.+)\s*$", line)
        if explicit_line:
            candidate = _clean_extracted_jd_name(explicit_line.group(1))
            if candidate:
                return candidate

        normalized_label = re.sub(r"[*_`]+", "", line).strip().lower()
        if normalized_label == "jd name" and idx + 1 < len(raw_lines):
            candidate = _clean_extracted_jd_name(raw_lines[idx + 1])
            if candidate:
                return candidate

    return ""


def _collect_text_values(value: Any, out: List[str]) -> None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            # Slack block payloads often keep useful text in `text`/`fallback` fields.
            if key in {"text", "fallback", "pretext", "title", "comment"}:
                _collect_text_values(nested, out)
            elif key in {"fields", "elements", "blocks", "attachments", "initial_comment"}:
                _collect_text_values(nested, out)
        return
    if isinstance(value, list):
        for nested in value:
            _collect_text_values(nested, out)


def _collect_thread_root_text_parts(message: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    _collect_text_values(message.get("text"), parts)
    _collect_text_values(message.get("blocks"), parts)
    _collect_text_values(message.get("attachments"), parts)

    # File-share roots may keep the initial comment under the file payload.
    for item in message.get("files") or []:
        _collect_text_values(item.get("initial_comment"), parts)
    return parts


def _extract_thread_root_text_for_validation(message: Dict[str, Any]) -> str:
    parts = _collect_thread_root_text_parts(message)
    return "\n".join(parts).strip().lower()


def _extract_google_sheet_url_from_text(text: str) -> str:
    if not text:
        return ""

    slack_url_match = re.search(
        r"<(https://docs\.google\.com/spreadsheets/d/[a-zA-Z0-9-_]+[^>|]*)(?:\|[^>]+)?>",
        text,
    )
    if slack_url_match:
        return slack_url_match.group(1).strip()

    url_match = GOOGLE_SHEET_URL_IN_TEXT_RE.search(text)
    if not url_match:
        return ""
    return url_match.group(0).rstrip(").,")


def _extract_google_sheet_url_from_message(message: Dict[str, Any]) -> str:
    for part in _collect_thread_root_text_parts(message):
        url = _extract_google_sheet_url_from_text(part)
        if url:
            return url
    return ""


def _is_result_message_thread_root(message: Dict[str, Any], sheet_url: str) -> bool:
    text = _extract_thread_root_text_for_validation(message)
    prefix = os.getenv("RESULT_MESSAGE_PREFIX", RESULT_MESSAGE_PREFIX_DEFAULT).strip().lower()
    strict = _is_truthy(os.getenv("THREAD_RESULT_STRICT"), default=True)
    has_sheet_link = bool(sheet_url)

    has_prefix = bool(prefix and prefix in text)
    has_new_summary_markers = (
        "scored:" in text
        and "finds (score >= 5)" in text
        and "score tally:" in text
        and "linkedin samples by score:" in text
    )
    if strict:
        return has_sheet_link and (has_prefix or has_new_summary_markers)
    return has_sheet_link and (has_prefix or has_new_summary_markers or not prefix)


def _load_candidates_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for row in rows:
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
                "generated_email": _pick_first(
                    row,
                    [
                        "AI Email",
                        "ai-email",
                        "ai_email",
                        "Generated Email",
                        "generated_email",
                    ],
                ),
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


class _SalesQLKeyPool:
    """Round-robin pool of SalesQL API keys with automatic failover.

    Keys are tried in round-robin order so that usage is spread evenly across
    accounts.  When a key returns 401/403 (insufficient credits or invalid),
    it is marked as exhausted and skipped for subsequent calls.  If every key
    is exhausted the pool raises ``RuntimeError``.
    """

    def __init__(self, api_keys: List[str]) -> None:
        if not api_keys:
            raise RuntimeError("No SalesQL API keys provided.")
        self._keys: List[str] = api_keys
        self._exhausted: set = set()
        self._index: int = 0

    def _next_key(self) -> str:
        """Return the next usable key via round-robin, skipping exhausted ones."""
        tried = 0
        while tried < len(self._keys):
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            if key not in self._exhausted:
                return key
            tried += 1
        raise RuntimeError(
            f"All {len(self._keys)} SalesQL API key(s) are exhausted (insufficient credits or invalid)."
        )

    def _mark_exhausted(self, key: str) -> None:
        self._exhausted.add(key)
        remaining = len(self._keys) - len(self._exhausted)
        print(f"[salesql] API key {key[:6]}…{key[-4:]} exhausted. {remaining} key(s) remaining.")

    @property
    def active_key_count(self) -> int:
        return len(self._keys) - len(self._exhausted)

    def enrich(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """Enrich a LinkedIn profile, rotating keys on 401/403."""
        last_error: Optional[Exception] = None
        attempts = 0
        while attempts < len(self._keys):
            try:
                key = self._next_key()
            except RuntimeError:
                break
            attempts += 1
            response = requests.get(
                SALESQL_ENRICH_URL,
                params={"linkedin_url": linkedin_url, "api_key": key},
                timeout=60,
            )
            # Profile not found — skip regardless of which key was used.
            if response.status_code in {400, 404, 422}:
                return None
            if response.status_code in {401, 403}:
                self._mark_exhausted(key)
                last_error = RuntimeError(
                    f"SalesQL API returned {response.status_code} for key {key[:6]}…{key[-4:]}."
                )
                continue
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                return None
            return body

        raise RuntimeError(
            f"All {len(self._keys)} SalesQL API key(s) are exhausted (insufficient credits or invalid)."
        ) from last_error


def _load_salesql_api_keys() -> List[str]:
    """Load SalesQL API keys from environment.

    Supports two env vars (both are merged and deduplicated):
      - SALESQL_API_KEYS  — comma-separated list (preferred)
      - SALESQL_API_KEY   — single key (backward-compatible)
    """
    raw_plural = (os.getenv("SALESQL_API_KEYS") or "").strip()
    raw_single = (os.getenv("SALESQL_API_KEY") or "").strip()
    keys: List[str] = []
    seen: set = set()
    for raw in (raw_plural, raw_single):
        for part in raw.split(","):
            k = part.strip()
            if k and k not in seen:
                keys.append(k)
                seen.add(k)
    if not keys:
        raise RuntimeError("Missing required env var: SALESQL_API_KEYS (or SALESQL_API_KEY)")
    print(f"[salesql] Loaded {len(keys)} API key(s): {', '.join(k[:6] + '…' + k[-4:] for k in keys)}")
    return keys


def _salesql_enrich(salesql_api_key: str, linkedin_url: str) -> Optional[Dict[str, Any]]:
    """Legacy single-key enrich (kept for backward compatibility)."""
    pool = _SalesQLKeyPool([salesql_api_key])
    return pool.enrich(linkedin_url)


def _build_email_metadata_from_salesql(
    candidates: List[Dict[str, Any]],
    salesql_api_key: str = "",
    salesql_key_pool: Optional[_SalesQLKeyPool] = None,
) -> Dict[str, Dict[str, str]]:
    metadata_by_email: Dict[str, Dict[str, str]] = {}
    seen_linkedin_urls = set()
    for candidate in candidates:
        linkedin_url = candidate["linkedin_url"]
        if linkedin_url in seen_linkedin_urls:
            continue
        seen_linkedin_urls.add(linkedin_url)
        if salesql_key_pool is not None:
            person = salesql_key_pool.enrich(linkedin_url)
        else:
            person = _salesql_enrich(salesql_api_key=salesql_api_key, linkedin_url=linkedin_url)
        if not person:
            continue

        fallback_first = candidate.get("first_name", "")
        fallback_last = candidate.get("last_name", "")
        fallback_company = candidate.get("company_name", "")
        fallback_generated_email = candidate.get("generated_email", "")

        first_name = (person.get("first_name") or fallback_first or "").strip()
        last_name = (person.get("last_name") or fallback_last or "").strip()
        org = person.get("organization") or {}
        company_name = (org.get("name") or fallback_company or "").strip()
        generated_email = (fallback_generated_email or "").strip()

        for email_obj in person.get("emails") or []:
            email_type = (email_obj.get("type") or "").strip().lower()
            if email_type != "direct":
                continue
            is_valid_value = email_obj.get("is_valid")
            status_value = (email_obj.get("status") or "").strip().lower()
            if is_valid_value is None:
                is_valid = status_value == "valid"
            else:
                is_valid = is_valid_value is True
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
                "generated_email": generated_email,
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
    max_wait_seconds = int(os.getenv("REOON_MAX_WAIT_SECONDS", "1800"))
    poll_interval_seconds = int(os.getenv("REOON_POLL_INTERVAL_SECONDS", "15"))
    deadline = time.monotonic() + max_wait_seconds
    last_status = "unknown"
    last_checked: Optional[int] = None
    last_total: Optional[int] = None

    while True:
        response = requests.get(
            REOON_GET_TASK_URL,
            params={"key": reoon_api_key, "task_id": task_id},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()
        status = (body.get("status") or "").lower()
        if status:
            last_status = status
        count_checked = body.get("count_checked")
        count_total = body.get("count_total")
        if isinstance(count_checked, int):
            last_checked = count_checked
        if isinstance(count_total, int):
            last_total = count_total
        if status == "completed":
            return body
        if status not in {"waiting", "running", "queued", "processing"}:
            raise RuntimeError(f"Unexpected Reoon status: {body}")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for Reoon task {task_id} after {max_wait_seconds}s "
                f"(last_status={last_status}, checked={last_checked}, total={last_total})."
            )
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
        if status not in {"pending", "running", "processing", "verifying"}:
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


def _build_instantly_campaign_name(*, jd_name: str, threshold: float, start_date: date) -> str:
    clean_jd_name = (jd_name or "").strip()
    if clean_jd_name:
        return clean_jd_name
    return f"JIT Outreach Threshold {threshold:g} ({start_date.isoformat()})"


def _create_instantly_campaign(instantly_api_key: str, threshold: float, jd_name: str = "") -> Dict[str, str]:
    timezone = os.getenv("INSTANTLY_CAMPAIGN_TIMEZONE", "Asia/Kolkata")
    start_hour = os.getenv("INSTANTLY_CAMPAIGN_START_HOUR", "08:00")
    end_hour = os.getenv("INSTANTLY_CAMPAIGN_END_HOUR", "18:00")
    duration_days = int(os.getenv("INSTANTLY_CAMPAIGN_DURATION_DAYS", "30"))

    start_date = date.today()
    end_date = start_date + timedelta(days=max(1, duration_days))
    campaign_name = _build_instantly_campaign_name(jd_name=jd_name, threshold=threshold, start_date=start_date)

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
                        "delay": 2,
                        "delay_unit": "days",
                        "variants": [{"subject": INSTANTLY_STEP_ONE_SUBJECT, "body": INSTANTLY_STEP_ONE_BODY}],
                    },
                    {
                        "type": "email",
                        "delay": 2,
                        "delay_unit": "days",
                        "variants": [{"subject": INSTANTLY_STEP_ONE_SUBJECT, "body": INSTANTLY_STEP_TWO_BODY}],
                    },
                    {
                        "type": "email",
                        "delay": 0,
                        "delay_unit": "days",
                        "variants": [{"subject": INSTANTLY_STEP_ONE_SUBJECT, "body": INSTANTLY_STEP_THREE_BODY}],
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
    return {"id": str(campaign_id), "name": campaign_name}


def _add_lead_to_instantly_campaign(
    instantly_api_key: str,
    campaign_id: str,
    lead: Dict[str, str],
) -> Dict[str, Any]:
    linkedin_url = lead.get("linkedin_url", "")
    generated_email = (lead.get("generated_email") or "").strip()
    personalization_body = _clean_personalization_snippet(generated_email)
    personalization_raw = personalization_body if personalization_body else (f"LinkedIn: {linkedin_url}" if linkedin_url else "")
    personalization = _format_instantly_html_text(personalization_raw)
    skip_if_in_workspace = _is_truthy(os.getenv("INSTANTLY_SKIP_IF_IN_WORKSPACE"), default=True)
    payload = {
        "email": lead["email"],
        "campaign": campaign_id,
        "skip_if_in_workspace": skip_if_in_workspace,
        "first_name": lead.get("first_name", ""),
        "last_name": lead.get("last_name", ""),
        "company_name": _clean_company_name(lead.get("company_name", "")),
        "website": linkedin_url,
        "personalization": personalization,
    }
    response = requests.post(
        INSTANTLY_LEAD_CREATE_URL,
        headers={"Authorization": f"Bearer {instantly_api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _classify_instantly_lead_add_result(campaign_id: str, response_body: Dict[str, Any]) -> Dict[str, str]:
    target_campaign = str(campaign_id or "").strip()
    response_campaign = str(response_body.get("campaign") or "").strip()
    lead_id = str(response_body.get("id") or "").strip()

    if response_campaign and response_campaign != target_campaign:
        return {
            "status": "skipped",
            "detail": (
                "already existed in Instantly workspace and remained attached to "
                f"campaign {response_campaign}"
            ),
        }

    if not lead_id:
        message = str(response_body.get("message") or "").strip()
        if not message:
            message = "Instantly response did not include lead id"
        return {"status": "error", "detail": message}

    return {"status": "added", "detail": ""}


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
    salesql_key_pool = _SalesQLKeyPool(_load_salesql_api_keys())
    reoon_api_key = _require_env("REOON_API_KEY")
    bounceban_api_key = _require_env("BOUNCEBAN_API_KEY")
    instantly_api_key = _require_env("INSTANTLY_API_KEY")

    def _update(text: str) -> None:
        if post_updates:
            _slack_post_message(slack_token=slack_token, channel_id=channel_id, thread_ts=thread_ts, text=text)

    root_message = _get_thread_root_message(slack_token=slack_token, channel_id=channel_id, thread_ts=thread_ts)
    sheet_url = _extract_google_sheet_url_from_message(root_message)
    jd_name = _extract_jd_name_from_message(root_message)
    source_type = "google_sheet"
    source_name = sheet_url or "unknown"
    source_url = sheet_url if sheet_url else None

    def _result_source_fields() -> Dict[str, Any]:
        return {
            "source_type": source_type,
            "source_name": source_name,
            "source_url": source_url,
        }

    if not _is_result_message_thread_root(root_message, sheet_url):
        return {
            "ignored": "not_result_message_thread",
            "threshold": threshold,
            "jd_name": jd_name,
            "campaign_id": None,
            "campaign_name": None,
            "leads_added": 0,
            "lead_skipped": [],
            "lead_errors": [],
            **_result_source_fields(),
        }

    if not sheet_url:
        raise RuntimeError("Thread root message must include a Google Sheet link.")
    sheet_payload = load_rows_from_google_sheet_url(sheet_url)
    source_name = (sheet_payload.get("spreadsheet_title") or source_name).strip()
    source_url = sheet_payload.get("spreadsheet_url") or sheet_url
    all_candidates = _load_candidates_from_rows(sheet_payload.get("rows") or [])

    threshold_candidates = [c for c in all_candidates if c["score"] >= threshold]
    threshold_candidates = [c for c in threshold_candidates if c["linkedin_url"]]

    if not threshold_candidates:
        return {
            "threshold": threshold,
            "jd_name": jd_name,
            "rows_with_score_and_linkedin": len(all_candidates),
            "rows_meeting_threshold": 0,
            "salesql_emails_found": 0,
            "reoon_passed": 0,
            "bounceban_deliverable": 0,
            "campaign_id": None,
            "campaign_name": None,
            "leads_added": 0,
            "lead_skipped": [],
            "lead_errors": [],
            **_result_source_fields(),
        }

    _update(f"Threshold {threshold:g} accepted {len(threshold_candidates)} candidates. Enriching emails via SaleSQL...")

    metadata_by_email = _build_email_metadata_from_salesql(
        candidates=threshold_candidates,
        salesql_key_pool=salesql_key_pool,
    )
    all_emails = _dedupe_preserve_order(list(metadata_by_email.keys()))
    if not all_emails:
        return {
            "threshold": threshold,
            "jd_name": jd_name,
            "rows_with_score_and_linkedin": len(all_candidates),
            "rows_meeting_threshold": len(threshold_candidates),
            "salesql_emails_found": 0,
            "reoon_passed": 0,
            "bounceban_deliverable": 0,
            "campaign_id": None,
            "campaign_name": None,
            "leads_added": 0,
            "lead_skipped": [],
            "lead_errors": [],
            **_result_source_fields(),
        }

    _update(f"SaleSQL found {len(all_emails)} emails. Sending to Reoon...")
    reoon_task_id = _submit_reoon_task(reoon_api_key=reoon_api_key, emails=all_emails)
    try:
        reoon_result = _poll_reoon_task(reoon_api_key=reoon_api_key, task_id=reoon_task_id)
    except RuntimeError as exc:
        message = str(exc)
        if "Timed out waiting for Reoon task" not in message:
            raise
        _update(
            f"Reoon task `{reoon_task_id}` did not finish in time. "
            "Stopping this run without campaign creation."
        )
        return {
            "threshold": threshold,
            "jd_name": jd_name,
            "rows_with_score_and_linkedin": len(all_candidates),
            "rows_meeting_threshold": len(threshold_candidates),
            "salesql_emails_found": len(all_emails),
            "reoon_passed": 0,
            "bounceban_deliverable": 0,
            "campaign_id": None,
            "campaign_name": None,
            "leads_added": 0,
            "lead_skipped": [],
            "lead_errors": [],
            "note": message,
            **_result_source_fields(),
        }
    reoon_passed = _dedupe_preserve_order(_filter_reoon_results(reoon_result=reoon_result, input_emails=all_emails))

    if not reoon_passed:
        return {
            "threshold": threshold,
            "jd_name": jd_name,
            "rows_with_score_and_linkedin": len(all_candidates),
            "rows_meeting_threshold": len(threshold_candidates),
            "salesql_emails_found": len(all_emails),
            "reoon_passed": 0,
            "bounceban_deliverable": 0,
            "campaign_id": None,
            "campaign_name": None,
            "leads_added": 0,
            "lead_skipped": [],
            "lead_errors": [],
            **_result_source_fields(),
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
            "jd_name": jd_name,
            "rows_with_score_and_linkedin": len(all_candidates),
            "rows_meeting_threshold": len(threshold_candidates),
            "salesql_emails_found": len(all_emails),
            "reoon_passed": len(reoon_passed),
            "bounceban_deliverable": 0,
            "campaign_id": None,
            "campaign_name": None,
            "leads_added": 0,
            "lead_skipped": [],
            "lead_errors": [],
            **_result_source_fields(),
        }

    _update(f"BounceBan deliverable: {len(deliverable)}. Creating Instantly campaign...")
    campaign = _create_instantly_campaign(
        instantly_api_key=instantly_api_key,
        threshold=threshold,
        jd_name=jd_name,
    )
    campaign_id = campaign["id"]
    campaign_name = campaign["name"]

    lead_skipped: List[str] = []
    lead_errors: List[str] = []
    added_count = 0
    fail_fast = _is_truthy(os.getenv("INSTANTLY_FAIL_FAST"), default=False)
    for email in deliverable:
        lead_meta = metadata_by_email.get(email) or {"email": email}
        try:
            add_response = _add_lead_to_instantly_campaign(
                instantly_api_key=instantly_api_key,
                campaign_id=campaign_id,
                lead=lead_meta,
            )
            outcome = _classify_instantly_lead_add_result(campaign_id=campaign_id, response_body=add_response)
            if outcome["status"] == "added":
                added_count += 1
                continue
            if outcome["status"] == "skipped":
                lead_skipped.append(f"{email}: {outcome['detail']}")
                continue
            if fail_fast:
                raise RuntimeError(outcome["detail"])
            lead_errors.append(f"{email}: {outcome['detail']}")
            continue
        except Exception as exc:
            lead_errors.append(f"{email}: {exc}")
            if fail_fast:
                raise

    return {
        "threshold": threshold,
        "jd_name": jd_name,
        "rows_with_score_and_linkedin": len(all_candidates),
        "rows_meeting_threshold": len(threshold_candidates),
        "salesql_emails_found": len(all_emails),
        "reoon_passed": len(reoon_passed),
        "bounceban_deliverable": len(deliverable),
        "campaign_id": campaign_id,
        "campaign_name": campaign_name,
        "leads_added": added_count,
        "lead_skipped": lead_skipped,
        "lead_errors": lead_errors,
        **_result_source_fields(),
    }


# ---------------------------------------------------------------------------
# HeyReach integration
# ---------------------------------------------------------------------------


def _heyreach_headers(api_key: str) -> Dict[str, str]:
    return {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _raise_heyreach_for_status(response: requests.Response, *, action: str) -> None:
    try:
        response.raise_for_status()
        return
    except requests.HTTPError as exc:
        detail = ""
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            detail = str(body.get("errorMessage") or body.get("detail") or body.get("title") or "").strip()
            if not detail:
                detail = str(body).strip()
        if not detail:
            detail = (response.text or "").strip()
        message = f"HeyReach {action} failed ({response.status_code})"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc


def _create_heyreach_list(heyreach_api_key: str, list_name: str) -> Dict[str, Any]:
    response = requests.post(
        HEYREACH_CREATE_LIST_URL,
        headers=_heyreach_headers(heyreach_api_key),
        json={"name": list_name, "type": HEYREACH_USER_LIST_TYPE},
        timeout=60,
    )
    _raise_heyreach_for_status(response, action="list creation")
    body = response.json()
    list_id = body.get("id")
    if not list_id:
        raise RuntimeError(f"HeyReach list creation failed: {body}")
    return {"id": int(list_id), "name": body.get("name", list_name)}


def _add_leads_to_heyreach_list(
    heyreach_api_key: str,
    list_id: int,
    leads: List[Dict[str, str]],
    calendly_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Add up to 1000 leads to a HeyReach list in one request."""
    heyreach_leads: List[Dict[str, Any]] = []
    for lead in leads:
        first_name = lead.get("first_name", "")
        last_name = lead.get("last_name", "")
        entry: Dict[str, Any] = {
            "profileUrl": lead.get("linkedin_url", ""),
            "firstName": first_name,
            "lastName": last_name,
            "companyName": _clean_company_name(lead.get("company_name", "")),
        }
        custom_fields: List[Dict[str, str]] = [
            {"name": "old_first_name", "value": first_name},
            {"name": "old_last_name", "value": last_name},
        ]
        personalization = _build_heyreach_personalization(lead)
        if personalization:
            custom_fields.append({"name": "personalization", "value": personalization})
        if calendly_url:
            custom_fields.append({"name": "Calendly", "value": calendly_url})
        entry["customUserFields"] = custom_fields
        heyreach_leads.append(entry)

    response = requests.post(
        HEYREACH_ADD_LEADS_URL,
        headers=_heyreach_headers(heyreach_api_key),
        json={"listId": list_id, "leads": heyreach_leads},
        timeout=120,
    )
    _raise_heyreach_for_status(response, action="lead import")
    return response.json()


def _build_heyreach_personalization(lead: Dict[str, str]) -> str:
    generated_email = (lead.get("generated_email") or "").strip()
    snippet = _clean_personalization_snippet(generated_email)
    if snippet:
        return snippet
    linkedin_url = lead.get("linkedin_url", "")
    if linkedin_url:
        return f"LinkedIn: {linkedin_url}"
    return ""


def _build_heyreach_list_name(*, jd_name: str, threshold: float, start_date: date) -> str:
    clean_jd_name = (jd_name or "").strip()
    if clean_jd_name:
        return clean_jd_name
    return f"JIT HeyReach Threshold {threshold:g} ({start_date.isoformat()})"


def run_thread_reply_heyreach_pipeline(
    *,
    slack_token: str,
    channel_id: str,
    thread_ts: str,
    threshold: float,
    post_updates: bool = True,
    calendly_url: Optional[str] = None,
) -> Dict[str, Any]:
    """HeyReach variant of the thread-reply pipeline.

    Skips email fetch+verify (SaleSQL, Reoon, BounceBan).  Instead it creates
    a HeyReach list and adds matching candidates directly by LinkedIn URL.
    """
    heyreach_api_key = _require_env("HEYREACH_API_KEY")

    def _update(text: str) -> None:
        if post_updates:
            _slack_post_message(slack_token=slack_token, channel_id=channel_id, thread_ts=thread_ts, text=text)

    root_message = _get_thread_root_message(slack_token=slack_token, channel_id=channel_id, thread_ts=thread_ts)
    sheet_url = _extract_google_sheet_url_from_message(root_message)
    jd_name = _extract_jd_name_from_message(root_message)
    source_type = "google_sheet"
    source_name = sheet_url or "unknown"
    source_url = sheet_url if sheet_url else None

    def _result_source_fields() -> Dict[str, Any]:
        return {
            "source_type": source_type,
            "source_name": source_name,
            "source_url": source_url,
        }

    if not _is_result_message_thread_root(root_message, sheet_url):
        return {
            "ignored": "not_result_message_thread",
            "threshold": threshold,
            "jd_name": jd_name,
            "heyreach_list_id": None,
            "heyreach_list_name": None,
            "leads_added": 0,
            "lead_errors": [],
            **_result_source_fields(),
        }

    if not sheet_url:
        raise RuntimeError("Thread root message must include a Google Sheet link.")
    sheet_payload = load_rows_from_google_sheet_url(sheet_url)
    source_name = (sheet_payload.get("spreadsheet_title") or source_name).strip()
    source_url = sheet_payload.get("spreadsheet_url") or sheet_url
    all_candidates = _load_candidates_from_rows(sheet_payload.get("rows") or [])

    threshold_candidates = [c for c in all_candidates if c["score"] >= threshold]
    threshold_candidates = [c for c in threshold_candidates if c["linkedin_url"]]

    if not threshold_candidates:
        return {
            "threshold": threshold,
            "jd_name": jd_name,
            "rows_with_score_and_linkedin": len(all_candidates),
            "rows_meeting_threshold": 0,
            "heyreach_list_id": None,
            "heyreach_list_name": None,
            "leads_added": 0,
            "lead_errors": [],
            **_result_source_fields(),
        }

    _update(
        f"Threshold {threshold:g} accepted {len(threshold_candidates)} candidates. "
        "Creating HeyReach list and adding leads..."
    )

    list_name = _build_heyreach_list_name(jd_name=jd_name, threshold=threshold, start_date=date.today())
    heyreach_list = _create_heyreach_list(heyreach_api_key=heyreach_api_key, list_name=list_name)
    list_id = heyreach_list["id"]

    # HeyReach API accepts a maximum of 100 leads per request.
    batch_size = 100
    total_added = 0
    total_updated = 0
    total_failed = 0
    lead_errors: List[str] = []
    for i in range(0, len(threshold_candidates), batch_size):
        batch = threshold_candidates[i : i + batch_size]
        try:
            result = _add_leads_to_heyreach_list(
                heyreach_api_key=heyreach_api_key,
                list_id=list_id,
                leads=batch,
                calendly_url=calendly_url,
            )
            total_added += result.get("addedLeadsCount", 0)
            total_updated += result.get("updatedLeadsCount", 0)
            total_failed += result.get("failedLeadsCount", 0)
        except Exception as exc:
            lead_errors.append(f"batch {i // batch_size}: {exc}")

    return {
        "threshold": threshold,
        "jd_name": jd_name,
        "rows_with_score_and_linkedin": len(all_candidates),
        "rows_meeting_threshold": len(threshold_candidates),
        "heyreach_list_id": list_id,
        "heyreach_list_name": heyreach_list["name"],
        "leads_added": total_added,
        "leads_updated": total_updated,
        "leads_failed": total_failed,
        "lead_errors": lead_errors,
        **_result_source_fields(),
    }
