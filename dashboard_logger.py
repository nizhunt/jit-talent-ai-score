import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from google_sheets import upsert_row_in_google_sheet_url


DASHBOARD_GOOGLE_SHEET_URL_ENV = "DASHBOARD_GOOGLE_SHEET_URL"
DASHBOARD_WORKSHEET_NAME_ENV = "DASHBOARD_WORKSHEET_NAME"
DEFAULT_DASHBOARD_WORKSHEET_NAME = "info-log"

COL_LAST_UPDATED_UTC = "Last Updated (UTC)"
COL_JD_CAMPAIGN_NAME = "JD Campaign Name"
COL_SCORED_SHEET_URL = "Scored Candidates Sheet URL"
COL_TOTAL_PROFILES_FOUND = "Total Profiles Found"
COL_PROFILES_AFTER_DEDUP = "Profiles After Deduplication"
COL_LOCATION_MISMATCH_MANUAL = "Location Mismatch (Manual Review)"
COL_MINIMUM_SCORE_FOR_CONTACT = "Minimum Score for Contact"
COL_CANDIDATES_ENTERED_ENRICHMENT = "Candidates Entered Enrichment"
COL_EMAILS_FOUND_SALESQL = "Emails Found via SaleSQL"
COL_EMAILS_PASSED_REOON = "Emails Passed Reoon"
COL_EMAILS_PASSED_BOUNCEBAN = "Emails Passed BounceBan"
COL_PRE_EXIST_IN_INSTANTLY = "pre-exist in instantly"
COL_NET_LEADS_ENROLLED_INSTANTLY = "Net Leads Enrolled to Instantly"
COL_INSTANTLY_CAMPAIGN_NAME = "Instantly Campaign Name"
COL_INSTANTLY_CAMPAIGN_URL = "Instantly Campaign URL"
COL_NOTES = "Notes"

SCORE_COLUMNS = {score: f"Score {score} Count" for score in range(10, -1, -1)}

DASHBOARD_COLUMNS = [
    COL_LAST_UPDATED_UTC,
    COL_JD_CAMPAIGN_NAME,
    COL_SCORED_SHEET_URL,
    COL_TOTAL_PROFILES_FOUND,
    COL_PROFILES_AFTER_DEDUP,
    COL_LOCATION_MISMATCH_MANUAL,
    *SCORE_COLUMNS.values(),
    COL_MINIMUM_SCORE_FOR_CONTACT,
    COL_CANDIDATES_ENTERED_ENRICHMENT,
    COL_EMAILS_FOUND_SALESQL,
    COL_EMAILS_PASSED_REOON,
    COL_EMAILS_PASSED_BOUNCEBAN,
    COL_PRE_EXIST_IN_INSTANTLY,
    COL_NET_LEADS_ENROLLED_INSTANTLY,
    COL_INSTANTLY_CAMPAIGN_NAME,
    COL_INSTANTLY_CAMPAIGN_URL,
    COL_NOTES,
]


def _dashboard_sheet_url() -> str:
    return (os.getenv(DASHBOARD_GOOGLE_SHEET_URL_ENV) or "").strip()


def _dashboard_worksheet_name() -> str:
    configured = (os.getenv(DASHBOARD_WORKSHEET_NAME_ENV) or "").strip()
    if configured:
        return configured
    return DEFAULT_DASHBOARD_WORKSHEET_NAME


def _now_utc_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _blank_dashboard_row() -> Dict[str, Any]:
    return {column: "" for column in DASHBOARD_COLUMNS}


def _normalize_score_counts(score_counts_by_score: Optional[Dict[Any, Any]]) -> Dict[int, int]:
    normalized = {score: 0 for score in range(0, 11)}
    if not score_counts_by_score:
        return normalized
    for raw_score, raw_count in score_counts_by_score.items():
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            continue
        if score < 0 or score > 10:
            continue
        normalized[score] = _safe_int(raw_count)
    return normalized


def _build_instantly_campaign_url(campaign_id: str) -> str:
    clean = (campaign_id or "").strip()
    if not clean:
        return ""
    return f"https://app.instantly.ai/app/campaign/{clean}/analytics"


def _upsert_dashboard_row(row: Dict[str, Any]) -> bool:
    sheet_url = _dashboard_sheet_url()
    if not sheet_url:
        return False

    worksheet_name = _dashboard_worksheet_name()
    upsert_row_in_google_sheet_url(
        sheet_url=sheet_url,
        row=row,
        key_column=COL_SCORED_SHEET_URL,
        worksheet_title=worksheet_name or None,
        ignore_blank_values=True,
    )
    return True


def log_jd_processing_dashboard_row(
    *,
    jd_name: str,
    candidate_sheet_url: str,
    total_profiles_found: int,
    profiles_after_dedup: int,
    score_counts_by_score: Optional[Dict[Any, Any]] = None,
) -> bool:
    clean_sheet_url = (candidate_sheet_url or "").strip()
    if not clean_sheet_url:
        return False

    score_counts = _normalize_score_counts(score_counts_by_score)
    row = _blank_dashboard_row()
    row[COL_LAST_UPDATED_UTC] = _now_utc_label()
    row[COL_JD_CAMPAIGN_NAME] = (jd_name or "").strip()
    row[COL_SCORED_SHEET_URL] = clean_sheet_url
    row[COL_TOTAL_PROFILES_FOUND] = _safe_int(total_profiles_found)
    row[COL_PROFILES_AFTER_DEDUP] = _safe_int(profiles_after_dedup)
    for score, column in SCORE_COLUMNS.items():
        row[column] = score_counts.get(score, 0)

    return _upsert_dashboard_row(row)


def log_enrichment_dashboard_row(
    *,
    jd_name: str,
    candidate_sheet_url: str,
    minimum_score_for_contact: float,
    candidates_entered_enrichment: int,
    emails_found_salesql: int,
    emails_passed_reoon: int,
    emails_passed_bounceban: int,
    pre_exist_in_instantly: int,
    net_leads_enrolled_instantly: int,
    instantly_campaign_name: str = "",
    instantly_campaign_id: str = "",
    instantly_campaign_url: str = "",
    notes: str = "",
) -> bool:
    clean_sheet_url = (candidate_sheet_url or "").strip()
    if not clean_sheet_url:
        return False

    row = _blank_dashboard_row()
    row[COL_LAST_UPDATED_UTC] = _now_utc_label()
    row[COL_JD_CAMPAIGN_NAME] = (jd_name or "").strip()
    row[COL_SCORED_SHEET_URL] = clean_sheet_url
    row[COL_MINIMUM_SCORE_FOR_CONTACT] = minimum_score_for_contact
    row[COL_CANDIDATES_ENTERED_ENRICHMENT] = _safe_int(candidates_entered_enrichment)
    row[COL_EMAILS_FOUND_SALESQL] = _safe_int(emails_found_salesql)
    row[COL_EMAILS_PASSED_REOON] = _safe_int(emails_passed_reoon)
    row[COL_EMAILS_PASSED_BOUNCEBAN] = _safe_int(emails_passed_bounceban)
    row[COL_PRE_EXIST_IN_INSTANTLY] = _safe_int(pre_exist_in_instantly)
    row[COL_NET_LEADS_ENROLLED_INSTANTLY] = _safe_int(net_leads_enrolled_instantly)
    row[COL_INSTANTLY_CAMPAIGN_NAME] = (instantly_campaign_name or "").strip()
    row[COL_INSTANTLY_CAMPAIGN_URL] = (instantly_campaign_url or "").strip() or _build_instantly_campaign_url(
        instantly_campaign_id
    )
    row[COL_NOTES] = (notes or "").strip()

    return _upsert_dashboard_row(row)
