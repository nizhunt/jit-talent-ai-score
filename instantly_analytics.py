"""Fetch Instantly campaign analytics and write metrics to the dashboard sheet."""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from dashboard_logger import (
    COL_INSTANTLY_CAMPAIGN_NAME,
    COL_SCORED_SHEET_URL,
    _dashboard_sheet_url,
    _dashboard_worksheet_name,
    _now_utc_label,
)
from google_sheets import (
    extract_spreadsheet_id_from_url,
    upsert_row_in_google_sheet_url,
    _build_google_clients,
    _get_spreadsheet_metadata,
    _escape_tab_name,
)

logger = logging.getLogger(__name__)

INSTANTLY_ANALYTICS_URL = "https://api.instantly.ai/api/v2/campaigns/analytics"

# New dashboard columns for Instantly analytics
COL_INST_EMAILS_SENT = "Emails Sent (Instantly)"
COL_INST_LEADS_CONTACTED = "Leads Contacted (Instantly)"
COL_INST_NEW_LEADS_CONTACTED = "New Leads Contacted (Instantly)"
COL_INST_REPLIES = "Replies (Instantly)"
COL_INST_REPLIES_UNIQUE = "Unique Replies (Instantly)"
COL_INST_BOUNCED = "Bounced (Instantly)"
COL_INST_UNSUBSCRIBED = "Unsubscribed (Instantly)"
COL_INST_OPPORTUNITIES = "Opportunities (Instantly)"
COL_INST_OPPORTUNITY_VALUE = "Opportunity Value (Instantly)"
COL_INST_ANALYTICS_UPDATED = "Analytics Updated (UTC)"

ANALYTICS_COLUMNS = [
    COL_INST_EMAILS_SENT,
    COL_INST_LEADS_CONTACTED,
    COL_INST_NEW_LEADS_CONTACTED,
    COL_INST_REPLIES,
    COL_INST_REPLIES_UNIQUE,
    COL_INST_BOUNCED,
    COL_INST_UNSUBSCRIBED,
    COL_INST_OPPORTUNITIES,
    COL_INST_OPPORTUNITY_VALUE,
    COL_INST_ANALYTICS_UPDATED,
]


def _get_instantly_api_key() -> str:
    key = (os.getenv("INSTANTLY_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing required env var: INSTANTLY_API_KEY")
    return key


def fetch_all_campaign_analytics(api_key: str) -> List[Dict[str, Any]]:
    """Fetch analytics for all campaigns from Instantly API v2."""
    resp = requests.get(
        INSTANTLY_ANALYTICS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Instantly analytics response type: {type(data)}")
    return data


def _read_dashboard_rows() -> List[Dict[str, Any]]:
    """Read all rows from the dashboard sheet."""
    sheet_url = _dashboard_sheet_url()
    if not sheet_url:
        raise RuntimeError("DASHBOARD_GOOGLE_SHEET_URL is not configured.")

    worksheet_name = _dashboard_worksheet_name()
    spreadsheet_id = extract_spreadsheet_id_from_url(sheet_url)
    clients = _build_google_clients()
    sheets = clients["sheets"]

    metadata = _get_spreadsheet_metadata(sheets=sheets, spreadsheet_id=spreadsheet_id)
    tab_titles = metadata["tab_titles"]

    tab_name = worksheet_name if worksheet_name in tab_titles else tab_titles[0]
    escaped_tab_name = _escape_tab_name(tab_name)

    values_resp = (
        sheets.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=f"'{escaped_tab_name}'")
        .execute()
    )
    values = values_resp.get("values") or []
    if not values:
        return []

    headers = [str(cell).strip() for cell in values[0]]
    rows: List[Dict[str, Any]] = []
    for raw in values[1:]:
        row: Dict[str, Any] = {}
        for idx, key in enumerate(headers):
            if not key:
                continue
            row[key] = raw[idx] if idx < len(raw) else ""
        rows.append(row)
    return rows


def _build_analytics_row(
    campaign_name: str,
    scored_sheet_url: str,
    analytics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a row dict with analytics columns for upsert."""
    return {
        COL_SCORED_SHEET_URL: scored_sheet_url,
        COL_INST_EMAILS_SENT: analytics.get("emails_sent_count", 0),
        COL_INST_LEADS_CONTACTED: analytics.get("contacted_count", 0),
        COL_INST_NEW_LEADS_CONTACTED: analytics.get("new_leads_contacted_count", 0),
        COL_INST_REPLIES: analytics.get("reply_count", 0),
        COL_INST_REPLIES_UNIQUE: analytics.get("reply_count_unique", 0),
        COL_INST_BOUNCED: analytics.get("bounced_count", 0),
        COL_INST_UNSUBSCRIBED: analytics.get("unsubscribed_count", 0),
        COL_INST_OPPORTUNITIES: analytics.get("total_opportunities", 0),
        COL_INST_OPPORTUNITY_VALUE: analytics.get("total_opportunity_value", 0),
        COL_INST_ANALYTICS_UPDATED: _now_utc_label(),
    }


def sync_instantly_analytics_to_dashboard() -> Dict[str, Any]:
    """Main entry point: fetch Instantly analytics and update matching dashboard rows.

    Returns a summary dict with counts of matched/updated/skipped rows.
    """
    api_key = _get_instantly_api_key()
    sheet_url = _dashboard_sheet_url()
    if not sheet_url:
        return {"ok": False, "error": "DASHBOARD_GOOGLE_SHEET_URL not configured"}

    worksheet_name = _dashboard_worksheet_name()

    # 1. Read dashboard rows to find which have Instantly campaign names
    logger.info("Reading dashboard rows from Google Sheet...")
    dashboard_rows = _read_dashboard_rows()
    logger.info("Found %d dashboard rows", len(dashboard_rows))

    # Build a map of campaign_name -> scored_sheet_url for rows that have both
    campaign_to_sheet: Dict[str, str] = {}
    for row in dashboard_rows:
        campaign_name = (row.get(COL_INSTANTLY_CAMPAIGN_NAME) or "").strip()
        scored_url = (row.get(COL_SCORED_SHEET_URL) or "").strip()
        if campaign_name and scored_url:
            campaign_to_sheet[campaign_name] = scored_url

    if not campaign_to_sheet:
        logger.info("No dashboard rows with Instantly campaign names found.")
        return {"ok": True, "dashboard_rows": len(dashboard_rows), "matched": 0, "updated": 0}

    logger.info(
        "Found %d dashboard rows with Instantly campaign names", len(campaign_to_sheet)
    )

    # 2. Fetch all campaign analytics from Instantly
    logger.info("Fetching Instantly campaign analytics...")
    all_analytics = fetch_all_campaign_analytics(api_key)
    logger.info("Fetched analytics for %d campaigns", len(all_analytics))

    # Build lookup by campaign name
    analytics_by_name: Dict[str, Dict[str, Any]] = {}
    for entry in all_analytics:
        name = (entry.get("campaign_name") or "").strip()
        if name:
            analytics_by_name[name] = entry

    # 3. Match and update
    matched = 0
    updated = 0
    skipped = 0
    errors: List[str] = []

    for campaign_name, scored_url in campaign_to_sheet.items():
        analytics = analytics_by_name.get(campaign_name)
        if not analytics:
            logger.debug("No analytics found for campaign: %s", campaign_name)
            skipped += 1
            continue

        matched += 1
        row = _build_analytics_row(campaign_name, scored_url, analytics)

        try:
            upsert_row_in_google_sheet_url(
                sheet_url=sheet_url,
                row=row,
                key_column=COL_SCORED_SHEET_URL,
                worksheet_title=worksheet_name or None,
                ignore_blank_values=True,
            )
            updated += 1
            logger.info(
                "Updated analytics for %s: sent=%s replies=%s bounced=%s opps=%s",
                campaign_name,
                analytics.get("emails_sent_count", 0),
                analytics.get("reply_count", 0),
                analytics.get("bounced_count", 0),
                analytics.get("total_opportunities", 0),
            )
        except Exception as exc:
            error_msg = f"Failed to update {campaign_name}: {exc}"
            logger.error(error_msg)
            errors.append(error_msg)

    result = {
        "ok": True,
        "dashboard_rows": len(dashboard_rows),
        "campaigns_with_name": len(campaign_to_sheet),
        "instantly_campaigns": len(all_analytics),
        "matched": matched,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info("Analytics sync complete: %s", result)
    return result
