import base64
import json
import os
import re
from typing import Any, Dict, List

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


GOOGLE_SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
GOOGLE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
GOOGLE_SHEET_URL_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _load_service_account_info() -> Dict[str, Any]:
    raw = _require_env("GOOGLE_SERVICE_ACCOUNT_JSON").strip()
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is empty.")

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        parsed = json.loads(decoded)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON must be raw JSON or base64-encoded JSON."
        ) from exc

    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON did not decode to a JSON object.")


def _build_google_clients() -> Dict[str, Any]:
    info = _load_service_account_info()
    creds = Credentials.from_service_account_info(
        info,
        scopes=[GOOGLE_SHEETS_SCOPE, GOOGLE_DRIVE_SCOPE],
    )
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    return {"sheets": sheets, "drive": drive}


def _sanitize_google_title(value: str, default: str, max_len: int) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "")).strip()
    if not cleaned:
        cleaned = default
    return cleaned[:max_len]


def _to_sheet_values(df: pd.DataFrame) -> List[List[Any]]:
    frame = df.astype(object).where(pd.notna(df), "")
    rows: List[List[Any]] = [list(map(str, frame.columns.tolist()))]
    for row in frame.itertuples(index=False, name=None):
        normalized: List[Any] = []
        for cell in row:
            if cell is None:
                normalized.append("")
            elif isinstance(cell, (int, float, bool)):
                normalized.append(cell)
            else:
                normalized.append(str(cell))
        rows.append(normalized)
    return rows


def _move_file_to_folder(drive: Any, file_id: str, folder_id: str) -> None:
    metadata = (
        drive.files()
        .get(fileId=file_id, fields="parents", supportsAllDrives=True)
        .execute()
    )
    parents = metadata.get("parents") or []
    remove_parents = ",".join(parents) if parents else None

    kwargs: Dict[str, Any] = {
        "fileId": file_id,
        "addParents": folder_id,
        "supportsAllDrives": True,
        "fields": "id,parents",
    }
    if remove_parents:
        kwargs["removeParents"] = remove_parents
    drive.files().update(**kwargs).execute()


def _ensure_domain_permission(drive: Any, file_id: str, domain: str, role: str) -> None:
    permission = {
        "type": "domain",
        "role": role,
        "domain": domain,
        "allowFileDiscovery": False,
    }
    try:
        (
            drive.permissions()
            .create(
                fileId=file_id,
                body=permission,
                supportsAllDrives=True,
                sendNotificationEmail=False,
                fields="id",
            )
            .execute()
        )
    except HttpError as exc:
        # Drive can return 409 when the same permission already exists.
        status_code = getattr(getattr(exc, "resp", None), "status", None) or getattr(exc, "status_code", None)
        if status_code == 409 or "already" in str(exc).lower():
            return
        raise


def create_google_sheet_from_dataframe(
    *,
    df: pd.DataFrame,
    spreadsheet_title: str,
    folder_id: str,
    workspace_domain: str,
    domain_role: str = "writer",
    worksheet_title: str = "Candidates",
) -> Dict[str, str]:
    clients = _build_google_clients()
    sheets = clients["sheets"]
    drive = clients["drive"]

    title = _sanitize_google_title(spreadsheet_title, default="Scored Candidates", max_len=200)
    tab_name = _sanitize_google_title(worksheet_title, default="Candidates", max_len=100)
    body = {"properties": {"title": title}, "sheets": [{"properties": {"title": tab_name}}]}

    created = (
        sheets.spreadsheets()
        .create(body=body, fields="spreadsheetId,spreadsheetUrl,properties/title")
        .execute()
    )
    spreadsheet_id = created["spreadsheetId"]
    spreadsheet_url = created.get("spreadsheetUrl") or f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"

    values = _to_sheet_values(df)
    (
        sheets.spreadsheets()
        .values()
        .update(
            spreadsheetId=spreadsheet_id,
            range=f"'{tab_name}'!A1",
            valueInputOption="RAW",
            body={"values": values},
        )
        .execute()
    )

    _move_file_to_folder(drive=drive, file_id=spreadsheet_id, folder_id=folder_id)
    _ensure_domain_permission(
        drive=drive,
        file_id=spreadsheet_id,
        domain=workspace_domain,
        role=domain_role,
    )

    return {
        "spreadsheet_id": spreadsheet_id,
        "spreadsheet_url": spreadsheet_url,
        "spreadsheet_title": title,
    }


def extract_spreadsheet_id_from_url(url: str) -> str:
    match = GOOGLE_SHEET_URL_RE.search((url or "").strip())
    if not match:
        raise RuntimeError("Google Sheet URL is invalid or missing spreadsheet id.")
    return match.group(1)


def load_rows_from_google_sheet_url(sheet_url: str) -> Dict[str, Any]:
    spreadsheet_id = extract_spreadsheet_id_from_url(sheet_url)
    clients = _build_google_clients()
    sheets = clients["sheets"]

    metadata = (
        sheets.spreadsheets()
        .get(spreadsheetId=spreadsheet_id, fields="properties/title,sheets/properties/title")
        .execute()
    )
    spreadsheet_title = ((metadata.get("properties") or {}).get("title") or "").strip()
    sheets_meta = metadata.get("sheets") or []
    if not sheets_meta:
        raise RuntimeError("Google Sheet has no worksheets.")
    first_tab = ((sheets_meta[0] or {}).get("properties") or {}).get("title") or "Candidates"
    escaped_tab_name = first_tab.replace("'", "''")
    range_name = f"'{escaped_tab_name}'"

    values_resp = (
        sheets.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_name)
        .execute()
    )
    values = values_resp.get("values") or []
    if not values:
        return {
            "spreadsheet_id": spreadsheet_id,
            "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}",
            "spreadsheet_title": spreadsheet_title or spreadsheet_id,
            "rows": [],
        }

    headers = [str(cell).strip() for cell in values[0]]
    rows: List[Dict[str, Any]] = []
    for raw in values[1:]:
        row: Dict[str, Any] = {}
        for idx, key in enumerate(headers):
            if not key:
                continue
            row[key] = raw[idx] if idx < len(raw) else ""
        rows.append(row)

    return {
        "spreadsheet_id": spreadsheet_id,
        "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}",
        "spreadsheet_title": spreadsheet_title or spreadsheet_id,
        "rows": rows,
    }
