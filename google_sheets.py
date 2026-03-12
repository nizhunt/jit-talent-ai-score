import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

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


def _extract_google_http_error_message(exc: HttpError) -> str:
    raw_content = getattr(exc, "content", None)
    if raw_content:
        try:
            decoded = raw_content.decode("utf-8", errors="replace") if isinstance(raw_content, (bytes, bytearray)) else str(raw_content)
            parsed = json.loads(decoded)
            message = ((parsed.get("error") or {}).get("message") or "").strip()
            if message:
                return message
            decoded = decoded.strip()
            if decoded:
                return decoded
        except Exception:
            pass
    return str(exc)


def _raise_google_http_error(action: str, exc: HttpError) -> None:
    status_code = getattr(getattr(exc, "resp", None), "status", None) or getattr(exc, "status_code", None)
    detail = _extract_google_http_error_message(exc)
    detail_lower = detail.lower() if detail else ""
    parts = [f"Google API failed while trying to {action}."]
    if status_code:
        parts.append(f"HTTP {status_code}.")
    if detail:
        parts.append(detail)
    if "storagequotaexceeded" in detail_lower or "storage quota has been exceeded" in detail_lower:
        parts.append(
            "Drive quota is full for the file owner context. "
            "Use a Shared Drive folder in GOOGLE_DRIVE_FOLDER_ID, or free storage for the owning account."
        )
    if status_code == 403:
        parts.append(
            "Verify GOOGLE_SERVICE_ACCOUNT_JSON has Sheets/Drive API access, "
            "and that the account can create files and share inside your Google Workspace."
        )
    raise RuntimeError(" ".join(parts)) from exc


def _sanitize_google_title(value: str, default: str, max_len: int) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "")).strip()
    if not cleaned:
        cleaned = default
    return cleaned[:max_len]


def _escape_tab_name(tab_name: str) -> str:
    return (tab_name or "").replace("'", "''")


def _column_index_to_letters(index: int) -> str:
    if index <= 0:
        raise ValueError("Column index must be >= 1")
    letters = ""
    num = index
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _normalize_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return value
    return str(value)


def _get_spreadsheet_metadata(sheets: Any, spreadsheet_id: str) -> Dict[str, Any]:
    metadata = (
        sheets.spreadsheets()
        .get(spreadsheetId=spreadsheet_id, fields="properties/title,sheets/properties/title")
        .execute()
    )
    spreadsheet_title = ((metadata.get("properties") or {}).get("title") or "").strip()
    sheets_meta = metadata.get("sheets") or []
    if not sheets_meta:
        raise RuntimeError("Google Sheet has no worksheets.")
    tab_titles: List[str] = []
    for item in sheets_meta:
        title = ((item or {}).get("properties") or {}).get("title")
        if title:
            tab_titles.append(str(title))
    if not tab_titles:
        raise RuntimeError("Google Sheet has no worksheets.")
    return {
        "spreadsheet_title": spreadsheet_title,
        "tab_titles": tab_titles,
    }


def _rename_single_worksheet_if_needed(
    *,
    sheets: Any,
    spreadsheet_id: str,
    target_title: str,
) -> None:
    metadata = (
        sheets.spreadsheets()
        .get(spreadsheetId=spreadsheet_id, fields="sheets/properties(sheetId,title)")
        .execute()
    )
    sheets_meta = metadata.get("sheets") or []
    if len(sheets_meta) != 1:
        return

    props = (sheets_meta[0] or {}).get("properties") or {}
    sheet_id = props.get("sheetId")
    current_title = str(props.get("title") or "").strip()
    desired_title = _sanitize_google_title(target_title, default="Candidates", max_len=100)
    if not sheet_id or not desired_title or current_title == desired_title:
        return

    (
        sheets.spreadsheets()
        .batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                "requests": [
                    {
                        "updateSheetProperties": {
                            "properties": {"sheetId": int(sheet_id), "title": desired_title},
                            "fields": "title",
                        }
                    }
                ]
            },
        )
        .execute()
    )


def _ensure_worksheet_title(
    *,
    sheets: Any,
    spreadsheet_id: str,
    requested_title: Optional[str],
    available_titles: List[str],
) -> str:
    if requested_title:
        requested_clean = _sanitize_google_title(requested_title, default="Dashboard", max_len=100)
        if requested_clean in available_titles:
            return requested_clean
        (
            sheets.spreadsheets()
            .batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "requests": [
                        {
                            "addSheet": {
                                "properties": {
                                    "title": requested_clean,
                                }
                            }
                        }
                    ]
                },
            )
            .execute()
        )
        return requested_clean
    return available_titles[0]


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
    created_sheet = create_google_sheet_placeholder(
        spreadsheet_title=spreadsheet_title,
        folder_id=folder_id,
        workspace_domain=workspace_domain,
        domain_role=domain_role,
        worksheet_title=worksheet_title,
    )
    write_result = write_dataframe_to_google_sheet(
        spreadsheet_id=created_sheet["spreadsheet_id"],
        df=df,
        worksheet_title=worksheet_title,
    )
    created_sheet["spreadsheet_title"] = write_result.get("spreadsheet_title") or created_sheet["spreadsheet_title"]
    created_sheet["worksheet_title"] = write_result.get("worksheet_title") or worksheet_title
    return created_sheet


def create_google_sheet_placeholder(
    *,
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

    try:
        created = (
            drive.files()
            .create(
                body={
                    "name": title,
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                    "parents": [folder_id],
                },
                supportsAllDrives=True,
                fields="id,name,webViewLink",
            )
            .execute()
        )
    except HttpError as exc:
        _raise_google_http_error("create the result spreadsheet", exc)
    spreadsheet_id = created["id"]
    spreadsheet_url = created.get("webViewLink") or f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"

    try:
        _rename_single_worksheet_if_needed(
            sheets=sheets,
            spreadsheet_id=spreadsheet_id,
            target_title=tab_name,
        )
    except HttpError as exc:
        _raise_google_http_error(f"rename initial worksheet to '{tab_name}'", exc)

    try:
        _ensure_domain_permission(
            drive=drive,
            file_id=spreadsheet_id,
            domain=workspace_domain,
            role=domain_role,
        )
    except HttpError as exc:
        _raise_google_http_error(f"share spreadsheet with domain '{workspace_domain}' as '{domain_role}'", exc)

    return {
        "spreadsheet_id": spreadsheet_id,
        "spreadsheet_url": spreadsheet_url,
        "spreadsheet_title": title,
        "worksheet_title": tab_name,
    }


def write_dataframe_to_google_sheet(
    *,
    spreadsheet_id: str,
    df: pd.DataFrame,
    worksheet_title: str = "Candidates",
) -> Dict[str, str]:
    clients = _build_google_clients()
    sheets = clients["sheets"]

    try:
        metadata = _get_spreadsheet_metadata(sheets=sheets, spreadsheet_id=spreadsheet_id)
    except HttpError as exc:
        _raise_google_http_error(f"read spreadsheet metadata for '{spreadsheet_id}'", exc)

    try:
        tab_name = _ensure_worksheet_title(
            sheets=sheets,
            spreadsheet_id=spreadsheet_id,
            requested_title=worksheet_title,
            available_titles=metadata["tab_titles"],
        )
    except HttpError as exc:
        _raise_google_http_error(f"ensure worksheet '{worksheet_title}' exists", exc)
    values = _to_sheet_values(df)
    try:
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
    except HttpError as exc:
        _raise_google_http_error(f"write candidate rows into spreadsheet '{spreadsheet_id}'", exc)

    return {
        "spreadsheet_id": spreadsheet_id,
        "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}",
        "spreadsheet_title": metadata.get("spreadsheet_title") or spreadsheet_id,
        "worksheet_title": tab_name,
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

    metadata = _get_spreadsheet_metadata(sheets=sheets, spreadsheet_id=spreadsheet_id)
    spreadsheet_title = metadata["spreadsheet_title"]
    tab_titles = metadata["tab_titles"]
    selected_tab = "Candidates" if "Candidates" in tab_titles else tab_titles[0]
    escaped_tab_name = _escape_tab_name(selected_tab)
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
            "worksheet_title": selected_tab,
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
        "worksheet_title": selected_tab,
        "rows": rows,
    }


def upsert_row_in_google_sheet_url(
    *,
    sheet_url: str,
    row: Dict[str, Any],
    key_column: str,
    worksheet_title: Optional[str] = None,
    ignore_blank_values: bool = False,
) -> Dict[str, Any]:
    key_name = (key_column or "").strip()
    if not key_name:
        raise RuntimeError("key_column is required for upsert.")

    if key_name not in row:
        raise RuntimeError(f"key column '{key_name}' missing from row payload.")
    key_value = str(row.get(key_name, "")).strip()
    if not key_value:
        raise RuntimeError(f"key column '{key_name}' cannot be blank.")

    spreadsheet_id = extract_spreadsheet_id_from_url(sheet_url)
    clients = _build_google_clients()
    sheets = clients["sheets"]

    metadata = _get_spreadsheet_metadata(sheets=sheets, spreadsheet_id=spreadsheet_id)
    spreadsheet_title = metadata["spreadsheet_title"]
    tab_name = _ensure_worksheet_title(
        sheets=sheets,
        spreadsheet_id=spreadsheet_id,
        requested_title=worksheet_title,
        available_titles=metadata["tab_titles"],
    )
    escaped_tab_name = _escape_tab_name(tab_name)
    full_tab_range = f"'{escaped_tab_name}'"

    existing_values_resp = (
        sheets.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=full_tab_range)
        .execute()
    )
    existing_values = existing_values_resp.get("values") or []
    headers: List[str] = []
    if existing_values:
        headers = [str(cell).strip() for cell in existing_values[0]]

    header_changed = False
    for column in row.keys():
        col_name = str(column).strip()
        if not col_name:
            continue
        if col_name not in headers:
            headers.append(col_name)
            header_changed = True

    if key_name not in headers:
        headers.append(key_name)
        header_changed = True

    if not existing_values or header_changed:
        header_range = f"'{escaped_tab_name}'!A1:{_column_index_to_letters(len(headers))}1"
        (
            sheets.spreadsheets()
            .values()
            .update(
                spreadsheetId=spreadsheet_id,
                range=header_range,
                valueInputOption="RAW",
                body={"values": [headers]},
            )
            .execute()
        )
        if existing_values:
            existing_values[0] = headers
        else:
            existing_values = [headers]

    key_col_idx = headers.index(key_name)
    data_rows = existing_values[1:] if len(existing_values) > 1 else []
    target_row_number: Optional[int] = None
    for idx, existing_row in enumerate(data_rows, start=2):
        existing_key = str(existing_row[key_col_idx]).strip() if key_col_idx < len(existing_row) else ""
        if existing_key == key_value:
            target_row_number = idx

    row_was_created = target_row_number is None
    if row_was_created:
        target_row_number = len(data_rows) + 2

    data_updates: List[Dict[str, Any]] = []
    for col_name, raw_value in row.items():
        clean_col = str(col_name).strip()
        if not clean_col:
            continue
        if clean_col not in headers:
            continue
        normalized = _normalize_cell_value(raw_value)
        if ignore_blank_values and normalized == "":
            continue
        col_idx = headers.index(clean_col) + 1
        col_letter = _column_index_to_letters(col_idx)
        data_updates.append(
            {
                "range": f"'{escaped_tab_name}'!{col_letter}{target_row_number}",
                "values": [[normalized]],
            }
        )

    if data_updates:
        (
            sheets.spreadsheets()
            .values()
            .batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "valueInputOption": "RAW",
                    "data": data_updates,
                },
            )
            .execute()
        )

    return {
        "spreadsheet_id": spreadsheet_id,
        "spreadsheet_url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}",
        "spreadsheet_title": spreadsheet_title or spreadsheet_id,
        "worksheet_title": tab_name,
        "row_number": target_row_number,
        "row_was_created": row_was_created,
    }
