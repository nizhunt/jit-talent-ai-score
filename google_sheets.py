import base64
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


GOOGLE_SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
GOOGLE_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
GOOGLE_SHEET_URL_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)")
GOOGLE_SHEET_CELL_SAFE_MAX_CHARS = 49_000
GOOGLE_SHEET_CELL_TRUNCATION_MARKER = " [TRUNCATED]"
DEFAULT_GOOGLE_SHEET_WRITE_CHUNK_ROWS = 300
DEFAULT_GOOGLE_SHEET_WRITE_MAX_ATTEMPTS = 4
DEFAULT_GOOGLE_SHEET_WRITE_RETRY_BASE_SECONDS = 1.0
DEFAULT_GOOGLE_SHEET_WRITE_MAX_DELAY_SECONDS = 8.0


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


def _trim_sheet_cell_text(value: str) -> str:
    text = value or ""
    if len(text) <= GOOGLE_SHEET_CELL_SAFE_MAX_CHARS:
        return text

    keep = GOOGLE_SHEET_CELL_SAFE_MAX_CHARS - len(GOOGLE_SHEET_CELL_TRUNCATION_MARKER)
    if keep <= 0:
        return text[:GOOGLE_SHEET_CELL_SAFE_MAX_CHARS]
    return text[:keep] + GOOGLE_SHEET_CELL_TRUNCATION_MARKER


def _normalize_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return value
    return _trim_sheet_cell_text(str(value))


def _safe_positive_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _safe_positive_float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _is_retryable_google_write_error(exc: HttpError) -> bool:
    status_code = getattr(getattr(exc, "resp", None), "status", None) or getattr(exc, "status_code", None)
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True

    detail = _extract_google_http_error_message(exc).lower()
    retryable_markers = (
        "rate limit",
        "quota exceeded",
        "backend error",
        "internal error",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "try again",
        "connection reset",
        "connection aborted",
    )
    return any(marker in detail for marker in retryable_markers)


def _execute_google_request_with_retries(*, action: str, request_factory: Callable[[], Any]) -> Any:
    max_attempts = _safe_positive_int_env("GOOGLE_SHEET_WRITE_MAX_ATTEMPTS", DEFAULT_GOOGLE_SHEET_WRITE_MAX_ATTEMPTS)
    retry_base_seconds = _safe_positive_float_env(
        "GOOGLE_SHEET_WRITE_RETRY_BASE_SECONDS",
        DEFAULT_GOOGLE_SHEET_WRITE_RETRY_BASE_SECONDS,
    )
    retry_max_delay_seconds = _safe_positive_float_env(
        "GOOGLE_SHEET_WRITE_MAX_DELAY_SECONDS",
        DEFAULT_GOOGLE_SHEET_WRITE_MAX_DELAY_SECONDS,
    )

    last_exc: Optional[HttpError] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return request_factory().execute()
        except HttpError as exc:
            last_exc = exc
            if attempt >= max_attempts or not _is_retryable_google_write_error(exc):
                raise
            delay = min(retry_max_delay_seconds, retry_base_seconds * (2 ** (attempt - 1)))
            print(
                f"[warn] retrying Google API call for {action} "
                f"(attempt {attempt}/{max_attempts}, sleep {delay:.1f}s): "
                f"{_extract_google_http_error_message(exc)}"
            )
            time.sleep(delay)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Google API retry wrapper failed without an exception while trying to {action}.")


def _get_spreadsheet_metadata(sheets: Any, spreadsheet_id: str) -> Dict[str, Any]:
    metadata = (
        sheets.spreadsheets()
        .get(
            spreadsheetId=spreadsheet_id,
            fields="properties/title,sheets/properties(title,sheetId,gridProperties(rowCount,columnCount))",
        )
        .execute()
    )
    spreadsheet_title = ((metadata.get("properties") or {}).get("title") or "").strip()
    sheets_meta = metadata.get("sheets") or []
    if not sheets_meta:
        raise RuntimeError("Google Sheet has no worksheets.")
    tab_titles: List[str] = []
    tab_ids_by_title: Dict[str, int] = {}
    tab_grid_by_title: Dict[str, Dict[str, int]] = {}
    for item in sheets_meta:
        properties = (item or {}).get("properties") or {}
        title = properties.get("title")
        if title:
            clean_title = str(title)
            tab_titles.append(clean_title)
            sheet_id = properties.get("sheetId")
            if isinstance(sheet_id, int):
                tab_ids_by_title[clean_title] = sheet_id
            grid_props = properties.get("gridProperties") or {}
            row_count = grid_props.get("rowCount")
            column_count = grid_props.get("columnCount")
            grid_entry: Dict[str, int] = {}
            if isinstance(row_count, int) and row_count > 0:
                grid_entry["row_count"] = row_count
            if isinstance(column_count, int) and column_count > 0:
                grid_entry["column_count"] = column_count
            if grid_entry:
                tab_grid_by_title[clean_title] = grid_entry
    if not tab_titles:
        raise RuntimeError("Google Sheet has no worksheets.")
    return {
        "spreadsheet_title": spreadsheet_title,
        "tab_titles": tab_titles,
        "tab_ids_by_title": tab_ids_by_title,
        "tab_grid_by_title": tab_grid_by_title,
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


def _ensure_worksheet_grid_capacity(
    *,
    sheets: Any,
    spreadsheet_id: str,
    sheet_id: int,
    current_rows: int,
    current_columns: int,
    required_rows: int,
    required_columns: int,
) -> None:
    safe_current_rows = max(0, int(current_rows or 0))
    safe_current_columns = max(0, int(current_columns or 0))
    target_rows = max(safe_current_rows, int(required_rows or 0))
    target_columns = max(safe_current_columns, int(required_columns or 0))

    if target_rows <= safe_current_rows and target_columns <= safe_current_columns:
        return

    target_rows = max(1, target_rows)
    target_columns = max(1, target_columns)
    _execute_google_request_with_retries(
        action=f"resize worksheet grid for sheetId {sheet_id} in spreadsheet '{spreadsheet_id}'",
        request_factory=lambda: (
            sheets.spreadsheets()
            .batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "requests": [
                        {
                            "updateSheetProperties": {
                                "properties": {
                                    "sheetId": int(sheet_id),
                                    "gridProperties": {
                                        "rowCount": target_rows,
                                        "columnCount": target_columns,
                                    },
                                },
                                "fields": "gridProperties.rowCount,gridProperties.columnCount",
                            }
                        }
                    ]
                },
            )
        ),
    )


def _to_sheet_values(df: pd.DataFrame) -> List[List[Any]]:
    frame = df.astype(object).where(pd.notna(df), "")
    rows: List[List[Any]] = [[_trim_sheet_cell_text(str(col)) for col in frame.columns.tolist()]]
    for row in frame.itertuples(index=False, name=None):
        normalized = [_normalize_cell_value(cell) for cell in row]
        rows.append(normalized)
    return rows


def _hide_columns_by_header_name(
    *,
    sheets: Any,
    spreadsheet_id: str,
    sheet_id: int,
    headers: List[str],
    hidden_column_names: List[str],
) -> None:
    if not hidden_column_names or not headers:
        return

    target_names = {str(name).strip().lower() for name in hidden_column_names if str(name).strip()}
    if not target_names:
        return

    indexes_to_hide: List[int] = []
    for idx, header in enumerate(headers):
        if str(header).strip().lower() in target_names:
            indexes_to_hide.append(idx)

    if not indexes_to_hide:
        return

    requests = [
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": idx,
                    "endIndex": idx + 1,
                },
                "properties": {"hiddenByUser": True},
                "fields": "hiddenByUser",
            }
        }
        for idx in sorted(set(indexes_to_hide))
    ]

    (
        sheets.spreadsheets()
        .batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests},
        )
        .execute()
    )


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
    hidden_column_names: Optional[List[str]] = None,
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
        hidden_column_names=hidden_column_names,
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


def _write_values_chunked(
    *,
    sheets: Any,
    spreadsheet_id: str,
    tab_name: str,
    values: List[List[Any]],
) -> None:
    escaped_tab_name = _escape_tab_name(tab_name)
    clear_action = f"clear worksheet '{tab_name}' in spreadsheet '{spreadsheet_id}'"
    _execute_google_request_with_retries(
        action=clear_action,
        request_factory=lambda: (
            sheets.spreadsheets()
            .values()
            .clear(
                spreadsheetId=spreadsheet_id,
                range=f"'{escaped_tab_name}'",
                body={},
            )
        ),
    )

    if not values:
        return

    headers = values[0]
    if headers:
        _execute_google_request_with_retries(
            action=f"write header row into worksheet '{tab_name}' for spreadsheet '{spreadsheet_id}'",
            request_factory=lambda: (
                sheets.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{escaped_tab_name}'!A1",
                    valueInputOption="RAW",
                    body={"values": [headers]},
                )
            ),
        )

    data_rows = values[1:]
    if not data_rows:
        return

    configured_chunk_rows = _safe_positive_int_env(
        "GOOGLE_SHEET_WRITE_CHUNK_ROWS",
        DEFAULT_GOOGLE_SHEET_WRITE_CHUNK_ROWS,
    )
    chunk_rows = max(1, configured_chunk_rows)
    row_cursor = 0

    while row_cursor < len(data_rows):
        chunk_end = min(len(data_rows), row_cursor + chunk_rows)
        chunk_values = data_rows[row_cursor:chunk_end]
        start_row = row_cursor + 2
        chunk_range = f"'{escaped_tab_name}'!A{start_row}"

        try:
            _execute_google_request_with_retries(
                action=(
                    f"write rows {start_row}-{start_row + len(chunk_values) - 1} "
                    f"into worksheet '{tab_name}' for spreadsheet '{spreadsheet_id}'"
                ),
                request_factory=lambda chunk_range=chunk_range, chunk_values=chunk_values: (
                    sheets.spreadsheets()
                    .values()
                    .update(
                        spreadsheetId=spreadsheet_id,
                        range=chunk_range,
                        valueInputOption="RAW",
                        body={"values": chunk_values},
                    )
                ),
            )
        except HttpError:
            if chunk_rows <= 1:
                raise
            next_chunk = max(1, chunk_rows // 2)
            print(
                f"[warn] Google Sheets write chunk failed at row {start_row}; "
                f"reducing chunk size from {chunk_rows} to {next_chunk} and retrying."
            )
            chunk_rows = next_chunk
            continue

        row_cursor = chunk_end


def write_dataframe_to_google_sheet(
    *,
    spreadsheet_id: str,
    df: pd.DataFrame,
    worksheet_title: str = "Candidates",
    hidden_column_names: Optional[List[str]] = None,
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

    try:
        metadata = _get_spreadsheet_metadata(sheets=sheets, spreadsheet_id=spreadsheet_id)
    except HttpError as exc:
        _raise_google_http_error(f"refresh spreadsheet metadata for '{spreadsheet_id}'", exc)

    values = _to_sheet_values(df)
    headers: List[str] = [str(h) for h in values[0]] if values else []
    tab_id = (metadata.get("tab_ids_by_title") or {}).get(tab_name)
    tab_grid = (metadata.get("tab_grid_by_title") or {}).get(tab_name) or {}
    required_rows = len(values)
    required_columns = max((len(row) for row in values), default=0)

    if isinstance(tab_id, int) and required_rows > 0 and required_columns > 0:
        try:
            _ensure_worksheet_grid_capacity(
                sheets=sheets,
                spreadsheet_id=spreadsheet_id,
                sheet_id=tab_id,
                current_rows=int(tab_grid.get("row_count") or 0),
                current_columns=int(tab_grid.get("column_count") or 0),
                required_rows=required_rows,
                required_columns=required_columns,
            )
        except HttpError as exc:
            _raise_google_http_error(
                f"resize worksheet '{tab_name}' grid in spreadsheet '{spreadsheet_id}'",
                exc,
            )

    try:
        _write_values_chunked(
            sheets=sheets,
            spreadsheet_id=spreadsheet_id,
            tab_name=tab_name,
            values=values,
        )
    except HttpError as exc:
        _raise_google_http_error(f"write candidate rows into spreadsheet '{spreadsheet_id}'", exc)

    if hidden_column_names:
        if isinstance(tab_id, int):
            try:
                _hide_columns_by_header_name(
                    sheets=sheets,
                    spreadsheet_id=spreadsheet_id,
                    sheet_id=tab_id,
                    headers=headers,
                    hidden_column_names=hidden_column_names,
                )
            except HttpError as exc:
                _raise_google_http_error(
                    f"hide configured columns in worksheet '{tab_name}' for spreadsheet '{spreadsheet_id}'",
                    exc,
                )

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
