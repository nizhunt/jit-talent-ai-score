#!/usr/bin/env python3
"""Standalone script to sync Instantly campaign analytics to the dashboard sheet.

Run directly via cron or Railway scheduled service:
    python sync_instantly_analytics.py

Schedule for every Friday 09:00 IST (03:30 UTC):
    30 3 * * 5 cd /path/to/repo && python sync_instantly_analytics.py
"""

import logging
import sys

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from instantly_analytics import sync_instantly_analytics_to_dashboard

    result = sync_instantly_analytics_to_dashboard()
    if not result.get("ok"):
        print(f"[error] analytics sync failed: {result}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[done] matched={result.get('matched', 0)} "
        f"updated={result.get('updated', 0)} "
        f"skipped={result.get('skipped', 0)} "
        f"errors={len(result.get('errors', []))}"
    )
    if result.get("errors"):
        for err in result["errors"]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
