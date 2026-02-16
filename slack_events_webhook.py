import argparse
import hashlib
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI

from process_candidates import CHANNEL_ID_DEFAULT, parse_args, run_pipeline_from_jd_text


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found in environment variables.")
    return value


def is_jd_message(text: str) -> bool:
    if not text:
        return False
    lines = text.splitlines()
    if not lines:
        return False
    return lines[0].strip().lower() == "# jd"


def extract_jd_text(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[1:]).strip()


def verify_slack_signature(
    signing_secret: str,
    timestamp: str,
    signature: str,
    body: bytes,
) -> bool:
    if not timestamp or not signature:
        return False

    try:
        ts = int(timestamp)
    except ValueError:
        return False

    if abs(time.time() - ts) > 60 * 5:
        return False

    base = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    expected = "v0=" + hmac.new(
        signing_secret.encode("utf-8"),
        base,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


class SlackEventsHandler(BaseHTTPRequestHandler):
    pipeline_lock = threading.Lock()
    seen_event_ids: Set[str] = set()
    seen_lock = threading.Lock()

    signing_secret: str = ""
    expected_channel_id: str = CHANNEL_ID_DEFAULT
    pipeline_args: Optional[argparse.Namespace] = None
    openai_api_key: str = ""
    exa_api_key: str = ""
    slack_token: str = ""

    def _send_json(self, status: int, payload: Dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/slack/events":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        signature = self.headers.get("X-Slack-Signature", "")
        timestamp = self.headers.get("X-Slack-Request-Timestamp", "")
        if not verify_slack_signature(self.signing_secret, timestamp, signature, body):
            self._send_json(401, {"ok": False, "error": "invalid_signature"})
            return

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "invalid_json"})
            return

        request_type = payload.get("type")
        if request_type == "url_verification":
            challenge = payload.get("challenge", "")
            self._send_text(200, challenge)
            return

        if request_type != "event_callback":
            self._send_json(200, {"ok": True, "ignored": "unsupported_type"})
            return

        event_id = payload.get("event_id", "")
        if not event_id:
            self._send_json(200, {"ok": True, "ignored": "missing_event_id"})
            return

        with self.seen_lock:
            if event_id in self.seen_event_ids:
                self._send_json(200, {"ok": True, "ignored": "duplicate_event"})
                return
            self.seen_event_ids.add(event_id)
            if len(self.seen_event_ids) > 5000:
                self.seen_event_ids.clear()
                self.seen_event_ids.add(event_id)

        event = payload.get("event", {}) or {}
        event_type = event.get("type")
        if event_type != "message":
            self._send_json(200, {"ok": True, "ignored": "non_message"})
            return

        if event.get("subtype") is not None or event.get("bot_id"):
            self._send_json(200, {"ok": True, "ignored": "non_user_message"})
            return

        channel_id = event.get("channel", "")
        if channel_id != self.expected_channel_id:
            self._send_json(200, {"ok": True, "ignored": "wrong_channel"})
            return

        text = event.get("text", "") or ""
        if not is_jd_message(text):
            self._send_json(200, {"ok": True, "ignored": "not_jd_format"})
            return

        jd_text = extract_jd_text(text)
        if not jd_text:
            self._send_json(200, {"ok": True, "ignored": "empty_jd"})
            return

        message_ts = event.get("ts")
        thread = threading.Thread(
            target=self._process_jd_event,
            args=(jd_text, message_ts),
            daemon=True,
        )
        thread.start()
        self._send_json(200, {"ok": True, "status": "accepted"})

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def log_message(self, format: str, *args) -> None:
        return

    @classmethod
    def _process_jd_event(cls, jd_text: str, message_ts: Optional[str]) -> None:
        if cls.pipeline_args is None:
            print("[webhook] Pipeline args not configured.")
            return

        if not cls.pipeline_lock.acquire(blocking=False):
            print("[webhook] Pipeline already running; skipping this event.")
            return

        try:
            print(f"[webhook] Processing # JD event ts={message_ts}")
            client = OpenAI(api_key=cls.openai_api_key)
            run_pipeline_from_jd_text(
                jd_text=jd_text,
                jd_message_ts=message_ts,
                args=cls.pipeline_args,
                client=client,
                exa_api_key=cls.exa_api_key,
                slack_token=cls.slack_token,
            )
            print(f"[webhook] Completed pipeline for ts={message_ts}")
        except Exception as exc:
            print(f"[webhook] Pipeline failed for ts={message_ts}: {exc}")
        finally:
            cls.pipeline_lock.release()


def build_pipeline_args(channel_id: str) -> argparse.Namespace:
    args = parse_args([])
    args.channel_id = channel_id
    args.debug = False
    args.stop_after = None
    return args


def parse_webhook_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slack Events API webhook for JD pipeline")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--channel-id", default=CHANNEL_ID_DEFAULT)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_webhook_args()

    SlackEventsHandler.signing_secret = require_env("SLACK_SIGNING_SECRET")
    SlackEventsHandler.openai_api_key = require_env("OPENAI_API_KEY")
    SlackEventsHandler.exa_api_key = require_env("EXA_API_KEY")
    SlackEventsHandler.slack_token = os.getenv("SLACK_BOT_TOKEN") or require_env("SLACK_USER_TOKEN")
    SlackEventsHandler.expected_channel_id = args.channel_id
    SlackEventsHandler.pipeline_args = build_pipeline_args(channel_id=args.channel_id)

    server = ThreadingHTTPServer((args.host, args.port), SlackEventsHandler)
    print(f"Slack webhook listening on http://{args.host}:{args.port}/slack/events")
    print("Health check: /healthz")
    server.serve_forever()


if __name__ == "__main__":
    main()
