import argparse
import os
import uuid
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from rq import Worker

from process_candidates import parse_args, post_slack_message, run_pipeline_from_jd_text
from queueing import get_queue_name, get_redis_connection


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def build_pipeline_args(channel_id: str) -> argparse.Namespace:
    args = parse_args([])
    args.channel_id = channel_id
    args.debug = False
    args.stop_after = None

    project_root = os.path.dirname(os.path.abspath(__file__))
    args.exa_query_prompt_path = os.path.join(project_root, "prompts", "exa-querry-creator-prompt.md")
    args.scorer_prompt_path = os.path.join(project_root, "prompts", "scorer_prompt.md")

    run_id = uuid.uuid4().hex[:12]
    run_dir = os.path.join("/tmp", f"jit-talent-worker-{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    args.jd_path = os.path.join(run_dir, "jd.md")
    args.results_json = os.path.join(run_dir, "exa-results.json")
    args.candidates_csv = os.path.join(run_dir, "candidates.csv")
    args.dedup_csv = os.path.join(run_dir, "candidates-dedup.csv")
    args.scored_csv = os.path.join(run_dir, "candidates-scored.csv")
    args.debug_dir = os.path.join(run_dir, "debug")
    return args


def process_jd_pipeline_job(
    *,
    jd_text: str,
    message_ts: Optional[str],
    channel_id: str,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()

    openai_api_key = require_env("OPENAI_API_KEY")
    exa_api_key = require_env("EXA_API_KEY")
    slack_token = os.getenv("SLACK_BOT_TOKEN") or require_env("SLACK_USER_TOKEN")

    post_slack_message(
        slack_token=slack_token,
        channel_id=channel_id,
        text="Thanks for sharing the JD. Processing has started.",
    )

    client = OpenAI(api_key=openai_api_key)
    args = build_pipeline_args(channel_id=channel_id)
    result = run_pipeline_from_jd_text(
        jd_text=jd_text,
        jd_message_ts=message_ts,
        args=args,
        client=client,
        exa_api_key=exa_api_key,
        slack_token=slack_token,
    )
    return {"ok": True, "event_id": event_id, "result": result}


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ worker for JD pipeline")
    parser.add_argument("--burst", action="store_true", help="Exit after current queue is drained")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_worker_args()

    conn = get_redis_connection()
    queue_name = get_queue_name()
    worker = Worker([queue_name], connection=conn)
    worker.work(with_scheduler=True, burst=args.burst)


if __name__ == "__main__":
    main()
