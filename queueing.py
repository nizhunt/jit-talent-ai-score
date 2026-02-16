import os
from typing import Any, Dict, Optional

from redis import Redis
from rq import Queue


DEFAULT_QUEUE_NAME = "jd-pipeline"
DEFAULT_EVENT_TTL_SECONDS = 60 * 60 * 24


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def get_redis_connection() -> Redis:
    redis_url = os.getenv("REDIS_URL") or os.getenv("KV_URL")
    if not redis_url:
        raise RuntimeError("Missing required env var: REDIS_URL (or KV_URL)")
    return Redis.from_url(redis_url)


def get_queue_name() -> str:
    return os.getenv("RQ_QUEUE_NAME", DEFAULT_QUEUE_NAME)


def get_queue() -> Queue:
    timeout = int(os.getenv("RQ_JOB_TIMEOUT", "7200"))
    return Queue(
        get_queue_name(),
        connection=get_redis_connection(),
        default_timeout=timeout,
    )


def mark_event_seen(event_id: str, ttl_seconds: Optional[int] = None) -> bool:
    if not event_id:
        return False
    ttl = ttl_seconds or int(os.getenv("SLACK_EVENT_TTL_SECONDS", str(DEFAULT_EVENT_TTL_SECONDS)))
    key = f"slack:event:{event_id}"
    conn = get_redis_connection()
    # Returns True only the first time this event id is seen.
    return bool(conn.set(key, "1", nx=True, ex=ttl))


def clear_event_seen(event_id: str) -> None:
    if not event_id:
        return
    key = f"slack:event:{event_id}"
    conn = get_redis_connection()
    conn.delete(key)


def enqueue_jd_pipeline_job(payload: Dict[str, Any]):
    queue = get_queue()
    return queue.enqueue(
        "worker.process_jd_pipeline_job",
        kwargs=payload,
        job_timeout=int(os.getenv("RQ_JOB_TIMEOUT", "7200")),
        result_ttl=int(os.getenv("RQ_RESULT_TTL", "86400")),
        failure_ttl=int(os.getenv("RQ_FAILURE_TTL", "604800")),
    )
