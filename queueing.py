import os
from typing import Any, Dict, Optional

from redis import Redis
from redis.exceptions import RedisError, ResponseError
from rq import Queue


DEFAULT_QUEUE_NAME = "jd-pipeline"
DEFAULT_EVENT_TTL_SECONDS = 60 * 60 * 24
UPSTASH_LIMIT_ERROR_SNIPPET = "max requests limit exceeded"


class QueueingUnavailableError(RuntimeError):
    pass


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


def _is_upstash_request_limit_error(exc: BaseException) -> bool:
    return UPSTASH_LIMIT_ERROR_SNIPPET in str(exc).lower()


def _is_truthy(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def mark_event_seen(event_id: str, ttl_seconds: Optional[int] = None) -> bool:
    if not event_id:
        return False
    ttl = ttl_seconds or int(os.getenv("SLACK_EVENT_TTL_SECONDS", str(DEFAULT_EVENT_TTL_SECONDS)))
    key = f"slack:event:{event_id}"
    # Returns True only the first time this event id is seen.
    try:
        conn = get_redis_connection()
        return bool(conn.set(key, "1", nx=True, ex=ttl))
    except RuntimeError as exc:
        raise QueueingUnavailableError(f"redis_config_error: {exc}") from exc
    except ResponseError as exc:
        dedup_fail_open = _is_truthy(os.getenv("SLACK_EVENT_DEDUP_FAIL_OPEN"), default=True)
        if _is_upstash_request_limit_error(exc) and dedup_fail_open:
            return True
        if _is_upstash_request_limit_error(exc):
            raise QueueingUnavailableError("redis_quota_exceeded") from exc
        raise QueueingUnavailableError(f"redis_error: {exc}") from exc
    except RedisError as exc:
        dedup_fail_open = _is_truthy(os.getenv("SLACK_EVENT_DEDUP_FAIL_OPEN"), default=True)
        if dedup_fail_open:
            return True
        raise QueueingUnavailableError(f"redis_unavailable: {exc}") from exc


def clear_event_seen(event_id: str) -> None:
    if not event_id:
        return
    key = f"slack:event:{event_id}"
    try:
        conn = get_redis_connection()
        conn.delete(key)
    except (RedisError, RuntimeError):
        # Best effort cleanup only; avoid masking the main error path.
        return


def enqueue_jd_pipeline_job(payload: Dict[str, Any]):
    try:
        queue = get_queue()
        return queue.enqueue(
            "worker.process_jd_pipeline_job",
            kwargs=payload,
            job_timeout=int(os.getenv("RQ_JOB_TIMEOUT", "7200")),
            result_ttl=int(os.getenv("RQ_RESULT_TTL", "86400")),
            failure_ttl=int(os.getenv("RQ_FAILURE_TTL", "604800")),
        )
    except RuntimeError as exc:
        raise QueueingUnavailableError(f"redis_config_error: {exc}") from exc
    except ResponseError as exc:
        if _is_upstash_request_limit_error(exc):
            raise QueueingUnavailableError("redis_quota_exceeded") from exc
        raise QueueingUnavailableError(f"redis_error: {exc}") from exc
    except RedisError as exc:
        raise QueueingUnavailableError(f"redis_unavailable: {exc}") from exc
