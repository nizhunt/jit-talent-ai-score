import os
from typing import Any, Dict, Optional

from redis import Redis
from redis.exceptions import RedisError, ResponseError
from rq import Queue


DEFAULT_JD_SOURCE_QUEUE_NAME = "jd-pipeline"
DEFAULT_JD_SCORE_QUEUE_NAME = "jd-pipeline-score"
DEFAULT_REPLY_QUEUE_NAME = "reply-enrichment"
DEFAULT_ADMIN_QUEUE_NAME = "jd-admin"
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


def get_jd_source_queue_name() -> str:
    return (
        os.getenv("RQ_JD_SOURCE_QUEUE_NAME")
        or os.getenv("RQ_JD_QUEUE_NAME")
        or os.getenv("RQ_QUEUE_NAME")
        or DEFAULT_JD_SOURCE_QUEUE_NAME
    )


def get_jd_score_queue_name() -> str:
    return os.getenv("RQ_JD_SCORE_QUEUE_NAME") or DEFAULT_JD_SCORE_QUEUE_NAME


def get_jd_queue_name() -> str:
    # Backward-compatible alias; "jd queue" now maps to source-stage queue.
    return get_jd_source_queue_name()


def get_reply_queue_name() -> str:
    return os.getenv("RQ_REPLY_QUEUE_NAME", DEFAULT_REPLY_QUEUE_NAME)


def get_admin_queue_name() -> str:
    return os.getenv("RQ_JD_ADMIN_QUEUE_NAME") or DEFAULT_ADMIN_QUEUE_NAME


def get_queue_name() -> str:
    # Backward-compatible alias for older callers that only know about one queue.
    return get_jd_queue_name()


def get_queue(queue_name: Optional[str] = None) -> Queue:
    timeout = int(os.getenv("RQ_JOB_TIMEOUT", "7200"))
    return Queue(
        queue_name or get_queue_name(),
        connection=get_redis_connection(),
        default_timeout=timeout,
    )


def _is_upstash_request_limit_error(exc: BaseException) -> bool:
    return UPSTASH_LIMIT_ERROR_SNIPPET in str(exc).lower()


def _is_truthy(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _queue_job_timeout_seconds() -> int:
    return int(os.getenv("RQ_JOB_TIMEOUT", "7200"))


def _queue_result_ttl_seconds() -> int:
    return int(os.getenv("RQ_RESULT_TTL", "86400"))


def _queue_failure_ttl_seconds() -> int:
    return int(os.getenv("RQ_FAILURE_TTL", "604800"))


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


def _enqueue_job(
    *,
    queue_name: str,
    function_name: str,
    payload: Dict[str, Any],
    job_id: Optional[str] = None,
):
    try:
        queue = get_queue(queue_name)
        print(
            f"[queue] enqueue_requested queue={queue_name} function={function_name} "
            f"job_id={job_id or 'auto'}"
        )
        try:
            job = queue.enqueue(
                function_name,
                kwargs=payload,
                job_timeout=_queue_job_timeout_seconds(),
                result_ttl=_queue_result_ttl_seconds(),
                failure_ttl=_queue_failure_ttl_seconds(),
                job_id=job_id,
            )
            print(
                f"[queue] enqueue_success queue={queue_name} function={function_name} "
                f"job_id={job.id}"
            )
            return job
        except Exception as exc:
            duplicate_job_id = bool(job_id) and "already exists" in str(exc).lower()
            if duplicate_job_id:
                existing = queue.fetch_job(str(job_id))
                if existing is not None:
                    print(
                        f"[queue] enqueue_dedup queue={queue_name} function={function_name} "
                        f"job_id={existing.id}"
                    )
                    return existing
            raise
    except RuntimeError as exc:
        raise QueueingUnavailableError(f"redis_config_error: {exc}") from exc
    except ResponseError as exc:
        if _is_upstash_request_limit_error(exc):
            raise QueueingUnavailableError("redis_quota_exceeded") from exc
        raise QueueingUnavailableError(f"redis_error: {exc}") from exc
    except RedisError as exc:
        raise QueueingUnavailableError(f"redis_unavailable: {exc}") from exc


def enqueue_jd_source_job(payload: Dict[str, Any]):
    return _enqueue_job(
        queue_name=get_jd_source_queue_name(),
        function_name="worker.process_jd_source_job",
        payload=payload,
    )


def enqueue_jd_score_job(payload: Dict[str, Any], *, job_id: Optional[str] = None):
    return _enqueue_job(
        queue_name=get_jd_score_queue_name(),
        function_name="worker.process_jd_score_job",
        payload=payload,
        job_id=job_id,
    )


def enqueue_jd_pipeline_job(payload: Dict[str, Any]):
    # Backward-compatible alias used by app.py and existing callers.
    return enqueue_jd_source_job(payload)


def enqueue_thread_reply_enrichment_job(payload: Dict[str, Any]):
    return _enqueue_job(
        queue_name=get_reply_queue_name(),
        function_name="worker.process_thread_reply_enrichment_job",
        payload=payload,
    )


def enqueue_jd_admin_job(payload: Dict[str, Any]):
    return _enqueue_job(
        queue_name=get_admin_queue_name(),
        function_name="worker.process_jd_admin_job",
        payload=payload,
    )
