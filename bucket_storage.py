import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    import boto3
except Exception:  # pragma: no cover - handled at runtime when dependency is missing
    boto3 = None

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


T = TypeVar("T")
DEFAULT_REGION = "auto"
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BASE_BACKOFF_SECONDS = 0.5
DEFAULT_MAX_BACKOFF_SECONDS = 8.0
DEFAULT_ADDRESSING_STYLE = "virtual"


def _is_truthy(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[bucket][warn] invalid {name}={raw!r}; using default {default}")
        return default
    if value <= 0:
        print(f"[bucket][warn] invalid {name}={raw!r}; using default {default}")
        return default
    return value


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        print(f"[bucket][warn] invalid {name}={raw!r}; using default {default}")
        return default
    if value <= 0:
        print(f"[bucket][warn] invalid {name}={raw!r}; using default {default}")
        return default
    return value


def _require_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = " or ".join(names)
    raise RuntimeError(f"Missing required bucket env var: {joined}")


def _resolve_bucket_name_from_env() -> str:
    aws_bucket = os.getenv("AWS_S3_BUCKET_NAME")
    if aws_bucket:
        return aws_bucket

    direct_bucket = os.getenv("BUCKET")
    if direct_bucket:
        return direct_bucket

    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if s3_bucket:
        print("[bucket][warn] using S3_BUCKET_NAME; Railway buckets typically provide BUCKET.")
        return s3_bucket

    legacy_bucket = os.getenv("RAILWAY_BUCKET_NAME")
    if legacy_bucket:
        print(
            "[bucket][warn] using RAILWAY_BUCKET_NAME as bucket identifier. "
            "Railway docs recommend BUCKET for S3 API bucket name."
        )
        return legacy_bucket

    raise RuntimeError(
        "Missing required bucket env var: AWS_S3_BUCKET_NAME (preferred), BUCKET, or S3_BUCKET_NAME."
    )


def _read_addressing_style_env(default: str = DEFAULT_ADDRESSING_STYLE) -> str:
    raw = (
        os.getenv("S3_ADDRESSING_STYLE")
        or os.getenv("RAILWAY_BUCKET_ADDRESSING_STYLE")
        or os.getenv("RAILWAY_BUCKET_URL_STYLE")
    )
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"auto", "virtual", "path"}:
        return value
    print(
        f"[bucket][warn] invalid S3 addressing style={raw!r}; "
        f"using default {default!r} (valid: auto|virtual|path)"
    )
    return default


def _not_found_error(exc: Exception) -> bool:
    if not isinstance(exc, ClientError):
        return False
    code = str((exc.response.get("Error") or {}).get("Code") or "").strip()
    return code in {"404", "NoSuchKey", "NotFound"}


@dataclass(frozen=True)
class S3BucketConfig:
    bucket_name: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region_name: str
    key_prefix: str
    use_ssl: bool
    addressing_style: str
    max_attempts: int
    base_backoff_seconds: float
    max_backoff_seconds: float


def load_s3_bucket_config_from_env() -> S3BucketConfig:
    key_prefix = (os.getenv("RAILWAY_BUCKET_KEY_PREFIX") or os.getenv("S3_KEY_PREFIX") or "").strip().strip("/")
    return S3BucketConfig(
        bucket_name=_resolve_bucket_name_from_env(),
        endpoint_url=_require_env("AWS_ENDPOINT_URL", "ENDPOINT", "RAILWAY_BUCKET_ENDPOINT_URL", "S3_ENDPOINT_URL"),
        access_key_id=_require_env(
            "AWS_ACCESS_KEY_ID",
            "ACCESS_KEY_ID",
            "RAILWAY_BUCKET_ACCESS_KEY_ID",
            "S3_ACCESS_KEY_ID",
        ),
        secret_access_key=_require_env(
            "AWS_SECRET_ACCESS_KEY",
            "SECRET_ACCESS_KEY",
            "RAILWAY_BUCKET_SECRET_ACCESS_KEY",
            "S3_SECRET_ACCESS_KEY",
        ),
        region_name=(
            os.getenv("AWS_DEFAULT_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("REGION")
            or os.getenv("RAILWAY_BUCKET_REGION")
            or os.getenv("S3_REGION")
            or DEFAULT_REGION
        ),
        key_prefix=key_prefix,
        use_ssl=_is_truthy(os.getenv("RAILWAY_BUCKET_USE_SSL"), default=True),
        addressing_style=_read_addressing_style_env(),
        max_attempts=_read_positive_int_env("RAILWAY_BUCKET_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
        base_backoff_seconds=_read_positive_float_env("RAILWAY_BUCKET_BASE_BACKOFF_SECONDS", DEFAULT_BASE_BACKOFF_SECONDS),
        max_backoff_seconds=_read_positive_float_env("RAILWAY_BUCKET_MAX_BACKOFF_SECONDS", DEFAULT_MAX_BACKOFF_SECONDS),
    )


class S3BucketClient:
    def __init__(self, config: S3BucketConfig, *, s3_client: Optional[Any] = None):
        self.config = config
        if s3_client is not None:
            self._client = s3_client
            return

        if boto3 is None:
            raise RuntimeError("boto3 is not installed. Add 'boto3' to requirements and install dependencies.")

        # Disable SDK-level retries so retry logs are owned by this helper.
        boto_config = Config(
            retries={"max_attempts": 1, "mode": "standard"},
            s3={"addressing_style": config.addressing_style},
        )
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name=config.region_name,
            use_ssl=config.use_ssl,
            config=boto_config,
        )

    def _full_key(self, key: str) -> str:
        cleaned_key = str(key or "").strip().lstrip("/")
        if not cleaned_key:
            raise ValueError("Bucket object key cannot be empty.")
        if self.config.key_prefix:
            return f"{self.config.key_prefix}/{cleaned_key}"
        return cleaned_key

    def _run_with_retries(self, operation: str, key: str, fn: Callable[[], T]) -> T:
        max_attempts = max(1, int(self.config.max_attempts))
        for attempt in range(1, max_attempts + 1):
            try:
                result = fn()
                if attempt > 1:
                    print(f"[bucket] op={operation} key={key} recovered_on_attempt={attempt}/{max_attempts}")
                return result
            except Exception as exc:
                if _not_found_error(exc):
                    raise
                if attempt >= max_attempts or not isinstance(exc, (ClientError, BotoCoreError, OSError, TimeoutError)):
                    print(f"[bucket][error] op={operation} key={key} attempt={attempt}/{max_attempts} failed: {exc}")
                    raise

                delay = min(
                    float(self.config.base_backoff_seconds) * (2 ** (attempt - 1)),
                    float(self.config.max_backoff_seconds),
                )
                jitter = random.uniform(0.0, min(0.25, delay / 2.0))
                sleep_for = delay + jitter
                print(
                    f"[bucket][retry] op={operation} key={key} attempt={attempt}/{max_attempts} "
                    f"error={exc} sleep={sleep_for:.2f}s"
                )
                time.sleep(sleep_for)
        raise RuntimeError(f"Bucket operation exhausted retries unexpectedly: op={operation} key={key}")

    def upload_bytes(self, key: str, payload: bytes, *, content_type: Optional[str] = None) -> Dict[str, Any]:
        object_key = self._full_key(key)
        payload_size = len(payload)

        def _put() -> Any:
            kwargs: Dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Key": object_key,
                "Body": payload,
            }
            if content_type:
                kwargs["ContentType"] = content_type
            return self._client.put_object(**kwargs)

        self._run_with_retries("upload", object_key, _put)
        print(f"[bucket] uploaded key={object_key} size_bytes={payload_size}")
        return {"key": object_key, "size_bytes": payload_size}

    def upload_text(self, key: str, text: str, *, content_type: str = "text/plain; charset=utf-8") -> Dict[str, Any]:
        return self.upload_bytes(key, (text or "").encode("utf-8"), content_type=content_type)

    def upload_json(self, key: str, value: Any) -> Dict[str, Any]:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return self.upload_bytes(key, payload, content_type="application/json")

    def upload_file(self, key: str, file_path: str, *, content_type: Optional[str] = None) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Upload file path does not exist: {file_path}")
        payload = path.read_bytes()
        return self.upload_bytes(key, payload, content_type=content_type)

    def download_bytes(self, key: str) -> bytes:
        object_key = self._full_key(key)

        def _get() -> bytes:
            response = self._client.get_object(Bucket=self.config.bucket_name, Key=object_key)
            body = response.get("Body")
            if body is None:
                return b""
            return body.read()

        try:
            payload = self._run_with_retries("download", object_key, _get)
        except Exception as exc:
            if _not_found_error(exc):
                raise FileNotFoundError(f"Bucket object not found: {object_key}") from exc
            raise

        print(f"[bucket] downloaded key={object_key} size_bytes={len(payload)}")
        return payload

    def download_text(self, key: str) -> str:
        return self.download_bytes(key).decode("utf-8")

    def download_json(self, key: str) -> Any:
        raw = self.download_text(key)
        return json.loads(raw)

    def download_file(self, key: str, file_path: str) -> Dict[str, Any]:
        payload = self.download_bytes(key)
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {"path": str(path), "size_bytes": len(payload)}

    def exists(self, key: str) -> bool:
        object_key = self._full_key(key)

        def _head() -> Any:
            return self._client.head_object(Bucket=self.config.bucket_name, Key=object_key)

        try:
            self._run_with_retries("exists", object_key, _head)
            return True
        except Exception as exc:
            if _not_found_error(exc):
                return False
            raise

    def delete_prefix(self, prefix: str) -> int:
        full_prefix = self._full_key(prefix).rstrip("/") + "/"
        deleted_count = 0
        continuation_token: Optional[str] = None

        while True:
            list_kwargs: Dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Prefix": full_prefix,
                "MaxKeys": 1000,
            }
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = self._run_with_retries(
                "list",
                full_prefix,
                lambda: self._client.list_objects_v2(**list_kwargs),
            )
            contents = response.get("Contents") or []
            keys = [str(item.get("Key")) for item in contents if item.get("Key")]
            if keys:
                for start in range(0, len(keys), 1000):
                    batch = keys[start : start + 1000]
                    self._run_with_retries(
                        "delete",
                        full_prefix,
                        lambda batch=batch: self._client.delete_objects(
                            Bucket=self.config.bucket_name,
                            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
                        ),
                    )
                    deleted_count += len(batch)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

        print(f"[bucket] deleted_prefix={full_prefix} deleted_objects={deleted_count}")
        return deleted_count


def build_s3_bucket_client_from_env() -> S3BucketClient:
    config = load_s3_bucket_config_from_env()
    return S3BucketClient(config=config)
