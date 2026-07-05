from __future__ import annotations

import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Iterable, List

from subtitle_translator.translators.base import BaseTranslator


DEFAULT_SARVAM_TRANSLATE_ENDPOINT = "https://api.sarvam.ai/translate"

_LANGUAGE_CODES = {
    "as": "as-IN",
    "bn": "bn-IN",
    "en": "en-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "kok": "kok-IN",
    "ks": "ks-IN",
    "mai": "mai-IN",
    "ml": "ml-IN",
    "mni": "mni-IN",
    "mr": "mr-IN",
    "ne": "ne-IN",
    "or": "od-IN",
    "od": "od-IN",
    "pa": "pa-IN",
    "sa": "sa-IN",
    "sd": "sd-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "ur": "ur-IN",
    "brx": "brx-IN",
    "doi": "doi-IN",
    "sat": "sat-IN",
}

_MODEL_INPUT_LIMITS = {
    "mayura:v1": 1000,
    "sarvam-translate:v1": 2000,
}

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।])\s+|\n+")


class SarvamApiError(RuntimeError):
    pass


class SarvamRateLimitError(SarvamApiError):
    pass


Transport = Callable[[dict], dict]


class SarvamApiTranslator(BaseTranslator):
    """Sarvam translation backend using the REST API.

    The API key is kept in memory only by this class. Callers should pass it
    from a password field, environment variable, or OS keychain helper.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "mayura:v1",
        mode: str = "classic-colloquial",
        numerals_format: str = "international",
        endpoint: str = DEFAULT_SARVAM_TRANSLATE_ENDPOINT,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        max_workers: int = 1,
        min_request_interval_seconds: float = 0.4,
        transport: Transport | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise SarvamApiError("Sarvam API key is required.")

        if model not in _MODEL_INPUT_LIMITS:
            raise SarvamApiError(
                "Unsupported Sarvam translation model. Use 'mayura:v1' or "
                "'sarvam-translate:v1'."
            )

        self._api_key = api_key
        self.model = model
        self.mode = "formal" if model == "sarvam-translate:v1" else mode
        self.numerals_format = numerals_format
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.retries = max(0, retries)
        self.max_workers = max(1, max_workers)
        self.min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self.max_input_chars = _MODEL_INPUT_LIMITS[model]
        self._transport = transport
        self._ssl_context = ssl_context or _create_ssl_context()
        self._metrics_lock = threading.Lock()
        self._attempted_request_count = 0
        self._successful_request_count = 0
        self._attempted_input_chars = 0
        self._successful_input_chars = 0
        self._request_ids: list[str] = []
        self._request_pace_lock = threading.Lock()
        self._next_request_at = 0.0

    @property
    def display_name(self) -> str:
        if self.model == "mayura:v1":
            return f"Sarvam API ({self.model}, {self.mode})"
        return f"Sarvam API ({self.model})"

    @property
    def pipeline_chunk_size(self) -> int:
        return 1

    @property
    def attempted_request_count(self) -> int:
        with self._metrics_lock:
            return self._attempted_request_count

    @property
    def successful_request_count(self) -> int:
        with self._metrics_lock:
            return self._successful_request_count

    @property
    def request_ids(self) -> list[str]:
        with self._metrics_lock:
            return list(self._request_ids)

    @property
    def attempted_input_chars(self) -> int:
        with self._metrics_lock:
            return self._attempted_input_chars

    @property
    def successful_input_chars(self) -> int:
        with self._metrics_lock:
            return self._successful_input_chars

    @property
    def usage_summary(self) -> str:
        attempted = self.attempted_request_count
        successful = self.successful_request_count
        if attempted == 0:
            return f"{self.display_name}; no Sarvam API requests sent"

        summary = (
            f"{self.display_name}; Sarvam API responses: "
            f"{successful} successful/{attempted} attempted; input chars: "
            f"{self.successful_input_chars} successful/{self.attempted_input_chars} sent"
        )
        request_ids = self.request_ids
        if request_ids:
            summary += f"; last request_id: {request_ids[-1]}"
        return summary

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        src_code = to_sarvam_language_code(source_lang)
        tgt_code = to_sarvam_language_code(target_lang)
        materialized = list(texts)
        if not materialized:
            return []

        if self.model == "sarvam-translate:v1" and src_code == "auto":
            raise SarvamApiError(
                "sarvam-translate:v1 does not support automatic source language "
                "detection. Choose a source language or use mayura:v1."
            )

        worker_count = min(self.max_workers, len(materialized))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(
                executor.map(
                    lambda text: self._translate_with_splitting(text, src_code, tgt_code),
                    materialized,
                )
            )

    def _translate_with_splitting(self, text: str, source_code: str, target_code: str) -> str:
        text = text.strip()
        if not text:
            return ""

        parts = _split_text_for_limit(text, self.max_input_chars)
        translated = [self._translate_one(part, source_code, target_code) for part in parts]
        return " ".join(part.strip() for part in translated if part.strip()).strip()

    def _translate_one(self, text: str, source_code: str, target_code: str) -> str:
        payload = {
            "input": text,
            "source_language_code": source_code,
            "target_language_code": target_code,
            "model": self.model,
            "numerals_format": self.numerals_format,
        }
        if self.model == "mayura:v1":
            payload["mode"] = self.mode

        data = self._post_json(payload)
        translated = data.get("translated_text")
        if not isinstance(translated, str):
            raise SarvamApiError("Sarvam response did not include translated_text.")
        self._record_success(data, len(text))
        return translated

    def _post_json(self, payload: dict) -> dict:
        if self._transport is not None:
            input_chars = _payload_input_chars(payload)
            self._record_attempt(input_chars)
            data = self._transport(payload)
            if not isinstance(data, dict):
                raise SarvamApiError("Sarvam transport returned a non-object response.")
            return data

        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": self._api_key,
            },
            method="POST",
        )

        last_error: Exception | None = None
        input_chars = _payload_input_chars(payload)
        for attempt in range(self.retries + 1):
            try:
                self._wait_for_request_slot()
                self._record_attempt(input_chars)
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                    context=self._ssl_context,
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS_CODES or attempt == self.retries:
                    raise _http_error_to_sarvam_error(exc) from exc
                _sleep_before_retry(exc, attempt)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt == self.retries:
                    raise _network_error_to_sarvam_error(exc) from exc
                _sleep_before_retry(None, attempt)

        raise SarvamApiError(f"Sarvam API request failed: {last_error}")

    def _record_attempt(self, input_chars: int) -> None:
        with self._metrics_lock:
            self._attempted_request_count += 1
            self._attempted_input_chars += input_chars

    def _record_success(self, data: dict, input_chars: int | None = None) -> None:
        request_id = data.get("request_id")
        with self._metrics_lock:
            self._successful_request_count += 1
            if input_chars is not None:
                self._successful_input_chars += input_chars
            if isinstance(request_id, str) and request_id:
                self._request_ids.append(request_id)

    def _wait_for_request_slot(self) -> None:
        if self.min_request_interval_seconds <= 0:
            return

        with self._request_pace_lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + (
                self.min_request_interval_seconds
            )

        if wait_seconds > 0:
            time.sleep(wait_seconds)


def to_sarvam_language_code(lang_code: str) -> str:
    normalized = lang_code.strip()
    if normalized == "auto":
        return "auto"
    if re.fullmatch(r"[a-z]{2,3}-IN", normalized):
        return normalized

    sarvam_code = _LANGUAGE_CODES.get(normalized.lower())
    if sarvam_code:
        return sarvam_code

    raise SarvamApiError(f"Unsupported Sarvam language code: {lang_code}")


def _split_text_for_limit(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]

    raw_parts = [part for part in _SENTENCE_BOUNDARY_RE.split(text) if part]
    parts: list[str] = []
    current = ""
    for raw in raw_parts:
        candidate = f"{current} {raw}".strip() if current else raw
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(raw) <= limit:
            current = raw
        else:
            parts.extend(_hard_split(raw, limit))
            current = ""
    if current:
        parts.append(current)
    return parts


def _hard_split(text: str, limit: int) -> list[str]:
    parts: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < max(1, limit // 2):
            split_at = limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return parts


def _http_error_to_sarvam_error(exc: urllib.error.HTTPError) -> SarvamApiError:
    body = exc.read().decode("utf-8", errors="replace")
    message = body
    code = ""
    try:
        parsed = json.loads(body)
        error = parsed.get("error", {})
        if isinstance(error, dict):
            message = str(error.get("message") or message)
            code = str(error.get("code") or "")
    except json.JSONDecodeError:
        pass

    detail = f"{code}: {message}" if code else message
    if exc.code == 403:
        prefix = "Sarvam rejected the API key or this account is not allowed to use the model"
    elif exc.code == 422:
        prefix = "Sarvam could not process this request"
    elif exc.code == 429:
        retry_after = _parse_retry_after(exc)
        retry_guidance = (
            f" Retry after about {retry_after:.0f} seconds, reduce batch size, "
            "or resume later from the saved checkpoint."
            if retry_after is not None
            else " Retry later, reduce batch size, or resume later from the saved checkpoint."
        )
        return SarvamRateLimitError(
            f"Sarvam rate limit exceeded: {detail}.{retry_guidance}"
        )
    else:
        prefix = f"Sarvam API returned HTTP {exc.code}"

    return SarvamApiError(f"{prefix}: {detail}")


def _payload_input_chars(payload: dict) -> int:
    value = payload.get("input", "")
    return len(value) if isinstance(value, str) else 0


def _network_error_to_sarvam_error(exc: Exception) -> SarvamApiError:
    if _is_certificate_error(exc):
        return SarvamApiError(
            "Could not validate Sarvam API TLS certificate. The Sarvam request "
            "was not sent successfully. Install/update Python certificates or "
            "reinstall project dependencies so the certifi CA bundle is available. "
            f"Details: {exc}"
        )
    return SarvamApiError(f"Could not reach Sarvam API: {exc}")


def _is_certificate_error(exc: Exception) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    reason = getattr(exc, "reason", None)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return True
    text = str(exc).lower()
    return "certificate_verify_failed" in text or "certificate verify failed" in text


def _create_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _sleep_before_retry(exc: urllib.error.HTTPError | None, attempt: int) -> None:
    delay = _retry_delay(exc, attempt)
    time.sleep(delay)


def _retry_delay(exc: urllib.error.HTTPError | None, attempt: int) -> float:
    retry_after = _parse_retry_after(exc) if exc is not None else None
    if retry_after is not None:
        return retry_after
    if exc is not None and exc.code == 429:
        return min(5.0 * (2.0 ** attempt), 30.0)
    return min(2.0 ** attempt, 8.0)


def _parse_retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if exc.headers else None
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
