from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
}

_MODEL_INPUT_LIMITS = {
    "mayura:v1": 1000,
    "sarvam-translate:v1": 2000,
}

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।])\s+|\n+")


class SarvamApiError(RuntimeError):
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
        retries: int = 2,
        max_workers: int = 4,
        transport: Transport | None = None,
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
        self.max_input_chars = _MODEL_INPUT_LIMITS[model]
        self._transport = transport

    def translate_batch(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        src_code = to_sarvam_language_code(source_lang)
        tgt_code = to_sarvam_language_code(target_lang)
        materialized = list(texts)
        if not materialized:
            return []

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
        return translated

    def _post_json(self, payload: dict) -> dict:
        if self._transport is not None:
            return self._transport(payload)

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
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_STATUS_CODES or attempt == self.retries:
                    raise _http_error_to_sarvam_error(exc) from exc
                _sleep_before_retry(exc, attempt)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt == self.retries:
                    raise SarvamApiError(f"Could not reach Sarvam API: {exc}") from exc
                _sleep_before_retry(None, attempt)

        raise SarvamApiError(f"Sarvam API request failed: {last_error}")


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

    if exc.code == 403:
        prefix = "Sarvam rejected the API key or this account is not allowed to use the model"
    elif exc.code == 422:
        prefix = "Sarvam could not process this request"
    elif exc.code == 429:
        prefix = "Sarvam rate limit exceeded"
    else:
        prefix = f"Sarvam API returned HTTP {exc.code}"

    detail = f"{code}: {message}" if code else message
    return SarvamApiError(f"{prefix}: {detail}")


def _sleep_before_retry(exc: urllib.error.HTTPError | None, attempt: int) -> None:
    retry_after = None
    if exc is not None:
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                retry_after = None

    delay = retry_after if retry_after is not None else min(2.0 ** attempt, 8.0)
    time.sleep(delay)
