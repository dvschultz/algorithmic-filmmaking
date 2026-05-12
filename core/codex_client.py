"""Thin HTTP client for the OpenAI Codex inference backend.

This module is what ``core.llm_client.complete_routed`` / ``stream_chat_routed``
delegate to when ``auth_mode == "subscription"``. LiteLLM doesn't speak the
Codex backend's wire shape, so we maintain a small focused client rather
than registering a custom LiteLLM provider (that path is more invasive
and harder to evolve as Codex's API changes).

Three responsibilities:

1. **Model coercion**: the analysis features in ``core/analysis/*`` pass
   model names like ``gemini/gemini-2.5-flash`` or ``anthropic/claude-...``
   from their per-feature settings. The Codex backend serves only OpenAI
   models, so we map every configured model to its OpenAI substitute
   when subscription mode is active. The substitution map lives in
   ``CODEX_MODEL_SUBSTITUTIONS`` and is mechanically tunable. Per R-A2,
   features whose only usable shape is Gemini-specific (e.g., video-file
   content blocks) fall back to ``frame mode`` semantics; the per-feature
   sites are aware of this and pass already-frame-shaped content.

2. **Bearer-token redaction**: ``_sanitize_for_log`` scrubs the
   Authorization header from any error message we propagate, so callers
   can safely ``logger.error(str(exc))`` without leaking tokens to log
   files. Mirrors the R-S4 pattern from
   ``core/spine/chatgpt_oauth_flow._redact_authorization``.

3. **Specific error taxonomy**: ``TokenMissingError``,
   ``TokenExpiredError``, ``QuotaExceededError``, ``CodexBackendError``,
   ``MalformedCodexResponseError`` — each error is what the routing
   helper raises so U8's UX layer can branch cleanly.

Placeholder values: ``CODEX_COMPLETION_ENDPOINT`` lives in
``core/spine/chatgpt_auth.py`` from U2 and is a placeholder until a
follow-up session reads Codex CLI's source. The request/response shape
implemented here matches the OpenAI Chat Completions API (the broadest
common ground); a future session refines the exact wire shape if the
Codex endpoint differs.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import httpx

from core.spine.chatgpt_auth import CODEX_COMPLETION_ENDPOINT

logger = logging.getLogger(__name__)


# --- Errors -----------------------------------------------------------------


class CodexError(Exception):
    """Base class for every error raised by this client."""


class TokenMissingError(CodexError):
    """auth_mode is subscription but no token is in keyring.

    Surfaced before any HTTP call — the routing helper recognizes the
    "subscription mode + no blob" state from ``load_active_auth`` and
    raises this directly so callers don't waste a round trip.
    """


class TokenExpiredError(CodexError):
    """The Codex backend returned 401 — token expired or revoked.

    The routing helper catches this once, attempts a refresh through
    ``core/spine/chatgpt_oauth_flow.refresh_token_blob``, and retries
    the call. If refresh also returns 401, this is re-raised to the
    UI layer (U8) which prompts the user to sign in again.
    """


class QuotaExceededError(CodexError):
    """The Codex backend returned 429 — Plus quota window exhausted."""


class CodexBackendError(CodexError):
    """Non-2xx, non-401, non-429 response from the Codex backend.

    Carries the status code so callers can branch on 5xx (transient,
    worth retrying) vs 4xx (permanent, surface to user). ``body``
    contains the response text with bearer tokens redacted.
    """

    def __init__(self, status: int, message: str, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class MalformedCodexResponseError(CodexError):
    """Codex response shape was 2xx but missing expected fields."""


# --- Model coercion ---------------------------------------------------------


# Per-feature OpenAI model substitutions applied when auth_mode is
# subscription. The keys are the lower-cased substring patterns of the
# configured model name; the values are OpenAI models the Codex backend
# is expected to serve.
#
# Calibration is tunable in one place — a follow-up session adjusts
# these based on actual Codex availability and quality observations.
# Per R-A2: video-input features (cinematography_video, description_video)
# fall back to frame mode upstream so we don't send Gemini-specific
# content blocks the Codex endpoint won't accept.
CODEX_MODEL_SUBSTITUTIONS: Tuple[Tuple[str, str], ...] = (
    # (substring-match, openai_model_substitute)
    ("gemini-3.1-flash-lite-preview", "gpt-4o-mini"),
    ("gemini-2.5-flash", "gpt-4o-mini"),
    ("gemini-2.5-pro", "gpt-4o"),
    ("gemini", "gpt-4o-mini"),  # any other gemini -> small OpenAI
    ("claude-sonnet", "gpt-4o"),
    ("claude-opus", "gpt-4o"),
    ("claude", "gpt-4o-mini"),  # any other claude -> small OpenAI
    ("vertex_ai/", "gpt-4o"),
    ("anthropic/", "gpt-4o-mini"),
    ("openrouter/", "gpt-4o-mini"),
)


def coerce_model_for_codex(model: str) -> str:
    """Map a configured model name to a Codex-supported OpenAI model.

    Looks the model up in the substitution table; if no rule matches,
    falls through to the model as-is — useful when the configured model
    is already an OpenAI name (``gpt-4o``, ``gpt-5.2``, ``openai/...``).
    """
    if not model:
        return "gpt-4o-mini"
    lowered = model.lower()
    # Strip provider prefix for the OpenAI passthrough case.
    if lowered.startswith("openai/"):
        return model.split("/", 1)[1]
    for pattern, target in CODEX_MODEL_SUBSTITUTIONS:
        if pattern in lowered:
            return target
    # Non-prefixed gpt-* models pass through.
    return model


# --- Bearer-token redaction -------------------------------------------------


_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+"
)


def _sanitize_for_log(text: str) -> str:
    """Redact bearer-token values from any free-form text.

    Best-effort: matches the common ``Authorization: Bearer <token>``
    pattern (case insensitive). Always run on any string that might
    transit through a logger or an Exception message — httpx error
    strings, in particular, can include the full request representation
    with headers.
    """
    if not text:
        return text
    return _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", text)


# --- Low-level HTTP helper --------------------------------------------------


def _post_completion(
    client: httpx.Client,
    *,
    token_blob: dict,
    payload: dict,
) -> dict:
    """POST a chat-completion request to the Codex backend; return JSON."""
    access_token = token_blob.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise TokenMissingError("ChatGPT OAuth token blob is missing access_token")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    try:
        response = client.post(
            CODEX_COMPLETION_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=60.0,
        )
    except httpx.HTTPError as exc:
        raise CodexBackendError(
            status=0,
            message=_sanitize_for_log(
                f"Network error contacting Codex backend: {exc}"
            ),
        ) from exc

    status = response.status_code
    if status == 401:
        raise TokenExpiredError("Codex backend rejected the bearer token (401)")
    if status == 429:
        try:
            body_text = _sanitize_for_log(response.text)
        except Exception:
            body_text = ""
        raise QuotaExceededError(
            f"Codex backend returned 429 (quota exhausted): {body_text}"
        )
    if status >= 400:
        try:
            body_text = _sanitize_for_log(response.text)
        except Exception:
            body_text = ""
        raise CodexBackendError(
            status=status,
            message=f"Codex backend returned {status}",
            body=body_text,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise MalformedCodexResponseError(
            f"Codex response was non-JSON: {exc}"
        ) from exc


# --- Public API -------------------------------------------------------------


class _NamespaceDict(dict):
    """Dict that also supports attribute access for OpenAI-shape responses.

    Call sites today do ``response.choices[0].message.content`` against
    litellm's ModelResponse (a pydantic object). Subscription-mode responses
    come from us as plain JSON dicts; wrapping them in this class keeps the
    call sites unchanged. ``getattr(response.choices[0], 'finish_reason',
    default)`` continues to work because attribute access falls back to
    dict lookup with the same default semantics as a missing key.

    Nested dicts and list elements are wrapped on the way in so the whole
    response tree is uniformly accessible.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _wrap_response(obj):
    """Recursively wrap dicts/lists so attribute access works downstream."""
    if isinstance(obj, dict):
        return _NamespaceDict(
            {k: _wrap_response(v) for k, v in obj.items()}
        )
    if isinstance(obj, list):
        return [_wrap_response(item) for item in obj]
    return obj


def complete(
    token_blob: dict,
    *,
    model: str,
    messages: list,
    client: Optional[httpx.Client] = None,
    **kwargs,
) -> dict:
    """Run a chat completion against the Codex backend.

    Returns the response dict in OpenAI Chat Completions format
    (``choices[].message.content``, ``usage.*``). Callers that want
    streaming use ``stream_complete`` instead.

    ``kwargs`` is passed through to the backend as additional fields in
    the JSON payload. Sites that previously passed ``response_format``,
    ``tools``, ``tool_choice``, ``temperature``, ``max_tokens``, etc. to
    ``litellm.completion`` flow through unchanged; the Codex backend's
    support for any individual field is what the kwargs-compatibility
    matrix (deferred to a follow-up session) verifies.
    """
    coerced_model = coerce_model_for_codex(model)
    payload = {"model": coerced_model, "messages": messages, **kwargs}
    if client is None:
        with httpx.Client() as owned:
            raw = _post_completion(owned, token_blob=token_blob, payload=payload)
    else:
        raw = _post_completion(client, token_blob=token_blob, payload=payload)
    return _wrap_response(raw)


async def stream_complete(
    token_blob: dict,
    *,
    model: str,
    messages: list,
    client: Optional[httpx.AsyncClient] = None,
    **kwargs,
):
    """Async generator that yields streaming delta chunks from the Codex backend.

    Emits dicts shaped like OpenAI streaming chunks
    (``choices[].delta.content``, optional ``choices[].delta.tool_calls``).
    The caller assembles a final response from the deltas exactly as
    they do for ``litellm.acompletion`` streaming today.

    The exact wire format for Codex streaming is verified during the
    live-smoke-test step (deferred to a follow-up session). For now,
    this implements the OpenAI Server-Sent-Events streaming convention
    — ``data: {json}\\n`` lines terminated by ``data: [DONE]\\n``.
    """
    coerced_model = coerce_model_for_codex(model)
    payload = {"model": coerced_model, "messages": messages, "stream": True, **kwargs}
    access_token = token_blob.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise TokenMissingError("ChatGPT OAuth token blob is missing access_token")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    try:
        async with client.stream(
            "POST",
            CODEX_COMPLETION_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=120.0,
        ) as response:
            status = response.status_code
            if status == 401:
                raise TokenExpiredError("Codex stream rejected the bearer token (401)")
            if status == 429:
                raise QuotaExceededError("Codex stream returned 429 (quota exhausted)")
            if status >= 400:
                try:
                    body_text = _sanitize_for_log(await response.aread())
                    body_text = body_text.decode("utf-8", errors="replace")
                except Exception:
                    body_text = ""
                raise CodexBackendError(
                    status=status,
                    message=f"Codex stream returned {status}",
                    body=body_text,
                )
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    # SSE comment line
                    continue
                if line.startswith("data: "):
                    payload_str = line[len("data: "):]
                    if payload_str.strip() == "[DONE]":
                        return
                    try:
                        import json as _json

                        yield _json.loads(payload_str)
                    except ValueError:
                        # Malformed chunk — log redacted, skip
                        logger.warning(
                            _sanitize_for_log(
                                f"Codex stream emitted non-JSON chunk: {payload_str[:200]}"
                            )
                        )
                        continue
    finally:
        if owns_client:
            await client.aclose()
