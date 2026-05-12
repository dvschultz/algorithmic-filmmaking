---
title: feat: ChatGPT subscription auth (Sign in with ChatGPT)
type: feat
status: active
date: 2026-05-12
deepened: 2026-05-12
origin: docs/brainstorms/2026-05-12-chatgpt-subscription-auth-requirements.md
---

# feat: ChatGPT subscription auth (Sign in with ChatGPT)

## Summary

Add a "Sign in with ChatGPT" auth path so ChatGPT Plus/Pro subscribers can use Scene Ripper's LLM features against their subscription quota. The plan introduces a global auth-mode toggle in settings, a self-contained OAuth flow (PKCE + loopback, the Codex CLI pattern), keyring-backed token storage, a tiny LLM-routing chokepoint that all five direct `litellm.completion()` call sites in `core/analysis/*` migrate through, plus subscription-aware error and quota UX. The existing API-key path remains the alternative and is fully preserved.

---

## Problem Frame

Scene Ripper's AI surface is meaningful but optional — a real segment of power users who already pay for ChatGPT Plus/Pro currently leave the AI features off rather than pay for the OpenAI API on top of their subscription. The product decision to address this unlock (vs. cost-replace) play, the user segment, the scope, and the gray-zone ToS posture are all settled in origin (see `docs/brainstorms/2026-05-12-chatgpt-subscription-auth-requirements.md`). This plan focuses on *how* to deliver R1–R17 cleanly inside the current PySide6 / LiteLLM / spine / MCP architecture without breaking the existing API-key path.

---

## Requirements

This plan implements the requirements from the origin document. Each implementation unit cites the R-IDs it advances.

- R1–R3. OAuth sign-in flow with clear error handling
- R4–R6, R14. Settings auth-mode toggle, identity display, sign-out
- R7. Subscription mode applies to every LLM-backed feature (with one correction below)
- R8. API-key mode unchanged
- R9–R11. Secure storage, refresh, re-auth UX
- R12–R13. Quota-aware UX
- R15–R17. Single-account v1, no impact to non-LLM features, existing API-key configurations preserved

**Origin actors:** A1 (ChatGPT Plus/Pro subscriber), A2 (existing API-key user)

**Origin flows:** F1 (first-time sign-in), F2 (returning user), F3 (expired/revoked token), F4 (switching auth modes), F5 (batch operation near quota)

**Origin acceptance examples:** AE1 (no internet → R3), AE2 (API-key mode unchanged → R8), AE3 (revoked token → R11), AE4 (quota warn + mid-job error → R12, R13), AE5 (mid-conversation mode switch → R5)

---

## Scope Boundaries

- Per-operation auth routing (origin scope boundary — global toggle only)
- Importing existing Codex CLI credentials from `~/.codex/auth.json` (deferred v1.1 polish)
- Claude Max, Gemini OAuth, GitHub Copilot OAuth — architecture leaves room; v1 implements neither
- Multi-account UI / explicit account switching beyond Sign out → Sign in
- Hosted inference proxy (centrally-paid inference) — different product
- Any change to non-LLM features (transcription, scene detection, vision/ML, embeddings, OCR-pipeline non-VLM tier, audio)
- Changing the existing API-key path — it stays functional, untouched
- `gaze` analysis is treated as out of scope for R7 because `core/analysis/gaze.py` is MediaPipe-only with no LLM call. R7 listed it; this is a requirements correction surfaced during research, not a plan-level exclusion of LLM work that actually exists

### Deferred to Follow-Up Work

- Feature flag / gradual rollout gating: the origin doc did not call for one and the auth path is opt-in by user action (sign-in click), but a settings-level flag could be added later if early-access gating becomes desirable
- A pre-flight quota probe of the Codex backend (if/when the backend ever exposes remaining-quota in a header) — current plan uses a heuristic estimator

---

## Context & Research

### Relevant Code and Patterns

**LLM client and routing**
- `core/llm_client.py` — `ProviderConfig`, `ProviderType`, `LLMClient.stream_chat`/`chat` (async), `complete_with_local_fallback` (sync, with Ollama fallback for 3 remix sequencers: storyteller, exquisite_corpus, free_association), `complete_with_enum_constraint` (Ollama-only, used by word_llm_composer via spine — unaffected), `create_provider_config_from_settings`. Note: `core/remix/drawing_vlm.py` uses `LLMClient.chat` (a separate path covered by U6's `LLMClient` update).
- Direct `litellm.completion()` call sites (the surface R7 actually requires touching):
  - `core/analysis/description.py` (lines 325, 683)
  - `core/analysis/cinematography.py` (lines 561, 694)
  - `core/analysis/custom_query.py` (line 160)
  - `core/analysis/ocr.py` (line 265)
  - `core/analysis/shots_cloud.py` (line 152)
- Chat surface: `ui/chat_worker.py`, `ui/chat_panel.py`, `ui/main_window.py` (chat-message flow ~L2195, settings flow ~L1706)

**Settings + keyring (the patterns to mirror)**
- `core/settings.py` — `Settings` dataclass; per-provider keyring constants (`KEYRING_OPENAI_API_KEY`, etc.); symmetric `get_<provider>_api_key()` / `set_<provider>_api_key()` helpers; `_get_password_from_keyring_services()` walks current + legacy service names for migration; service name `com.algorithmic-filmmaking.scene-ripper`
- `ui/settings_dialog.py` — 6-tab dialog; "API Keys" tab uses `QLineEdit(EchoMode=Password)` + Show toggle; save flow `_on_accept` → `_save_to_settings` → `_save_llm_api_keys`; propagation to running components via `ui/main_window.py:_apply_settings`
- Existing keyring access already abstracted through `_set_provider_api_key_in_keyring(keyring_key, value)` and `_get_provider_api_key_from_keyring(keyring_key)` — reuse for the OAuth token blob

**Worker pattern**
- `ui/workers/base.py` — `CancellableWorker(QThread)` with `threading.Event` cancellation; subclasses check `is_cancelled()` between blocking calls (not during)
- `ui/chat_worker.py` — uses `asyncio.run(self._async_run())` inside `run()`, the pattern to mirror for polling-based work
- `ui/workers/free_association_worker.py` — documents the constraint that `litellm.completion` cannot be interrupted mid-call; same applies to `httpx.post()` during OAuth code exchange (acceptable since each call is bounded and we check cancellation between iterations)
- Browser launch convention: `QDesktopServices.openUrl(QUrl(url))` on the main thread (used 10+ places in `ui/main_window.py`)

**Spine layering**
- `core/spine/` is flat, modules sit at `core/spine/<topic>.py`
- `tests/test_spine_imports.py` enforces no `PySide6`, `mpv`, `av`, `faster_whisper`, `paddleocr`, `mlx_vlm` at module import; `httpx` and `keyring` are allowed
- Only `core/spine/words.py` imports from `core/llm_client.py` today, and only lazily inside a function body — the established pattern for cross-layer LLM access from spine

**MCP server**
- `scene_ripper_mcp/server.py` — `FastMCP` with `lifespan` that calls `load_settings()` once at startup (line ~45); LLM-routed tools live in `scene_ripper_mcp/tools/jobs.py` (`start_describe`, `start_analyze_classify`, `start_analyze_cinematography`, `start_custom_query`, `start_analyze_clips`, `start_extract_text`); the server runs as an independent process — it shares the JSON config and the keyring with the GUI but not in-memory state

### Institutional Learnings

From `docs/solutions/`:
- `runtime-errors/transcription-mlx-auto-fallback-groq-key-20260512.md` — the keyring-only / read-at-call-time / pre-flight configuration gate pattern. Mirror it for OAuth tokens (storage, gating, configuration-error surfacing)
- `runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` — `Qt.UniqueConnection` + per-handler guard flag + `@Slot()` decorator + guard reset on new operation. Mandatory for the OAuth worker (a duplicate `auth_complete` could double-write the token or open two browser tabs)
- `ui-bugs/pyside6-thumbnail-source-id-mismatch.md` and `ui-bugs/timeline-widget-sequence-mismatch-20260124.md` — single live source of truth; do not cache auth-mode at construction time, read it at call time from `core/settings.py`
- `logic-errors/circular-import-config-consolidation.md` — auth abstraction lives in a neutral spine-safe module (no PySide6/LiteLLM at top level), not inside `core/llm_client.py` or `core/chat_tools.py`
- `security-issues/url-scheme-validation-bypass.md` — validate scheme allowlist (`https://` for OpenAI, `http://localhost:*` for the loopback redirect) before any URL is used
- `reliability-issues/subprocess-cleanup-on-exception.md` — wrap any transient resources (loopback HTTP server, polling timers) in `try/finally` inside the worker's `run()`

### External References

- Codex CLI source (open-source) — reference implementation of the OAuth flow against OpenAI's Codex auth endpoint. Plan-time: borrow the client_id, endpoints, and scopes from Codex CLI's source as the starting point; implementation discovers the exact constants
- Hermes-agent (Nous Research) — concrete demonstration of the same pattern in a desktop app using device-code phrasing for what is in fact PKCE + loopback. Reference for token storage shape and auth-mode UX
- Origin doc Dependencies / Assumptions — captures the ToS gray-zone posture and what breaks if OpenAI revokes the client_id

---

## Key Technical Decisions

- **The LLM-routing chokepoint is introduced and the five direct `core/analysis/*` call sites migrate through it.** The alternative ("subscription mode is chat-only in v1") violates R7. Migration is mechanical: one new function in `core/llm_client.py` mirroring the `litellm.completion()` signature, five callsite swaps. (User-confirmed at Phase 5.1.5.)
- **OAuth flow is PKCE + loopback redirect, not RFC 8628 device-code.** The origin doc said "device-code" loosely; the actual Codex CLI flow OpenAI exposes is authorization-code + PKCE + `http://localhost:<port>` redirect. Plan uses the latter throughout. Same UX from the user's perspective (browser opens, they sign in, app gets a token).
- **`gaze` is excluded from R7's scope** because `core/analysis/gaze.py` performs no LLM call. Treated as a requirements correction, not a planning decision. (see origin: R7)
- **The new auth abstraction lives in `core/spine/chatgpt_auth.py`** (spine-safe, no top-level PySide6/LiteLLM imports). This avoids the circular-import risk of putting it in `core/llm_client.py` or `core/chat_tools.py`, and pre-positions Approach 4 (Claude Max, Gemini OAuth) as future drop-in modules behind the same `AuthMode` interface.
- **Auth-mode state is read at call time from `core/settings.py`**, never cached in worker or client constructors. This is what makes AE5 (mid-conversation switch without restart) structurally easy — workers are already short-lived (chat worker is recreated per message; analysis workers re-resolve `api_key` per-call).
- **The Codex backend gets its own thin HTTP client at `core/codex_client.py`** (httpx-based). LiteLLM does not speak the Codex backend's request/response shape; trying to force-fit it into LiteLLM as a custom provider is more invasive than a small, focused client.
- **OAuth tokens are stored as a single JSON-serialized blob under one new keyring key** (`KEYRING_CHATGPT_OAUTH_TOKEN`). Storing `access_token`, `refresh_token`, `expires_at`, `account_email` separately would create atomicity hazards during refresh. The blob is reread on each LLM call (microseconds).
- **MCP server reloads settings + token on each LLM-routed tool invocation.** Cheap (JSON read + one keyring read); avoids the staleness trap where a GUI sign-in is invisible to a running MCP server.
- **`complete_with_local_fallback()` skips the Ollama fallback when subscription mode is active.** Falling back from a Plus subscription to a (potentially uninstalled) local Ollama is worse UX than surfacing the auth or quota error; subscription users explicitly chose subscription. The fallback path stays intact for API-key mode (unchanged).
- **No environment-variable override for the OAuth token.** API keys have env-var overrides for power users; OAuth tokens are short-lived secrets with refresh semantics — keyring-only.
- **Test posture for OAuth + worker units: test-first.** Security-sensitive, no existing patterns in the repo, high-blast-radius bugs (token persistence, duplicate signals, scheme validation) all benefit from failing tests up front. Other units follow the project's normal test-alongside posture.

---

## Open Questions

### Resolved During Planning

- *Should the auth-mode toggle live in a new tab or in API Keys?* — Place in the existing "API Keys" tab. Grouped naturally with other LLM credentials; avoids adding tab surface. (See U5.)
- *Codex backend or full LiteLLM custom provider?* — Standalone `core/codex_client.py`. (See Key Technical Decisions.)
- *Token blob format in keyring?* — Single JSON blob under one keyring key. (See Key Technical Decisions.)
- *MCP server staleness handling?* — Per-call settings reload. (See Key Technical Decisions.)
- *Ollama fallback behavior in subscription mode?* — Skipped. (See Key Technical Decisions.)
- *PKCE + loopback or RFC 8628 device-code?* — PKCE + loopback (the actual Codex CLI flow). (See Key Technical Decisions.)

### Deferred to Implementation

- *Exact OAuth `client_id`, authorization endpoint, token endpoint, and scopes.* Pulled from Codex CLI's open-source source at implementation time; constants land in `core/spine/chatgpt_auth.py`. The plan does not lock these in — they may change as OpenAI evolves the endpoint.
- *Exact request/response shape for the Codex backend.* Discovered by reading Codex CLI's client; encoded in `core/codex_client.py`.
- *Whether the Codex backend returns a remaining-quota signal in any response header.* If yes, the heuristic estimator in U8 can be replaced with a real check; if no, the heuristic stands.
- *Specific user-facing wording for the quota warning, the re-auth prompt, the sign-in success toast.* Wording lands during implementation review.
- *Exact heuristic for the pre-flight quota warning* (tokens per clip × clip count × safety margin). Calibrated against real Codex behavior during implementation.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

**Layered shape:**

```
                  ┌──────────────────────────────────────────┐
                  │  ui/settings_dialog.py                    │
                  │  ui/workers/oauth_worker.py               │
                  │  ui/chat_worker.py, ui/chat_panel.py      │
                  └──────────────────┬───────────────────────┘
                                     │ Qt signals, settings I/O
                                     ▼
                  ┌──────────────────────────────────────────┐
                  │  core/llm_client.py                       │
                  │   - complete_routed()  ← new chokepoint   │
                  │   - existing helpers route through it     │
                  └────┬───────────────────────────┬─────────┘
                       │ auth_mode == api_key      │ auth_mode == subscription
                       ▼                            ▼
              litellm.completion()       core/codex_client.py
                                                   │
                                                   ▼
                                         OpenAI Codex backend
                                                   ▲
                                                   │ bearer token (refreshed)
                  ┌──────────────────────────────────────────┐
                  │  core/spine/chatgpt_auth.py               │
                  │   - AuthMode enum                         │
                  │   - load_active_auth() (reads settings    │
                  │     + keyring at call time)               │
                  │   - PKCE/loopback OAuth flow              │
                  │   - token refresh                         │
                  └────────────────┬─────────────────────────┘
                                   │ token blob (JSON) ⇄ keyring
                                   ▼
                       core/settings.py keyring helpers
                       (KEYRING_CHATGPT_OAUTH_TOKEN)
```

**Auth-mode state machine:**

```
   ┌──────────────────┐  user picks API key in settings   ┌──────────────────┐
   │  Unauthenticated │ ────────────────────────────────► │   API-key mode   │
   │ (first launch /  │                                    │ (existing path,   │
   │  signed out)     │ ◄──────────────────────────────── │  unchanged)       │
   └────────┬─────────┘  user clicks "Sign out" /          └─────────┬────────┘
            │            clears API key                              │
            │ user picks subscription + sign-in succeeds              │ user switches
            ▼                                                          ▼
   ┌──────────────────┐  token expires + refresh fails    ┌──────────────────┐
   │ Subscription mode│ ────────────────────────────────► │  Subscription:   │
   │  + valid token   │                                    │  re-auth needed   │
   └──────────────────┘ ◄──────────────────────────────── └──────────────────┘
                          user signs in again
```

---

## Implementation Units

### U1. Auth-mode and token storage on the Settings layer

**Goal:** Extend `core/settings.py` with the new auth-mode field, OAuth keyring constants, and symmetric token-blob helpers — the foundation every other unit reads from.

**Requirements:** R4, R6, R9, R14, R17

**Dependencies:** None

**Files:**
- Modify: `core/settings.py`
- Test: `tests/test_settings.py`

**Approach:**
- Add to `Settings` dataclass: `auth_mode: str = "api_key"` (values: `"api_key"`, `"subscription"`), `chatgpt_account_email: str = ""` (display-only, non-secret)
- Add load/save handling for both fields in `_load_from_json` / `_settings_to_json`
- Add new keyring constant `KEYRING_CHATGPT_OAUTH_TOKEN`
- Add helpers `get_chatgpt_oauth_token()` / `set_chatgpt_oauth_token(token_blob: dict | None)` / `clear_chatgpt_oauth_token()`, mirroring the existing per-provider keyring helpers but operating on a JSON-serialized blob
- Token blob shape: `{"access_token", "refresh_token", "id_token", "expires_at_unix", "account_email"}` — opaque to `core/settings.py`, parsed by `core/spine/chatgpt_auth.py`
- No env-var override for the token (intentional, see Key Technical Decisions); env-var override fine for `auth_mode` if desired (low priority)

**Execution note:** Test-first. Token persistence is high-blast-radius; mocks should drive the design.

**Patterns to follow:**
- `core/settings.py` existing keyring helpers (`_set_provider_api_key_in_keyring`, `_get_password_from_keyring_services`)
- Tests at `tests/test_settings.py:209-227` for keyring mocking via `sys.modules` injection

**Test scenarios:**
- Happy path: set token blob → get returns the same blob → clear → get returns None
- Edge case: JSON serialization round-trip preserves all fields and types (especially `expires_at_unix` int)
- Edge case: `auth_mode` defaults to `"api_key"` for migrating users (existing config.json with no auth_mode key)
- Error path: keyring set fails (no Keychain access) → helper returns False, no crash
- Error path: keyring get returns malformed JSON → helper returns None, logs warning
- Error path: keyring delete on missing key → no crash
- Integration: `auth_mode = "subscription"` persists across `save_settings()` / `load_settings()` round trip

**Verification:**
- `tests/test_settings.py` passes for the new helpers
- Loading an existing config.json without `auth_mode` produces a `Settings` with `auth_mode = "api_key"` (backward compatibility)

---

### U2. Auth-state module (spine-safe)

**Goal:** Provide a neutral, zero-dependency module that owns the active auth state — what mode is on, is the token valid, when does it expire — read at call time by every consumer.

**Requirements:** R5, R7, R8 (via the read-at-call-time discipline that makes AE5 work)

**Dependencies:** U1

**Files:**
- Create: `core/spine/chatgpt_auth.py`
- Test: `tests/test_chatgpt_auth.py`

**Approach:**
- Define `AuthMode` enum (`API_KEY`, `SUBSCRIPTION`) and `AuthIdentity` dataclass (`account_email`, `expires_at_unix`, `seconds_until_expiry`)
- Top-level imports: stdlib only (`time`, `json`, `dataclasses`, `enum`). No PySide6, no LiteLLM, no httpx at module scope. Lazy-import `keyring` inside functions if needed (it's already used by `core/settings.py` so it'd be in `sys.modules` anyway).
- Pure-function API:
  - `load_active_auth() -> tuple[AuthMode, AuthIdentity | None]` — reads `Settings.auth_mode` + the token blob from keyring; returns the current state
  - `is_token_expired(identity: AuthIdentity, leeway_seconds: int = 60) -> bool`
  - Constants for `CLIENT_ID`, `AUTHORIZATION_ENDPOINT`, `TOKEN_ENDPOINT`, `REDIRECT_URI_HOST`, `SCOPES` — borrowed from Codex CLI source at implementation time (Deferred to Implementation question)
- `tests/test_spine_imports.py` must continue to pass — this module must NOT trigger any forbidden imports

**Execution note:** Test-first. The boundary test is enforced; structural mistakes here fail loudly.

**Patterns to follow:**
- `core/spine/words.py:451` — lazy import of `core/llm_client` inside a function body, the existing pattern for cross-layer access from spine
- `tests/test_spine_imports.py` — list of forbidden imports

**Test scenarios:**
- Happy path: settings configured for API-key mode → `load_active_auth()` returns `(API_KEY, None)`
- Happy path: settings configured for subscription mode with a valid blob → `load_active_auth()` returns `(SUBSCRIPTION, AuthIdentity(...))`
- Edge case: settings configured for subscription but no blob in keyring → `(SUBSCRIPTION, None)` (the "re-auth needed" state)
- Edge case: token expires at `now - 1s` → `is_token_expired()` returns True; at `now + 120s` with leeway 60 → False
- Integration: importing `core.spine.chatgpt_auth` does not pull `PySide6`, `litellm`, or `mlx_vlm` into `sys.modules` (extend `tests/test_spine_imports.py`)

**Verification:**
- `tests/test_chatgpt_auth.py` passes
- `tests/test_spine_imports.py` passes with the new module in the spine list

---

### U3. OAuth flow (PKCE + loopback, GUI-free)

**Goal:** Implement the PKCE + loopback OAuth flow against OpenAI's Codex auth endpoint as a pure-Python module — no Qt dependency, fully unit-testable.

**Requirements:** R1, R2, R3, R10

**Dependencies:** U2

**Files:**
- Create: `core/spine/chatgpt_oauth_flow.py`
- Test: `tests/test_chatgpt_oauth_flow.py`

**Approach:**
- Pure-Python module using `httpx` (already a core dep) and stdlib `http.server`, `secrets`, `hashlib`, `base64`, `urllib.parse`
- Function-level API (no class state):
  - `generate_pkce_pair() -> tuple[verifier, challenge]`
  - `build_authorization_url(verifier_challenge, state) -> tuple[url, redirect_uri]` — picks a random free port on localhost for the loopback receiver
  - `run_loopback_listener(port, expected_state, timeout_seconds) -> AuthorizationCode | None` — spawns an `http.server.HTTPServer`, accepts one redirect, validates `state`, returns the `code` query param (or None on timeout/cancel)
  - `exchange_code_for_token(code, verifier, redirect_uri) -> TokenBlob` — POST to TOKEN_ENDPOINT
  - `refresh_token(refresh_token: str) -> TokenBlob` — POST to TOKEN_ENDPOINT with `grant_type=refresh_token`
- URL scheme validation everywhere: `https://` for OpenAI endpoints (constant), `http://localhost:*` for the redirect_uri (validated against scheme + hostname allowlist before use). See `docs/solutions/security-issues/url-scheme-validation-bypass.md`.
- All HTTP via `httpx.Client` so the worker (U4) can inject a `httpx.MockTransport` for tests
- `try/finally` around the loopback server: `server.server_close()` on every exit path (success, error, timeout, cancellation). See `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`.
- Errors raised as specific exception classes: `OAuthCancelledError`, `OAuthTimeoutError`, `OAuthHTTPError`, `OAuthMalformedResponseError`

**Execution note:** Test-first throughout. Security-sensitive, externally-dependent, no prior art in the repo.

**Technical design:** *(directional, not specification)*

```
generate_pkce_pair() ──► (verifier, challenge)
                              │
build_authorization_url(challenge, state) ──► (auth_url, redirect_uri_with_port)
                              │
   ┌──────────────────────────┘
   │  GUI (U4/U5) opens auth_url in browser
   ▼
run_loopback_listener(port, expected_state, timeout) ──► AuthorizationCode
                              │
exchange_code_for_token(code, verifier, redirect_uri) ──► TokenBlob
                              │
                              ▼
                        return to caller
```

**Patterns to follow:**
- `core/transcription.py` for `httpx.Client` request patterns
- `tests/test_settings.py` for HTTP mocking via test-injected transports

**Test scenarios:**
- Happy path: PKCE pair satisfies S256 transform (`challenge == BASE64URL(SHA256(verifier))` without padding)
- Happy path: `build_authorization_url` produces a URL with `response_type=code`, `code_challenge_method=S256`, the right scopes, and a loopback `redirect_uri` whose port is free
- Happy path: `run_loopback_listener` returns the `code` when the loopback receives a valid redirect matching `expected_state`
- Edge case: redirect arrives with mismatched `state` → returns None and emits a specific log line (state-mismatch is a CSRF guard, must be explicit)
- Edge case: redirect URL has scheme `javascript:` or `file:` → rejected by validator before any parsing
- Edge case: loopback timeout (no redirect within window) → returns None, server cleanly closed
- Error path: `exchange_code_for_token` 4xx → `OAuthHTTPError` with the status code and (non-secret-bearing) response body
- Error path: response missing `access_token` → `OAuthMalformedResponseError`
- Error path: `refresh_token` 401 → `OAuthHTTPError(401)`; caller (U6) interprets as "re-auth required"
- Integration: full flow end-to-end via `httpx.MockTransport` — generate pair → build URL → simulate redirect → exchange → assert token blob shape

**Verification:**
- `tests/test_chatgpt_oauth_flow.py` passes with full path coverage
- Loopback server confirmed closed (via `port` reusable on next iteration) on every exit path

---

### U4. OAuth worker (Qt wrapper around the flow)

**Goal:** Wrap U3 in a `CancellableWorker` that opens the browser, runs the loopback listener, and emits typed signals back to the settings dialog — without freezing the UI or duplicating signals.

**Requirements:** R1, R2, R3

**Dependencies:** U3

**Files:**
- Create: `ui/workers/oauth_worker.py`
- Test: `tests/test_oauth_worker.py`

**Approach:**
- Subclass `CancellableWorker`. `run()` uses `asyncio.run(...)` over an async coroutine that orchestrates the U3 functions (the polling-style flow benefits from `await asyncio.sleep()` between cancellation checks, mirroring `ui/chat_worker.py`)
- Signals (each declared on the class):
  - `authorization_url_ready = Signal(str)` — emitted when the URL is built; the dialog's slot calls `QDesktopServices.openUrl(QUrl(url))` on the main thread
  - `auth_complete = Signal(dict)` — emitted exactly once with the token blob; settings dialog persists it via U1 helpers
  - `auth_failed = Signal(str, str)` — `(category, user_message)`; categories: `"cancelled"`, `"timeout"`, `"network"`, `"rejected"`, `"malformed"`
- All `connect()` calls in the settings dialog use `Qt.UniqueConnection`. The worker reset its internal `_finished_handled` guard at the top of `run()` (mirrors `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`). Handler slots in the dialog use `@Slot()` decorators and guard their own `_oauth_in_progress` flag.
- `try/finally` in `run()` ensures the loopback server is closed even on cancellation
- Cancellation between every async iteration; in-flight `httpx.post()` is non-interruptible (acknowledged constraint from `ui/workers/free_association_worker.py:8`), but each call is short

**Execution note:** Test-first. Signal-duplication and state-mutation-on-failure are the two prior bug classes this worker has to avoid.

**Patterns to follow:**
- `ui/chat_worker.py` — `asyncio.run()` inside `run()`
- `ui/workers/free_association_worker.py` — cancellation discipline around blocking HTTP
- `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md` — UniqueConnection + guard flag + @Slot pattern

**Test scenarios:**
- Happy path: full flow completes → `authorization_url_ready` fires once, `auth_complete` fires once with the expected token blob, no `auth_failed`
- Happy path: signals emitted on the worker's thread, slots invoked on the main thread (Qt-standard for queued signals)
- Edge case: user cancels mid-flow → `auth_failed("cancelled", ...)` fires exactly once; `auth_complete` does not fire; settings/keyring untouched
- Edge case: loopback timeout → `auth_failed("timeout", ...)` exactly once; loopback server confirmed closed (port reusable)
- Edge case: two successive sign-in attempts in the same dialog session → second attempt's signals do not bleed into the first attempt's slots (guard flag reset confirmed)
- Error path: network error during token exchange → `auth_failed("network", ...)`; auth-mode in settings remains unchanged (R3)
- Error path: malformed token response → `auth_failed("malformed", ...)`
- Integration: `auth_complete` handler in a test harness writes blob to a mock keyring and updates `Settings.auth_mode` to `"subscription"`; no concurrent write or duplicate write

**Verification:**
- `tests/test_oauth_worker.py` passes
- All `connect()` sites use `Qt.UniqueConnection`; `@Slot()` decorators present on every handler
- No path through the worker mutates `Settings` directly — only via signals received by the settings dialog (single source of truth for state changes)

---

### U5. Settings dialog UI for sign-in

**Goal:** Extend the existing settings dialog with the auth-mode toggle, the "Sign in with ChatGPT" affordance, identity display, and sign-out — wired to U4.

**Requirements:** R4, R5, R6, R14, R17

**Dependencies:** U1, U4

**Files:**
- Modify: `ui/settings_dialog.py`
- Test: `tests/test_settings_dialog.py`

**Approach:**
- Place the new controls in the existing "API Keys" tab (`_create_api_keys_tab`), at the top — above the per-provider key fields. Visually grouped as "OpenAI authentication" with a `QRadioButton` pair: `Sign in with ChatGPT (use my subscription)` / `Use OpenAI API key`.
- Subscription branch:
  - When signed out: shows a "Sign in with ChatGPT" button. Click → instantiates `OAuthWorker`, connects signals with `Qt.UniqueConnection`, starts the worker. `authorization_url_ready` slot calls `QDesktopServices.openUrl(QUrl(url))`. `auth_complete` slot persists the blob via U1 helpers, updates the identity label, and clears the in-progress guard.
  - When signed in: shows the signed-in identity (`Settings.chatgpt_account_email`) and a "Sign out" button. Sign out → `clear_chatgpt_oauth_token()` + reset `chatgpt_account_email`.
- API-key branch: the existing per-provider key fields, unchanged. Hidden visually when subscription radio is selected (still functional — just not the active mode).
- `_save_to_settings` writes `auth_mode` and `chatgpt_account_email`; keyring writes are gated as today (`_save_llm_api_keys`)
- Mode propagation: existing `_apply_settings` in `ui/main_window.py` already recreates per-message chat workers and re-resolves api_keys per-call inside analysis modules; no additional propagation code needed — this is what makes AE5 structurally easy

**Patterns to follow:**
- `ui/settings_dialog.py:_create_api_keys_tab` (line ~949) for layout
- `ui/main_window.py:_apply_settings` (line ~1718) for the existing propagation path

**Test scenarios:**
- Happy path: open dialog with `auth_mode = "api_key"` → API-key radio selected, API-key fields visible
- Happy path: switch radio to subscription, click Sign In, mock OAuthWorker emits `auth_complete` with blob → blob persisted, identity label shows the account email, save dialog → `Settings.auth_mode = "subscription"` and `chatgpt_account_email` set
- Happy path: open dialog with `auth_mode = "subscription"` and a valid stored blob → subscription radio selected, identity shown, Sign Out button present
- Edge case: click Sign Out → `clear_chatgpt_oauth_token()` invoked, identity label cleared, Sign In button reappears
- Error path: OAuthWorker emits `auth_failed("cancelled", ...)` → no persistence; auth-mode unchanged from its pre-click state; user-facing message visible briefly in the dialog
- Error path: OAuthWorker emits `auth_failed("network", ...)` → same as cancelled, plus the network category surfaces a "could not reach OpenAI" message (covers AE1, R3)
- Error path: OAuthWorker emits `auth_failed("timeout", ...)` → Sign-in button re-enabled, brief in-dialog message ("sign-in timed out — the browser window may not have opened; try again"); auth-mode unchanged
- Error path: OAuthWorker emits `auth_failed("rejected", ...)` and `auth_failed("malformed", ...)` → mirror cancelled handling with category-specific user-facing message; auth-mode unchanged
- Integration: switch from API-key mode to subscription mode and save → next `load_settings()` returns `auth_mode = "subscription"`; existing OpenAI API key in keyring is *not* deleted (R17)

**Verification:**
- `tests/test_settings_dialog.py` passes
- Manual: open dialog, switch modes, save, reopen — state round-trips
- AE2 holds: a user who never touches subscription mode sees zero behavioral change

---

### U6. LLM-routing chokepoint + Codex backend client

**Goal:** Introduce one routing function in `core/llm_client.py` and a thin `core/codex_client.py` that backs it for subscription mode. This is the single seam every LLM call will go through after U7.

**Requirements:** R7, R8, R10, R11

**Dependencies:** U2, U3 (token refresh logic)

**Files:**
- Modify: `core/llm_client.py`
- Create: `core/codex_client.py`
- Test: `tests/test_llm_routing.py`, `tests/test_codex_client.py`

**Approach:**
- New function in `core/llm_client.py`: `complete_routed(model, messages, **kwargs)` — same signature shape as `litellm.completion`. At call time:
  1. `load_active_auth()` from U2
  2. If `AuthMode.API_KEY` → delegate to `litellm.completion(model, messages, **kwargs)` with the api_key resolved as today via `get_api_key_for_model(model)`. Behavior bit-identical to current code paths (R8)
  3. If `AuthMode.SUBSCRIPTION` → delegate to `core/codex_client.py:complete(token_blob, model, messages, **kwargs)`. If the token is near expiry, refresh first via U3
- `core/codex_client.py`:
  - Thin httpx-based client targeting the Codex backend
  - `complete(token_blob, model, messages, ...)` POSTs to the Codex completion endpoint with `Authorization: Bearer <access_token>`
  - On 401 → `TokenExpiredError` (caught by U8)
  - On 429 → `QuotaExceededError` (caught by U9)
  - On other 4xx/5xx → `CodexBackendError(status, body)`
  - Exact URL, request shape, and response normalization are Deferred to Implementation — designed to expose the same response surface (`choices[].message.content`, `usage.*`) that callers expect today
- Existing `LLMClient.stream_chat` / `chat` / `complete_with_local_fallback` are updated to delegate to `complete_routed` for the cloud branch. The Ollama-fallback inside `complete_with_local_fallback` is gated to API-key mode only (in subscription mode, the auth/quota errors propagate so U8/U9 can handle them).

**Patterns to follow:**
- `tests/test_llm_client_fallback.py` — `monkeypatch.setattr("litellm.completion", fake)` for mocking
- `core/transcription.py` for `httpx.Client` usage patterns

**Test scenarios:**
- Happy path (API-key mode): `complete_routed(model="gpt-4o", messages=[...])` calls `litellm.completion` with the OpenAI api_key from keyring; return shape matches current behavior bit-for-bit (R8)
- Happy path (subscription mode): `complete_routed` calls into `core/codex_client.py:complete` with the bearer token; response normalized to the same dict shape callers expect
- Happy path (token near expiry in subscription mode): refresh is called transparently before the LLM call; new blob persisted to keyring; LLM call proceeds with the new access_token
- Edge case: `auth_mode = "subscription"` but no token in keyring → `TokenMissingError`; UI catches this in U8
- Error path: `core/codex_client.py:complete` 401 → `TokenExpiredError`; routing helper attempts one refresh; if refresh also 401 → raise to caller
- Error path: 429 → `QuotaExceededError` with the response body if it carries quota signals
- Integration: a mock chat-agent call in subscription mode flows through `complete_routed` → `core/codex_client.py:complete` → `httpx.MockTransport` (no live network); assert the exact bearer header was sent and the response was normalized correctly
- Integration: `complete_with_local_fallback` in subscription mode with the Codex backend returning 429 → `QuotaExceededError` raised, Ollama fallback NOT invoked

**Verification:**
- `tests/test_llm_routing.py` and `tests/test_codex_client.py` pass
- API-key mode passes every existing `core/llm_client.py` test unmodified
- No call to `litellm.completion` survives outside `complete_routed` (post U7)

---

### U7. Migrate direct `litellm.completion()` call sites

**Goal:** Route the five direct `core/analysis/*` call sites through `complete_routed` so that R7 is delivered for every LLM-backed feature, not just chat.

**Requirements:** R7

**Dependencies:** U6

**Files:**
- Modify: `core/analysis/description.py`
- Modify: `core/analysis/cinematography.py`
- Modify: `core/analysis/custom_query.py`
- Modify: `core/analysis/ocr.py`
- Modify: `core/analysis/shots_cloud.py`
- Test: extend `tests/test_description.py`, `tests/test_cinematography.py`, `tests/test_custom_query.py`, `tests/test_ocr.py`, `tests/test_shots_cloud.py` (file names match existing patterns; create if missing)

**Approach:**
- In each file, replace `litellm.completion(model=..., messages=..., api_key=..., **kwargs)` with `complete_routed(model=..., messages=..., **kwargs)`. The `api_key` resolution inside `complete_routed` handles API-key mode, so the explicit `api_key=` kwarg can be dropped (or left for `complete_routed` to ignore in subscription mode).
- This is purely mechanical — five call sites, one swap each, identical call signatures
- No behavior change in API-key mode (R8). Subscription mode now reaches these features (R7).

**Patterns to follow:**
- Existing call sites themselves — the swap is to use the new helper, not to redesign the calls

**Test scenarios:**
- Happy path per file (API-key mode): existing test expectations preserved bit-for-bit
- Integration per file (subscription mode): a parameterized test fixture runs each feature under subscription mode with a `httpx.MockTransport` against `core/codex_client.py` — confirm the call reaches the Codex backend with the right model name and messages
- Edge case: per-file feature called with `auth_mode = "subscription"` but no token → `TokenMissingError` surfaces (handled by U8)

**Verification:**
- All five test files pass under both auth modes
- Grep confirms zero `litellm.completion` calls remain in `core/analysis/*`
- Existing functional tests for description/cinematography/custom-query/ocr/shots_cloud unchanged in API-key mode

---

### U8. Subscription-mode error UX: re-auth and quota

**Goal:** Translate the new exception types from U6 (`TokenExpiredError`, `TokenMissingError`, `QuotaExceededError`, `CodexBackendError`) into user-visible UX — re-auth prompts and quota-specific errors.

**Requirements:** R3 (cases that surface from runtime), R11, R12, R13

**Dependencies:** U6, U7

**Files:**
- Modify: `core/llm_client.py` (catch points within `complete_with_local_fallback`, `LLMClient.chat`/`stream_chat`)
- Modify: `ui/chat_panel.py` (chat error display)
- Modify: `ui/chat_worker.py` (catch in async path)
- Modify: `ui/workers/description_worker.py`, `ui/workers/cinematography_worker.py`, `ui/workers/custom_query_worker.py`, `ui/workers/ocr_worker.py`, `ui/workers/shots_cloud_worker.py` (mirror the error-catch pattern across each batch worker matched to the 5 analysis files migrated in U7; confirm exact filenames via grep on the analysis-module imports)
- Modify: `ui/main_window.py` (re-auth dialog launcher)
- Test: `tests/test_subscription_error_ux.py`

**Approach:**
- Define `TokenExpiredError`, `TokenMissingError`, `QuotaExceededError`, `CodexBackendError` in `core/llm_client.py` (or a small `core/llm_errors.py` if it grows)
- `ChatAgentWorker` and analysis workers catch these specific types and emit error signals carrying a structured payload (`category`, `user_message`)
- Main window listens for `category="token_required"` and `category="token_expired"` → opens a small dialog with "Sign in again" / "Switch to API key mode" actions
- Quota errors render with explicit Plus framing: "You've hit your ChatGPT Plus quota for this 3-hour window. Wait until <X>, or switch to API key mode to keep working." (Exact wording: Deferred to Implementation.)
- AE3 covers token-expired surfaces; the same pattern handles `TokenMissingError` (the user picked subscription mode but never signed in, or signed out)

**Patterns to follow:**
- `ui/chat_panel.py` existing error rendering for the chat surface
- `ui/workers/free_association_worker.py:8`-style cancellation discipline for the analysis workers
- `docs/solutions/runtime-errors/transcription-mlx-auto-fallback-groq-key-20260512.md` — pre-flight configuration gate pattern (a `TokenMissingError` surfaced before the LLM call is the parallel)

**Test scenarios:**
- Happy path: Codex backend returns 401 → `TokenExpiredError` raised → routing helper attempts one refresh → if refresh succeeds, call retries transparently
- Happy path: Codex backend returns 429 → `QuotaExceededError` raised → worker emits structured error → UI renders quota-specific message (covers AE4)
- Error path: Codex backend returns 401, refresh also fails → `TokenExpiredError` surfaces; chat UI shows re-auth prompt (covers AE3, R11)
- Error path: `auth_mode = "subscription"` but no stored token → `TokenMissingError` immediately on first LLM call; UI prompts sign-in or mode switch
- Error path: unexpected 5xx from Codex backend → `CodexBackendError`; UI renders a generic-but-actionable message ("OpenAI's Codex service had a hiccup — try again or switch to API key mode")
- Integration: mid-conversation token expiry — chat worker receives `TokenExpiredError`, surfaces the prompt; user clicks "Sign in again", succeeds, the next chat message works without app restart (covers R5, AE5)

**Verification:**
- `tests/test_subscription_error_ux.py` passes
- AE3 and AE4 verifiable end-to-end via mock backend responses
- No raw `TokenExpiredError`/`QuotaExceededError` stack traces ever reach the user

---

### U9. Quota pre-flight estimator

**Goal:** Implement R12 — warn before starting a batch operation likely to exceed remaining quota. Heuristic-based, since the Codex backend's remaining-quota surface is not assumed.

**Requirements:** R12

**Dependencies:** U6, U8

**Files:**
- Create: `core/quota_estimator.py` (small, pure-function)
- Modify: `ui/workers/description_worker.py`, `ui/workers/cinematography_worker.py` (or whatever the matching worker file is), and the batch-runner workers — wire the estimator before `run()` starts
- Test: `tests/test_quota_estimator.py`

**Approach:**
- `core/quota_estimator.py` exposes `estimate_batch_load(operation, count) -> dict` returning `{estimated_messages, estimated_tokens}` based on per-operation heuristics (e.g., describe: ~1 message per clip; classify: ~1 message per clip; custom-query: ~1 message per clip per query)
- Worker code, when in subscription mode, calls the estimator before starting and compares to a configurable per-3-hour budget (default value tuned by feel — see Deferred to Implementation)
- If estimated load × safety margin > assumed remaining budget → emit a `quota_warning` signal; UI shows a confirm dialog: "This operation may exceed your ChatGPT Plus quota. Proceed? / Cancel / Switch to API key mode"
- The warning is a heuristic — when subscription users hit quota anyway, U8 handles the mid-job error path (covers AE4)
- API-key mode skips the estimator entirely (R8 unchanged)

**Patterns to follow:**
- Settings dialog confirm patterns for the "proceed / switch mode" UX
- `core/feature_registry.py` for the small-stateless-helper-with-tests style

**Test scenarios:**
- Happy path: `estimate_batch_load("describe", 200)` returns a reasonable estimate that triggers a warning under default budget
- Happy path: `estimate_batch_load("describe", 5)` does not trigger a warning
- Edge case: unknown operation name → estimator returns zero with a debug log (no false alarm)
- Edge case: per-operation heuristic constants tunable via a single module-level dict (so calibration changes are one-line)
- Integration: a mock 200-clip describe job in subscription mode shows the warning dialog before starting; in API-key mode, the dialog never shows

**Verification:**
- `tests/test_quota_estimator.py` passes
- AE4's pre-flight branch verifiable end-to-end (mock the dialog confirm)
- Per-operation heuristics documented as tunable, not load-bearing

---

### U10. MCP server: per-call settings reload for LLM-routed tools

**Goal:** Ensure the MCP server sees auth-mode changes made via the GUI without requiring a restart.

**Requirements:** R7 (consistency: the MCP surface must respect the same auth-mode as the GUI)

**Dependencies:** U6 (routing helper used inside spine/analysis paths must be live)

**Files:**
- Modify: `scene_ripper_mcp/server.py` (lifespan / context construction)
- Modify: `scene_ripper_mcp/tools/jobs.py` (LLM-routed tool entry points: `start_describe`, `start_analyze_classify`, `start_analyze_cinematography`, `start_custom_query`, `start_analyze_clips`, `start_extract_text`)
- Test: `scene_ripper_mcp/tests/test_auth_mode_propagation.py`

**Approach:**
- Add `get_current_settings()` helper in `scene_ripper_mcp/server.py` (or appropriate module) that calls `load_settings()` and returns a fresh `Settings` instance each call. JSON read + keyring read is microseconds; safe at per-tool-call frequency
- Each LLM-routed tool calls `get_current_settings()` at the top of its handler (or the spine `analyze` calls do it). After U6/U7, the analysis modules already use `complete_routed`, which reads auth state at call time — so this unit's actual work is making sure no settings caching in the MCP server defeats that read.
- ProjectModel and other non-auth context can remain cached at lifespan level — only the auth-related read needs to be per-call

**Patterns to follow:**
- `scene_ripper_mcp/server.py:lifespan` for the existing once-at-startup pattern (modify only the auth path)
- `core/spine/chatgpt_auth.py:load_active_auth()` from U2 — the right entry point

**Test scenarios:**
- Happy path: change `auth_mode` and write a token blob to keyring; invoke a subscription-routed MCP tool; tool reaches the Codex backend (mock)
- Happy path: switch `auth_mode` back to API key in the JSON config; invoke the same MCP tool; tool routes through `litellm.completion` (mock)
- Edge case: MCP server running, GUI signs in (writes to keyring), then GUI signs out (clears keyring) — next MCP tool call sees `TokenMissingError` and surfaces it to the MCP client per `core/spine` error contracts (not crash)
- Integration: subscription-mode MCP `start_describe` against a mock Codex backend completes a small job and returns results in the same shape as API-key mode

**Verification:**
- `scene_ripper_mcp/tests/test_auth_mode_propagation.py` passes
- Manual: start MCP server, sign in via the GUI, observe that the next MCP tool call honors subscription mode without restart

---

## System-Wide Impact

- **Interaction graph:** The new auth abstraction is read by `core/llm_client.py:complete_routed`, which is called from `ui/chat_worker.py`, `core/analysis/*` (5 files post-U7), the 3 remix sequencers that use `complete_with_local_fallback` (storyteller, exquisite_corpus, free_association), `core/remix/drawing_vlm.py` (transitively, via `LLMClient.chat`), and indirectly by every MCP LLM-routed tool. The Settings dialog writes auth state; everyone else reads. No other component mutates auth state.
- **Error propagation:** New exception types (`TokenExpiredError`, `TokenMissingError`, `QuotaExceededError`, `CodexBackendError`) flow from `core/codex_client.py` up through `complete_routed`, get caught by workers in `ui/workers/*` and `ui/chat_worker.py`, and surface to the user via structured error signals. The MCP server forwards these to its callers using its existing error-result conventions.
- **State lifecycle risks:** The OAuth worker's `auth_complete` could double-fire under signal-duplication bugs (mitigated by U4's UniqueConnection + guard flag). Token refresh during an in-flight LLM call could race with a parallel call also trying to refresh — mitigated by serializing refresh inside `complete_routed` (one refresh attempt per failed call; if a parallel call already refreshed, the retry uses the fresh token).
- **API surface parity:** The MCP server (U10) and the GUI (everywhere else) must honor the same auth mode. Spine layering ensures both read through `core/spine/chatgpt_auth.py:load_active_auth()`. Tests in `scene_ripper_mcp/tests/test_auth_mode_propagation.py` enforce parity.
- **Integration coverage:** End-to-end happy paths under both modes (chat + each analysis feature) need real exercise with mocked HTTP. The flows F1–F5 from origin are testable end-to-end with `httpx.MockTransport`.
- **Unchanged invariants:** Non-LLM features (transcription, scene detection, vision/ML, OCR-text-tier, embeddings, face detection, audio analysis) are not touched (R16). The existing per-provider API-key fields, env-var overrides, and `_save_llm_api_keys` flow remain functional and behaviorally identical in API-key mode (R8, R17). `tests/test_spine_imports.py`'s forbidden-import list is unchanged.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| OpenAI revokes the Codex CLI `client_id` or changes the auth endpoint shape | Token failures surface as `TokenExpiredError` with a clear re-auth prompt. If the endpoint moves entirely, API-key mode remains fully functional as the alternative (R17). Plan-level acceptance of ToS gray zone is documented in origin. |
| Codex backend response shape changes silently | `core/codex_client.py` normalizes responses to the existing `choices[].message.content` shape. Schema drift surfaces as `CodexBackendError`; tests using `httpx.MockTransport` pin the expected shape. |
| The chokepoint migration (U7) accidentally regresses an existing analysis feature | Each migrated file keeps its existing test suite; per-file integration tests confirm API-key-mode behavior is bit-identical. |
| Quota estimator (U9) is too eager or too lax | Per-operation heuristics live in one module-level dict, tunable in a one-line change. Heuristic is acknowledged as approximate (Deferred to Implementation) and U8 handles the mid-job case as a safety net. |
| OAuth worker signal duplication causes double-write to keyring or two browser tabs | UniqueConnection + guard flag + `@Slot()` decorators, tested explicitly in U4. Direct mirror of the prior solution in `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`. |
| Spine boundary regression — accidentally imports PySide6/LiteLLM at module top level in `core/spine/chatgpt_auth.py` | `tests/test_spine_imports.py` enforces this; U2 explicitly extends the test to cover the new module. |
| The 4 remix sequencers' Ollama fallback masks subscription-mode errors during dev | `complete_with_local_fallback` skips the fallback when `auth_mode = "subscription"` (Key Technical Decisions). Tests in U6 confirm. |
| MCP server staleness — a long-running MCP server misses a GUI sign-in | U10 mandates per-call settings reload at the auth read; cheap operation, no realistic perf impact. |

---

## Documentation / Operational Notes

- `docs/user-guide/` should gain a short page on "Signing in with ChatGPT" — covering when to use it vs. API keys, the quota caveat for batch jobs, and how to switch modes
- The MCP server reference (`docs/user-guide/headless-mcp.md`) should note that subscription auth is honored automatically once configured in the GUI
- No operational rollout / monitoring changes needed; the feature is fully client-side
- After landing, capture the OAuth pattern as a new entry in `docs/solutions/` if any non-obvious bugs surface during implementation (per the project's `docs/solutions/` discipline)

---

## Sources & References

- **Origin document:** [docs/brainstorms/2026-05-12-chatgpt-subscription-auth-requirements.md](docs/brainstorms/2026-05-12-chatgpt-subscription-auth-requirements.md)
- Related code:
  - `core/llm_client.py`, `core/settings.py`, `core/spine/`, `core/analysis/`
  - `ui/settings_dialog.py`, `ui/chat_worker.py`, `ui/workers/base.py`
  - `scene_ripper_mcp/server.py`, `scene_ripper_mcp/tools/jobs.py`
- Project guidance: `CLAUDE.md`, `AGENTS.md`
- Institutional learnings: `docs/solutions/runtime-errors/transcription-mlx-auto-fallback-groq-key-20260512.md`, `docs/solutions/runtime-errors/qthread-destroyed-duplicate-signal-delivery-20260124.md`, `docs/solutions/logic-errors/circular-import-config-consolidation.md`, `docs/solutions/security-issues/url-scheme-validation-bypass.md`, `docs/solutions/reliability-issues/subprocess-cleanup-on-exception.md`, `docs/solutions/ui-bugs/timeline-widget-sequence-mismatch-20260124.md`, `docs/solutions/ui-bugs/pyside6-thumbnail-source-id-mismatch.md`
- External: Codex CLI source (open-source) for the OAuth flow shape; Hermes-agent for desktop-app precedent

---

## Review-Applied Revisions (2026-05-12 doc-review pass)

This section is **authoritative** and supersedes the original plan body where they conflict. It captures every decision applied in the interactive document-review pass. The original body sections above remain readable for context but should be read alongside the revisions here.

### Architectural revisions (P0)

**R-A1. Split LLM-routing chokepoint into sync and async-streaming variants.** U6 introduces two helpers in `core/llm_client.py`:
- `complete_routed(model, messages, **kwargs)` — sync, mirrors `litellm.completion` signature, used by the 5 `core/analysis/*` migration sites and by `complete_with_local_fallback`.
- `stream_chat_routed(model, messages, tools, **kwargs)` — async generator, yields streaming chunks and surfaces `tool_calls`. Used by `LLMClient.stream_chat` / `chat` and consumed by `ui/chat_worker.py`.

`core/codex_client.py` gains a corresponding `stream_complete(...)` async generator that handles Codex backend SSE/chunk decoding and `tool_calls` in streamed deltas. The plan's previous claim that `LLMClient.stream_chat` delegates through `complete_routed` is replaced by this two-helper design.

**R-A2. Per-feature OpenAI-model substitution map for subscription mode.** In `core/codex_client.py`, a module-level dict maps each analysis feature's configured non-OpenAI model to a Codex-supported OpenAI equivalent: e.g. `cinematography_frame` (currently Gemini) → `gpt-4o`; `cinematography_video` (currently Gemini video) → falls back to **frame mode** in subscription (no video-content blocks sent); `description_video` → frame mode; `describe`, `classify`, `custom_query`, `ocr` → an OpenAI model declared per-feature in the substitution map. The substitution applies only when `auth_mode == SUBSCRIPTION`; API-key mode preserves the configured per-feature model. New Key Technical Decision documents which features lose video-input support in subscription mode.

**R-A3. Pre-U7 kwargs compatibility matrix.** Before U7's mechanical swap lands, U6 produces a per-call-site kwargs matrix enumerating every kwarg each of the 5 analysis call sites passes today (`response_format`, video-content blocks, `tools`, `tool_choice`, `max_tokens`, etc.) and whether the Codex backend supports each. Sites with structured-output kwargs (cinematography `response_format: json_object`, shots_cloud JSON parsing) get explicit per-site treatment: translate, degrade with post-parse-and-retry, or remain API-key-only when no graceful path exists. The "purely mechanical" framing in U7 applies only to sites where kwargs cleanly map.

**R-A4. Codex backend tool-calling support is explicitly designed for.** `core/codex_client.py:stream_complete` handles `tool_calls` in streamed deltas and final responses, mirroring OpenAI's function-call format. If the Codex auth endpoint does not support OpenAI-format function calls, chat-agent in subscription mode degrades to a no-tool-use chat with a clear in-chat banner ("Tool use is unavailable on ChatGPT subscription — switch to API key for full agent capabilities"). Verification required in U6 before U7 starts.

### Auth-flow security hardening

**R-S1. State parameter generated inside the flow module.** U3's `build_authorization_url` no longer accepts a caller-supplied `state`; it generates one internally via `secrets.token_urlsafe(32)` and returns `(url, redirect_uri, state)`. A test asserts successive calls produce distinct values of at least 256 bits.

**R-S2. Loopback server binds explicitly to 127.0.0.1.** U3's `run_loopback_listener` calls `HTTPServer(('127.0.0.1', port), ...)` — never `''` or `'0.0.0.0'`. The single-shot handler also validates the request's `Host` header is `localhost` or `127.0.0.1` and the request path matches the expected callback path before parsing the query. A test asserts a second connection from another local socket cannot reach the handler after the first redirect lands.

**R-S3. TLS scheme assertion tests on endpoint constants.** Add to `tests/test_chatgpt_auth.py`: `assert AUTHORIZATION_ENDPOINT.startswith('https://')`, `assert TOKEN_ENDPOINT.startswith('https://')`, `assert CODEX_COMPLETION_ENDPOINT.startswith('https://')`. One line per constant. Catches `http://` transcription mistakes at merge time.

**R-S4. Bearer-token redaction in httpx exception messages.** `core/codex_client.py` defines `_sanitize_headers(headers: dict) -> dict` that replaces any case-insensitive `authorization` value with `[REDACTED]`. `httpx.HTTPStatusError` and `httpx.RequestError` are caught and re-raised as `CodexBackendError` with sanitized message text. A test triggers an httpx error against a bearer-bearing URL and asserts the raised exception message does not contain the token.

**R-S5. Token-refresh mutex + GUI-only-refresh cross-process contract.** `core/spine/chatgpt_auth.py` exposes a module-level `threading.Lock` (sync) and `asyncio.Lock` (async) that serialize refresh inside one process. The holder re-reads the keyring blob after acquiring the lock and skips its own refresh if `expires_at_unix` is newer (another caller already refreshed). Cross-process: only the GUI process is permitted to call refresh; the MCP server treats keyring as read-only and on 401 surfaces a `TokenExpiredError` to its caller per existing MCP error-result conventions. Added to System-Wide Impact's State-lifecycle risks. Tests in `tests/test_codex_client.py` fire two threads against a mock 401-then-200 endpoint and assert exactly one refresh call.

### Settings-dialog UX revisions (U5)

**R-U1. Sign-in button gated on `_oauth_in_progress`.** While the OAuthWorker is running, the Sign-in button is disabled and relabeled ("Waiting for browser…"). Re-enabled on `auth_complete`, `auth_failed` (any category), or worker cancellation.

**R-U2. Auth-mode radios locked during OAuth.** Both radios are disabled while `_oauth_in_progress` is true. Prevents the "mid-flight mode switch overwritten by late `auth_complete`" race. New test scenario in U5.

**R-U3. OK/Save button gated when subscription radio + no token.** When the subscription radio is selected and no token blob exists in keyring, the dialog's OK button is disabled with an inline message: "Sign in first to use ChatGPT subscription mode." Re-enabled on `auth_complete` or when the user reverts to API-key radio. Prevents the silent broken state ("saved subscription but never signed in" → every LLM call raises `TokenMissingError`).

**R-U4. Identity display fallback chain.** When the OAuth response omits `account_email`, the identity label decodes the `sub` claim from the `id_token` and shows that. If neither is available, falls back to `Signed in (ChatGPT Plus)`. Updates U1's token-blob handling to optionally surface `sub` and U5's display logic. Test scenario covers the email-absent response shape.

### Error-UX revisions (U8)

**R-E1. "Sign in again" routes to the settings dialog.** U8's re-auth dialog's "Sign in again" action opens the settings dialog focused on the API Keys tab with the subscription panel scrolled into view. The OAuthWorker is not instantiated from the main window directly. Reuses U5's sign-in flow as the single OAuth entry point.

**R-E2. Mid-stream chat tool-use during token expiry.** When `TokenExpiredError` surfaces mid-turn (after one or more tool calls have already executed), the chat worker attempts one transparent refresh-and-retry of the in-flight LLM segment using R-S5's mutex pattern. If refresh fails, the chat surfaces "Your ChatGPT sign-in expired mid-response — please sign in again and rerun your last message" and the agent loop aborts cleanly with no half-executed tool calls left in inconsistent state. `core/plan_controller.py` multi-step plans use the same pattern. New test scenario in U8 covering refresh-during-tool-use.

**R-E3. Refresh keyring-write failure policy.** When refresh succeeds against the Codex backend but the subsequent `set_chatgpt_oauth_token()` keyring write fails, the in-memory new access_token is used for the current call but a hard error is surfaced before the next call: "Could not persist refreshed sign-in to keyring — sign in again." Prevents the silent invalidate-the-refresh-chain failure mode. U6's Approach updated; test scenario added.

### Quota UX revisions (U9 / U8)

**R-Q1. Pre-flight dialog's "Switch to API key mode" is current-operation-only.** Selecting this option in U9's pre-flight quota dialog routes only the current operation through the API-key path; `Settings.auth_mode` is unchanged. Dialog includes a note: "This operation will run on your API key; your auth mode stays on subscription. Change permanently in Settings." Preserves R4 (single global toggle). Test asserts `Settings.auth_mode` round-trips unchanged after the dialog's Switch action.

**R-Q2. Pre-flight + mid-job dialog single-owner flow.** Once the user proceeds past U9's pre-flight, that dialog is dismissed and does not reappear. The mid-job quota path is owned entirely by U8's `QuotaExceededError` toast. Test asserts no second pre-flight dialog appears after a Proceed-then-mid-job-error sequence.

**R-Q3. U9 default to wide tolerance until calibration data exists.** Per-operation heuristic constants in `core/quota_estimator.py` default to permissive values for v1 (warn only on clearly-large batches: >300 clips for describe, >500 for classify). A calibration step is added to Documentation/Operational Notes: collect one week of subscription-mode usage data before tightening thresholds. Wide defaults prevent training users to dismiss warnings before the estimator is accurate.

### Cross-process & MCP revisions

**R-M1. Auth-state snapshot pinned to MCP jobs at start.** U10 revises the per-call reload behavior: the MCP server reloads settings + token blob at the **start of each job** (e.g., at the top of `start_describe`'s handler), then **pins that snapshot** to the job for its lifetime. Per-clip iterations inside the job use the snapshot. If the snapshot becomes invalid mid-job (refresh fails, mode changed externally), the job fails loudly with a clear status ("auth changed mid-job — restart in current mode") rather than silently switching backends per clip. Per-call reload still happens between jobs.

**R-M2. MCP trust-boundary statement.** Added to U10 and to `docs/user-guide/headless-mcp.md`: "The MCP server is trusted to the same degree as the GUI because both run as the same OS user and share the keyring. This is acceptable for a local, single-user desktop application where the MCP server is launched by the GUI or by the user themselves. If the MCP server is ever exposed over a network socket rather than stdio, this decision must be revisited and an MCP-level auth layer added before that point."

**R-M3. U10 test placement note.** U10's Files section clarifies: "Test placement at `scene_ripper_mcp/tests/test_auth_mode_propagation.py` is intentional, matching the existing `scene_ripper_mcp/tests/` convention for MCP-server-specific tests (e.g., `test_integration.py`, `test_jobs_runtime.py`). It is not inconsistent with U1–U9's `tests/` placement — those are app-wide tests; this is MCP-scoped."

### Scope & coverage revisions

**R-Sc1. Composer review R7 correction.** Added to Scope Boundaries (parallel to the gaze correction): "**LLM Composer** (`word_llm_composer` / `core/spine/words.py:compose_with_llm`) uses `complete_with_enum_constraint`, which is Ollama's JSON-schema enum constraint API — not exposed by the Codex/OpenAI backend. In v1, the LLM Composer remains Ollama-only regardless of auth mode. R7 listed it; this is a requirements correction surfaced during review." Plus added to Deferred to Follow-Up Work: "LLM Composer cloud path — add cloud-routed variant (LLM returns JSON, vocabulary validated client-side with retry on OOV) when the composer is extended."

**R-Sc2. `drawing_vlm.py` transitive-coverage note.** Added to U7's Verification: "`core/remix/drawing_vlm.py` uses `LLMClient.chat` and is covered transitively by U6's split (it consumes `stream_chat_routed`). No direct migration needed in U7. Grep over `core/remix/` confirms no surviving `litellm.completion` calls after U6."

### Implementation-discipline revisions

**R-T1. Live sign-in smoke test acceptance step.** Added to U3 Verification: "A live sign-in smoke test (manual or env-var gated) exercises the real OpenAI Codex auth endpoint end-to-end once before U3/U4 can be claimed complete. Unit tests with `httpx.MockTransport` prove the implementation does what the code claims; the smoke test proves the code's claims match the real upstream contract." Explicitly notes that test-first does not substitute for contract verification.

**R-T2. Ollama-fallback policy is availability-gated, not blanket-skipped.** Key Technical Decisions revision: in subscription mode, `complete_with_local_fallback` detects whether Ollama is reachable at call time before deciding whether to fall back. If reachable and configured for the relevant model → fall back with a clear message ("Subscription quota hit — falling back to local Ollama for this call"). If not reachable → surface the auth/quota error. Replaces the previous blanket "skip Ollama in subscription" policy.

**R-T3. Linux plaintext keyring backend detection.** U1's `set_chatgpt_oauth_token()` detects the active keyring backend after a successful write. If the backend is `PlaintextKeyring` or any class from `keyrings.alt`, the helper refuses the write, calls `clear_chatgpt_oauth_token()` to remove the entry, returns False, and surfaces a user-facing error: "Secure storage is not available on this system — ChatGPT sign-in requires a system keychain (macOS Keychain, GNOME Keyring, KWallet, or Windows Credential Manager). API key mode remains available." Fail-closed for OAuth tokens; API-key path unchanged.

### U8 / U9 worker-file dependency note

**R-D1. U9's edits to worker pre-run hooks land on top of U8's exception-catch wiring.** U9's Approach gains a note: "Both U8 and U9 modify the same pre-`run()` block in `ui/workers/description_worker.py` (and the other analysis workers). U9 depends on U8 (declared) — U8's exception-catch wiring must land first; U9's quota-estimator calls slot in above it. Implementer should land U8's edit, run the worker, then land U9's edit on top."

### Resolved-During-Planning additions

The following questions are now resolved by the revisions above and should be considered settled (no longer open):

- *How does the chat agent get streaming + tool-use in subscription mode?* → R-A1 (split chokepoint with streaming variant + tool_calls support).
- *Which OpenAI model substitutes for each non-OpenAI configured model in subscription?* → R-A2 (per-feature substitution map in `core/codex_client.py`).
- *Is the U7 migration mechanical?* → R-A3 (only for kwarg-compatible sites; structured-output sites get per-site treatment).
- *How is refresh serialized across threads and processes?* → R-S5 (in-process locks + GUI-only-refresh contract).
- *How is the OAuth state parameter generated?* → R-S1 (`secrets.token_urlsafe(32)` inside the flow module).
- *How is the loopback callback hardened against local-process injection?* → R-S2 (explicit 127.0.0.1 + Host/path validation).
- *What does "Switch to API key mode" in the quota dialog do?* → R-Q1 (current-operation-only override; settings unchanged).
- *Where does "Sign in again" route?* → R-E1 (settings dialog focused on subscription panel).
- *How is mid-stream token expiry handled?* → R-E2 (transparent refresh-retry + clean fallback).
- *How does the MCP server handle long-running jobs across auth changes?* → R-M1 (snapshot at job start; fail loudly on mid-job invalidation).
- *What happens on Linux without a Secret Service backend?* → R-T3 (refuse the write; user-facing error; API-key mode remains).
- *Does subscription mode skip Ollama fallback unconditionally?* → R-T2 (availability-gated; falls back only if Ollama is reachable).
- *What is the identity-display fallback when email is absent?* → R-U4 (id_token sub claim → "Signed in (ChatGPT Plus)" fallback).

### Coverage from this review pass

- **Total findings:** 33 (25 actionable @ confidence 75+, 8 FYI @ confidence 50)
- **Applied:** 5 safe_auto (silent during headless pass) + 25 manual/gated (this section)
- **Deferred:** 0
- **Skipped:** 0
- **FYI observations recorded (no plan body change):**
  - Sign-out confirmation step not specified (P2): consider a confirm dialog before Sign Out, mirroring existing destructive-action patterns in the settings dialog
  - Accessibility properties (`setAccessibleName`, focus order) not addressed for new controls in U5
  - `core/quota_estimator.py` as a separate module may be over-abstracted given a single consumer — consider inlining if a second consumer doesn't appear
  - `AuthMode` enum should NOT acquire abstract base classes / plugin registries / extension hooks beyond the two-value enum + branch-in-load_active_auth pattern; cap v1 there
  - Malformed-JSON keyring entry should self-clear via `clear_chatgpt_oauth_token()` to break the perpetual `TokenMissingError` loop (defense-in-depth on top of R-T3)
  - Auth-URL Qt slot in U5 should re-validate `url.startswith('https://')` before calling `QDesktopServices.openUrl` (defense-in-depth on top of R-S2)
  - Codex backend response sanitization threat model — currently trusted equivalently to the OpenAI API under API-key mode; document if this becomes a concern
  - Loopback port-collision and firewall-prompt UX — surface as a distinct `bind_failed` error category in U3 if the bind contention proves frequent in practice
