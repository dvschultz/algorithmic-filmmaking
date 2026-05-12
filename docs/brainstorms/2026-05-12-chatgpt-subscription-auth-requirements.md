---
date: 2026-05-12
topic: chatgpt-subscription-auth
---

# ChatGPT Subscription Auth

## Summary

A "Sign in with ChatGPT" auth option that lets ChatGPT Plus/Pro subscribers use Scene Ripper's LLM features against their existing subscription quota instead of paying for OpenAI API tokens. The existing API-key path remains unchanged as the alternative; users pick one auth mode globally in settings.

---

## Problem Frame

Scene Ripper has a meaningful LLM-backed surface — chat agent, clip description, shot classification, cinematography analysis, custom queries, composer review, gaze analysis. Today, all of those features require the user to supply an OpenAI API key (or another provider's API key) and to top up a separate billing account.

A real segment of Scene Ripper's power users already pay $20–$200/month for ChatGPT Plus or Pro. For those users, configuring and paying for the OpenAI API *in addition* to the subscription they already have feels insulting enough that they simply leave the AI features off. The result: a chunk of the app's differentiating capability is shelf-ware for the segment most likely to value it.

Recent ecosystem precedent (Hermes, OpenCode, Roo Code) demonstrates a viable OAuth pattern against OpenAI's Codex auth endpoint that uses a user's ChatGPT subscription quota for inference. The pattern is gray-zone with respect to OpenAI's Terms of Service — not officially endorsed, not officially prohibited — but it is tolerated in practice and used by multiple production tools today. An OpenAI engineer has loosely characterized the category as "permissive" while disclaiming legal authority.

---

## Actors

- A1. ChatGPT Plus/Pro subscriber (target user): an existing Scene Ripper user who pays for ChatGPT and has been skipping the AI features rather than paying for the OpenAI API on top.
- A2. Existing API-key user: a Scene Ripper user who has already configured an OpenAI, Anthropic, Gemini, or other API key and uses the AI features today via that path.

---

## Key Flows

- F1. First-time sign-in with ChatGPT
  - **Trigger:** User opens settings and selects "Use ChatGPT subscription" / "Sign in with ChatGPT."
  - **Actors:** A1
  - **Steps:** App opens the user's default browser to OpenAI's sign-in page. User completes sign-in to chatgpt.com. App receives the token from the OAuth callback. App stores the token in secure OS storage. Settings updates to show the signed-in identity and confirms ChatGPT subscription is now the active auth mode.
  - **Outcome:** All LLM-backed features in the app route inference via the user's ChatGPT subscription quota.
  - **Covered by:** R1, R2, R6, R9

- F2. Returning user with a valid token
  - **Trigger:** User launches the app; subscription mode is the previously selected auth mode; stored token is still valid.
  - **Actors:** A1
  - **Steps:** App reads the stored token from secure storage on startup and resumes subscription mode without prompting.
  - **Outcome:** AI features work immediately; user sees no auth interruption.
  - **Covered by:** R9, R10

- F3. Expired or revoked token
  - **Trigger:** A previously stored token has expired or been revoked; an LLM-backed feature is invoked.
  - **Actors:** A1
  - **Steps:** App attempts a refresh; on failure, surfaces a clear in-app prompt: "Your ChatGPT sign-in has expired — please sign in again." User can re-run sign-in or switch to API-key mode.
  - **Outcome:** No silent failures; the user understands what happened and what to do.
  - **Covered by:** R10, R11

- F4. Switching auth modes
  - **Trigger:** User changes the auth-mode toggle in settings.
  - **Actors:** A1, A2
  - **Steps:** User selects the other mode in settings. App applies the change to all subsequent LLM-backed feature calls without requiring a restart.
  - **Outcome:** The next LLM call uses the newly selected mode.
  - **Covered by:** R4, R5

- F5. Long batch operation when quota is near the limit
  - **Trigger:** User in subscription mode initiates a batch LLM operation (e.g., describe across many clips).
  - **Actors:** A1
  - **Steps:** App estimates likely quota consumption against the user's remaining subscription quota. If the operation is likely to exceed remaining quota, the app warns *before* starting and gives the user a clear choice (proceed anyway, cancel, or switch to API-key mode). If the user proceeds and hits the quota wall mid-operation, the app surfaces a quota-specific error rather than a generic rate-limit message.
  - **Outcome:** Users do not get blindsided by quota walls mid-batch; the failure mode is legible and recoverable.
  - **Covered by:** R12, R13

---

## Requirements

**Authentication and sign-in**
- R1. The app provides a "Sign in with ChatGPT" action in settings that initiates a device-code OAuth flow against OpenAI's Codex auth endpoint, without requiring Codex CLI or any other external OpenAI tool to be installed.
- R2. The OAuth flow opens the user's default browser to OpenAI's sign-in page; on successful authorization, the app receives and stores the resulting token without further user action.
- R3. The app handles auth-failure cases — no internet, user cancels in the browser, OpenAI rejects the request — with clear, specific in-app error messages and does not change the current auth mode on failure.

**Auth-mode selection and settings**
- R4. Settings exposes a single global auth-mode toggle: "Use ChatGPT subscription" or "Use API key." The choice applies to every LLM-backed feature in the app; routing is not configurable per-operation.
- R5. Switching auth modes takes effect immediately for subsequent LLM calls; no app restart is required.
- R6. While ChatGPT subscription mode is active, settings displays the currently signed-in identity (account email or display name from the OAuth response).
- R14. Settings provides a "Sign out" action that clears stored tokens and reverts the app to a state where the user must pick an auth mode again.

**LLM feature routing**
- R7. When ChatGPT subscription mode is active, every LLM-backed feature in the app — chat agent, describe, classify-shots, cinematography, custom-query, composer review, gaze — routes inference requests through the ChatGPT subscription token against the Codex backend.
- R8. When API-key mode is active, every LLM-backed feature behaves exactly as it does today: no observable difference from the current behavior, and no new dependencies on subscription-auth code paths.

**Token storage and lifecycle**
- R9. Tokens are stored in secure OS storage (macOS Keychain on macOS, platform equivalent elsewhere) — never in plaintext on disk.
- R10. The app refreshes tokens automatically when they expire and the refresh succeeds, without user interaction.
- R11. When a token is unrecoverable — refresh fails, account revoked, OpenAI rejects the credential — the app surfaces a clear in-app prompt asking the user to sign in again, rather than failing LLM calls silently or with raw error text.

**Quota awareness**
- R12. Before initiating a batch operation likely to consume substantial subscription quota (describe / classify / custom-query across many clips), the app estimates expected consumption and warns the user when remaining quota is likely insufficient — before starting the operation.
- R13. When a quota-related error is returned mid-operation, the app surfaces a quota-specific error message (clearly framed as "you've hit your ChatGPT Plus quota") with explicit next-step options (wait and retry, or switch to API-key mode), rather than a generic rate-limit error.

**Compatibility and preservation**
- R15. Account switching is supported via Sign out → Sign in; no explicit multi-account UI is provided in v1.
- R16. No change to non-LLM features (transcription, scene detection, vision/ML, embeddings, OCR, face detection, audio analysis) — they continue using their existing local or external paths regardless of auth mode.
- R17. Existing user settings for OpenAI, Anthropic, Gemini, and other API keys are preserved and remain fully functional when API-key mode is active.

---

## Acceptance Examples

- AE1. **Covers R3.** Given the user clicks "Sign in with ChatGPT" with no internet connection, when the OAuth flow fails to reach OpenAI, the app shows a specific "Could not reach OpenAI to sign in — check your connection" message and leaves the current auth mode unchanged.
- AE2. **Covers R8.** Given the user has API-key mode active and has never signed in with ChatGPT, when they invoke any LLM-backed feature, the request routes through the configured API key exactly as today, with no new code path introduced by this feature.
- AE3. **Covers R11.** Given a previously valid subscription token has been revoked by OpenAI, when the user invokes an LLM-backed feature, the app surfaces a "Your ChatGPT sign-in has expired — please sign in again" prompt and offers the option to re-authenticate or switch to API-key mode, rather than failing the LLM call with a raw error.
- AE4. **Covers R12, R13.** Given the user is in subscription mode and triggers `describe` across 200 clips, when the estimate suggests the operation is likely to exceed remaining quota, the app warns before starting; if the user proceeds anyway and hits the quota wall mid-operation, the app surfaces a quota-specific error with clear next-step options.
- AE5. **Covers R5.** Given the user is signed in via subscription mode and is mid-conversation in the chat panel, when they switch to API-key mode in settings, the next chat message routes through the API key without requiring an app restart.

---

## Success Criteria

- Scene Ripper users who already subscribe to ChatGPT Plus or Pro and previously did not use the AI features begin using them. The "paying twice" objection no longer applies for that segment.
- A user can sign in once, complete a normal Scene Ripper session that uses the AI features, and never see a raw OAuth error, an unexplained rate-limit, or an unauthenticated state they don't know how to recover from.
- Existing API-key users see no behavioral change in their AI-feature workflows.
- A downstream planner reading this document can specify the OAuth flow mechanics, settings UI changes, token storage layer, LLM-routing changes, and quota-aware UX without needing to invent any product behavior.

---

## Scope Boundaries

- Per-operation auth routing (selecting subscription auth for one job and API key for another within the same session) — explicitly rejected in favor of the global mode toggle.
- Importing existing Codex CLI credentials from `~/.codex/auth.json` (Approach 3 in the discussion) — deferred. The target user almost certainly does not have Codex CLI installed; this is a v1.1 polish at most.
- Claude Max (Anthropic OAuth), Gemini OAuth, GitHub Copilot OAuth, or any other provider's subscription-auth path — separate future brainstorm. The v1 architecture should not foreclose adding these, but v1 itself ships ChatGPT only.
- Multi-account UI / explicit account switching beyond Sign out → Sign in.
- Scene Ripper running a hosted inference proxy (subsidizing inference on behalf of users) — a different product entirely.
- Any change to non-LLM features (transcription, scene detection, vision/ML, embeddings, OCR, etc.).
- Any change to the existing OpenAI / Anthropic / Gemini API-key paths — they remain fully functional as the alternative.

---

## Key Decisions

- **Approach 1 over Approaches 2 and 3:** Scene Ripper runs the OAuth flow itself rather than depending on Codex CLI being installed or imported. Rationale: the target user is a Plus subscriber who has not installed Codex CLI and likely never will; telling them to install another tool to use this one violates the entire pitch ("remove the paying-twice friction" by adding setup-another-OpenAI-tool friction).
- **Global auth-mode toggle over per-operation routing:** simpler mental model, simpler UI, simpler implementation. Users who want bulk-on-API-key can switch modes; that case is rare enough to not warrant per-job UI surface.
- **Keep architecture open for Approach 4 without building it:** separate the concept of "auth provider" from "LLM client" so that Claude Max, Gemini OAuth, etc. can be added later without rework.
- **API-key path preserved unchanged:** users who prefer it, or who hit Plus quota walls, keep their working setup with zero migration.
- **Subscription mode applies to *all* LLM-backed features, not only OpenAI-flavored model calls:** when subscription mode is active, the app uses the ChatGPT-subscription-exposed models for everything, including features that today might call Anthropic or Gemini under a user's API-key configuration. Rationale: a user who flipped the toggle is saying "use my Plus," not "use my Plus for OpenAI calls and my Anthropic key for Claude calls."

---

## Dependencies / Assumptions

- OpenAI's Codex auth endpoint and the device-code OAuth flow it supports remain accessible to non-Codex applications. This is currently the case for Hermes, OpenCode, Roo Code, and others; OpenAI could revoke this at any point.
- ChatGPT Plus/Pro subscriptions continue to expose inference quota to clients authenticated via this flow.
- Terms of Service posture: this feature aligns with current production-tool practice in the category. OpenAI has neither endorsed nor prohibited the pattern. If OpenAI revokes the OAuth client_id, shuts down third-party use, or otherwise blocks this path, the feature breaks for all users until alternative auth is available. Users in API-key mode are unaffected.
- The Plus quota model in 2026 still rate-limits chat-style usage in 3-hour windows. Bulk inference operations through Scene Ripper can plausibly exhaust quota faster than chat usage would; quota-aware UX is required, not optional.

---

## Outstanding Questions

### Deferred to Planning

- [Affects R1][Technical] Which OAuth `client_id` does Scene Ripper use? Borrowing the public Codex CLI client_id (as Hermes appears to) versus attempting to register a separate one — tradeoffs in fragility, ToS posture, and account-revocation blast radius need to be evaluated during planning.
- [Affects R7][Technical] What is the exact request and response shape expected by the Codex backend, and how does it map onto the app's current LLM-client abstraction (LiteLLM-based)? Whether this requires a separate non-LiteLLM client path, a custom LiteLLM provider, or a wrapper around the existing client is a planning decision.
- [Affects R12][Needs research] Does the Codex backend expose remaining-quota information in any header or endpoint that the app can use for pre-flight estimates? If not, the warning in R12 needs a heuristic (estimated tokens per clip × clip count vs. a user-supplied or default budget) rather than a real check.
- [Affects R9][Technical] Which secure storage library / mechanism is used on each supported platform? `keyring` on Python is a likely candidate but the exact selection is planning territory.
- [Affects R12, R13] What is the user-facing wording for the quota-warning prompt and quota-exhausted error? Wording can be settled at planning or in implementation review.
