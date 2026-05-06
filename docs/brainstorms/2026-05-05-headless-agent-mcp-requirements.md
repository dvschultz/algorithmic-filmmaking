---
date: 2026-05-05
topic: headless-agent-mcp
---

# Headless Agent-Callable Scene Ripper

## Summary

Expand `scene-ripper-mcp` into a headless surface that Claude Code (and other MCP-capable agent harnesses) can drive for long-running multi-film work. Refactor the GUI's chat agent and the MCP server onto a shared, GUI-agnostic tool spine, then add async/job semantics for ops that take minutes to hours. The desktop GUI continues to work unchanged for students.

---

## Problem Frame

Scene Ripper today has two surfaces: a desktop GUI (the primary path, used by students and newcomers) and an MCP server (`scene-ripper-mcp`) exposing a small synchronous tool catalog for agent integrations. The GUI also embeds an LLM-driven chat agent (`ui/chat_worker.py` + `core/chat_tools.py`) that can plan and execute multi-step work inside an open project.

For the project owner, the GUI is now friction. Most editorial work happens from Claude Code, where multi-step iteration ("start a process, look at the result, run the next step") is the natural cadence. The current options force a bad choice:

- Open the GUI and use the embedded chat agent — but the GUI's pacing is wrong for someone who already knows the tools.
- Use the MCP server from Claude Code — but the synchronous-only tool catalog can't host multi-film batch jobs that run for minutes-to-hours, and it covers a narrower surface than the GUI's chat agent has.
- Use Claude Code with file-system tools and shell out — but that loses the project model, the analysis pipeline, and everything `core/chat_tools.py` already knows how to do.

Compounding this: every new tool added to the GUI's chat agent has to be re-implemented in `scene_ripper_mcp/tools/` to be reachable from Claude Code, and vice versa. The two catalogs drift.

The pain shows up acutely on multi-film projects — work that involves importing several films, running detection and analysis across all of them, generating sequences, and iterating. Today that's a long GUI session; the want is a Claude Code session that delegates the slow parts and stays unblocked.

---

## Actors

- A1. **Power user (project owner)**: Operates almost exclusively from Claude Code; wants to drive Scene Ripper headlessly for long-running multi-film work. Primary actor for this feature.
- A2. **Student / GUI user**: Continues to use the desktop GUI as today. Does not interact with the MCP surface. Their experience must not regress.
- A3. **Calling agent (Claude Code, Hermes, Openclaw, similar)**: An MCP-capable harness that issues tool calls, polls jobs, and composes Scene Ripper with other tools in its environment. Plans and decides; does not delegate planning back to a Scene Ripper-side LLM.
- A4. **Scene Ripper MCP server**: Long-lived process that hosts the tool spine, owns the job registry, and runs work in the background. May be started by the calling agent or run as a service.

---

## Key Flows

- F1. **Claude Code drives a multi-film analysis end-to-end (headless)**
  - **Trigger:** Power user, in Claude Code, asks to process a folder of films through scene detection and analysis.
  - **Actors:** A1, A3, A4
  - **Steps:**
    1. Calling agent creates or opens a project via MCP, passing a project handle.
    2. Calling agent imports sources (local files or YouTube URLs).
    3. Calling agent invokes a long-running op (e.g., bulk scene detection across all sources). Op returns a job handle immediately.
    4. Calling agent polls job status; gets progress updates and final completion signal.
    5. Calling agent retrieves job result, decides next step, invokes the next long-running op (analysis, sequencing).
    6. Calling agent saves the project. The desktop GUI can later open the saved file to review or render.
  - **Outcome:** A real film project has been driven end-to-end without opening the GUI; the saved `.sceneripper` file is identical in shape to one a GUI user would produce.
  - **Covered by:** R1, R2, R3, R4, R5, R6, R8, R9, R10

- F2. **Power user resumes a long-running job after restart**
  - **Trigger:** A long job (multi-film analysis) was kicked off; the MCP server (or the calling agent) is restarted before the job finishes.
  - **Actors:** A1, A3, A4
  - **Steps:**
    1. MCP server restarts; rehydrates job registry from on-disk state.
    2. In-flight jobs that survived the restart resume or report a clear status (`running` / `crashed` / `completed`).
    3. Calling agent polls the same job handle from before the restart and gets a coherent answer.
  - **Outcome:** The user does not lose work to a restart, or — if a job genuinely cannot be resumed — gets a clear failure with enough information to retry.
  - **Covered by:** R7, R8

- F3. **GUI chat agent uses the same tool a headless caller does**
  - **Trigger:** A new operation is added to the shared tool spine (e.g., a new analysis op).
  - **Actors:** A1, A2, A4
  - **Steps:**
    1. Operation is implemented once against the shared tool spine.
    2. The GUI's chat agent picks it up via the spine; no separate registration needed.
    3. The MCP server picks it up via the spine; no separate registration needed.
  - **Outcome:** The two surfaces stay in lockstep; new SR capabilities are reachable from both without duplicate work.
  - **Covered by:** R11, R12, R13

---

## Requirements

**MCP surface and async jobs**
- R1. The MCP server exposes start/status/result/cancel tools for long-running operations. Starting an op returns a job handle immediately; status and result are retrieved by polling.
- R2. The MCP server can run multiple jobs concurrently. Status and results are addressable independently per job handle.
- R3. Job state is persisted to disk so jobs survive MCP server restarts. After restart, the server reports a coherent status for any job started in a prior session.
- R4. Jobs are cancellable from the calling agent. A cancelled job stops cleanly without corrupting the project file.
- R5. The MCP server backward-compatibly retains its existing synchronous tools for short ops; the async pattern is reserved for ops where blocking would degrade the calling agent's session.
- R6. Long-running ops report progress in a form the calling agent can poll (e.g., percent complete and a current-step description). Progress is visible while the job is running, not only at completion.

**Headless project workflow**
- R7. Project handles in MCP calls are lightweight references rather than full project state passed inline per call. A calling agent can keep a session driving the same project across many tool calls without round-tripping the project's contents.
- R8. The MCP surface covers the operations needed for the multi-film analysis flow (F1) end-to-end: create or open project, import sources, run detection, run analysis, build sequences, save. No required step in F1 forces a fallback to the GUI.
- R9. Saved project files produced via the MCP surface are byte-equivalent in structure to GUI-produced files and round-trip cleanly through the GUI.
- R10. The headless surface does not require any GUI process to be running. The MCP server can be started, used end-to-end, and shut down without the desktop app being open.

**Shared tool spine**
- R11. A shared GUI-agnostic tool spine exists. Each operation in the spine takes a `Project` (and op-specific arguments) — not a `main_window` or other Qt-bound context. The spine has no dependency on PySide6.
- R12. The GUI's chat agent (`core/chat_tools.py`) is refactored to consume the spine for operations that don't require GUI-specific affordances (dialogs, tab navigation, plan controller's interactive surface). GUI-specific tools remain in `core/chat_tools.py` as wrappers.
- R13. The MCP server consumes the spine for the same operations the chat agent does. New operations added to the spine are reachable from both surfaces without separate registration.
- R14. Refactor proceeds tool-by-tool with regression coverage for the GUI's chat agent at each step. No big-bang rewrite.

---

## Acceptance Examples

- AE1. **Covers R1, R6.** Given a project with 12 imported sources, when the calling agent invokes the bulk scene-detection start tool, then the call returns a job handle within seconds (not minutes), and subsequent status polls report progress proportional to the number of sources processed.
- AE2. **Covers R3, R7.** Given a job started in MCP server session A, when the MCP server is killed and restarted as session B, then the calling agent can poll the same job handle and receive a coherent status (`running`, `crashed`, or `completed`) — never a "job unknown" error.
- AE3. **Covers R4.** Given a long-running analysis job is in flight, when the calling agent invokes cancel on its job handle, then the underlying work stops, no partial corruption is committed to the project file, and subsequent status polls report `cancelled`.
- AE4. **Covers R9, R10.** Given the MCP server is the only Scene Ripper process running, when a calling agent drives F1 end-to-end and saves the project, then a student opening that `.sceneripper` file in the GUI sees the same project shape (sources, clips, analysis fields, sequence) they would expect from a GUI-produced file.
- AE5. **Covers R11, R12, R13.** Given a new operation is added to the shared tool spine, when no other change is made, then the operation is invokable from both the GUI's chat agent and the MCP server with consistent arguments and behavior.

---

## Success Criteria

- One real multi-film project is driven from initial import through analysis and sequence generation entirely from Claude Code, without the GUI being opened. The resulting `.sceneripper` file opens cleanly in the GUI for review or render.
- A long-running job (≥ 30 minutes) survives at least one MCP server restart without losing progress or requiring the user to re-issue the start call.
- After the spine refactor, the GUI's chat agent passes its existing test surface and exhibits no behavioral regressions on the GUI side. Students using the GUI cannot tell the refactor happened.
- A new SR operation added during the implementation period requires only one tool definition to be reachable from both the GUI agent and the MCP server.
- Downstream planning (`/ce-plan` or equivalent) does not need to invent new product behavior, scope boundaries, or success criteria from this doc.

---

## Scope Boundaries

- No second LLM agent loop runs inside Scene Ripper. The calling agent (Claude Code, Hermes, Openclaw) plans; Scene Ripper executes. Delegated agent-as-tool is rejected for v1.
- No drive-the-running-GUI bidirectional mode. MCP and GUI do not modify the same live project simultaneously; coordinating a running GUI with concurrent MCP edits is out.
- No campaign / declarative-spec orchestration layer in v1. Multi-film batch work is composed by the calling agent from granular ops. A higher-level "campaign" primitive is a candidate follow-up after granular use reveals the right shape.
- No standalone `scene-ripper chat ...` CLI entry point in v1. Delivery is via MCP, since the named harnesses (Claude Code, Hermes, Openclaw) are MCP-capable.
- No streaming MCP responses. Polling is the chosen async shape (chosen for harness compatibility and connection-loss resilience).
- No new GUI-side features for students arising from this work. Their experience is unchanged.
- No refactor of `core/chat_tools.py` beyond what is needed to share the spine. Tightly GUI-coupled flows (dialogs, plan controller's interactive surface, GUI state mirroring) stay where they are.
- Authentication, rate limiting, or remote-network exposure of the MCP server are not in scope. The server runs locally, as today.

---

## Key Decisions

- **Async pattern is polling, not streaming or sync-with-long-timeout**: works across MCP harnesses uniformly, survives transient connection loss, supports multiple concurrent jobs per session, and matches the calling agent's natural "kick it off, check back later" cadence.
- **Approach B (shared tool spine) chosen over Approach A (parallel catalogs) and C (campaign abstraction)**: the catalog-drift cost of A compounds with every new feature; C is premature without granular usage informing the right primitive. The spine refactor pays back across both surfaces and is the only path that prevents continued drift.
- **No SR-side LLM agent**: the calling agent already plans well. A second LLM in the chain adds latency, cost, and failure modes without giving the user something they couldn't do from Claude Code directly.
- **GUI coexistence deferred**: the power user's actual workflow has the GUI closed. Live MCP-modifies-running-GUI sync is real engineering (locking, IPC, GUI refresh) and not on the path to value for v1.
- **Refactor is tool-by-tool with regression coverage, not big-bang**: `core/chat_tools.py` is large and GUI-coupled in many places. Incremental extraction with the GUI agent's tests as the gate keeps blast radius bounded.

---

## Dependencies / Assumptions

- The `Project` model (`core/project.py`) is already the source of truth for project state and is not Qt-bound; the spine can take it directly. (Consistent with project conventions.)
- `core/feature_registry.py`'s on-demand install path works headlessly when invoked from a non-GUI process. If features that require user-facing install prompts are exercised from MCP, the MCP server must surface a clear "feature unavailable" error to the calling agent rather than blocking on a UI dialog. **Verification deferred to planning.**
- `plan_controller`'s interactive surface (the GUI's plan presentation flow) is GUI-bound by design; only its non-interactive pieces, if any, can move into the spine.
- `scene_ripper_mcp/security.py`'s path-validation guarantees continue to apply to the expanded surface.

---

## Outstanding Questions

### Resolve Before Planning

- *(none — all scope-shaping questions resolved during brainstorm)*

### Deferred to Planning

- [Affects R6][Technical] What progress granularity is feasible for each long-running op (per-source, per-clip, per-frame)? The answer affects how useful "running" status is to the calling agent.
- [Affects R3][Technical] What is the right on-disk shape for persisted job state? (Single JSON registry vs per-job files vs SQLite.) Affects restart-rehydration behavior.
- [Affects R7][Technical] What is the right project handle shape — file path, an MCP-server-issued session ID, or both? Affects how the calling agent addresses a project across many tool calls.
- [Affects R11, R12][Needs research] Which `core/chat_tools.py` tools genuinely require `main_window` (GUI-coupled) vs which only use `main_window.project` (trivially extractable)? An audit at planning time will scope the refactor.
- [Affects R4][Technical] Which long-running ops are cleanly cancellable vs cancellable-only-at-boundaries (e.g., between sources)? Answer informs cancel semantics in the contract.
- [Affects R10][Technical] How does on-demand dependency install (`core/feature_registry.py`) behave when invoked from a non-GUI process? Need to verify or define the headless install/skip behavior.
