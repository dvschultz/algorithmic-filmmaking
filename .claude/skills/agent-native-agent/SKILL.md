# Agent-Native Architecture Reviewer

Review product plans and features to ensure they follow agent-native principles. Use this skill when planning new features, reviewing architecture decisions, or auditing existing functionality for agent accessibility.

## Core Principles to Enforce

### 1. Parity
**Agents must achieve everything users can do via UI.**

For every feature, ask:
- Can an agent trigger this action programmatically?
- Is there an API/CLI/tool equivalent to the UI action?
- Are there "orphan UI actions" that only humans can perform?

**Red flags:**
- Buttons that only work via mouse click
- Drag-and-drop without file path alternative
- Modal dialogs that require human confirmation with no bypass
- Features locked behind visual-only workflows

### 2. Granularity
**Tools are atomic primitives; features emerge from composition.**

For every tool/API:
- Does it do ONE conceptual action?
- Can it be composed with other tools?
- Is it a primitive or a choreographed sequence?

**Red flags:**
- Wizard-style multi-step flows with no individual step access
- Monolithic "do everything" functions
- Features that bundle unrelated actions

### 3. Composability
**New features deploy via prompts alone.**

For every new feature request, ask:
- Can this be achieved by combining existing tools?
- Does it require code changes, or just a new prompt?
- Are we building a shortcut or a gate?

**Red flags:**
- Every feature requires new code
- Existing primitives can't be combined
- Features are siloed and non-interoperable

### 4. Emergent Capability
**Agents compose tools for unanticipated requests.**

For product evolution:
- What are users asking agents to do that we didn't anticipate?
- Which tool combinations are most common?
- What latent demand patterns are emerging?

**Red flags:**
- No telemetry on agent tool usage
- No way to observe what users request
- Ignoring novel use patterns

### 5. Improvement Over Time
**Apps evolve through context and prompts, not just code.**

For maintenance:
- Can behavior improve without shipping updates?
- Is context accumulated and preserved?
- Can prompts be refined based on outcomes?

**Red flags:**
- Every improvement requires a deploy
- No persistent context between sessions
- Static, unchangeable behavior

---

## Technical Checklist

### Data Organization
- [ ] **Files as universal interface** - Data stored in files agents can read/write
- [ ] **User-inspectable** - Humans can view/edit the same data agents use
- [ ] **Entity-scoped directories** - `{type}/{id}/` structure (e.g., `projects/abc123/`)
- [ ] **Markdown for content** - Human-readable, agent-parseable
- [ ] **JSON for structured data** - Machine-readable metadata
- [ ] **Context.md pattern** - Working memory that persists across sessions

### API/Tool Design
- [ ] **One action per tool** - Atomic primitives
- [ ] **Explicit completion signals** - Clear success/failure/partial states
- [ ] **Checkpoint/resume support** - Long tasks can be interrupted and continued
- [ ] **Bounded context design** - Agents can summarize mid-session

### User Agency Framework
| Stakes | Reversibility | Agent Behavior |
|--------|---------------|----------------|
| Low | Easy | Auto-apply |
| Low | Hard | Confirm first |
| High | Easy | Confirm first |
| High | Hard | Explicit approval required |

### Failure Modes to Avoid
- [ ] **No orphan UI actions** - Every button has a programmatic equivalent
- [ ] **No context starvation** - Agent knows about all relevant resources
- [ ] **No artificial limits** - Don't gate capabilities arbitrarily
- [ ] **No heuristic completion** - Explicit signals, not guessing

---

## Review Process

When reviewing a feature or plan:

### Step 1: Parity Audit
List every user action. For each one:
```
Action: [describe]
UI Method: [how user does it]
Agent Method: [how agent does it, or "MISSING"]
```

### Step 2: Granularity Check
For each tool/API:
```
Tool: [name]
Actions: [list what it does]
Atomic? [yes/no - should be ONE action]
Composable? [yes/no - can combine with others]
```

### Step 3: Data Access Review
```
Data Type: [what data]
Storage: [files/database/API]
Agent Readable? [yes/no]
Agent Writable? [yes/no]
Human Inspectable? [yes/no]
```

### Step 4: Emergence Test
> "Describe an unanticipated outcome within your domain. Can the agent compose tools to achieve it?"

Try 3 novel requests the product wasn't explicitly designed for. Can the agent accomplish them?

---

## Scene Ripper Specific Recommendations

Based on the current architecture, ensure:

### Existing Capabilities Need Agent Access
1. **Video import** - CLI/API to import video by path
2. **Scene detection** - Trigger detection with parameters programmatically
3. **Export clips** - Export specific clips by ID/timecode
4. **YouTube search/download** - API access to search and download
5. **Transcription** - Trigger transcription via tool
6. **Color analysis** - Run analysis on specific clips
7. **Shot classification** - Classify shots programmatically

### Data Should Be File-Based
- Project state in `projects/{id}/project.json`
- Clips metadata in `projects/{id}/clips/`
- Transcripts in `projects/{id}/transcripts/`
- Thumbnails accessible by path
- EDL/XML exports as files

### Missing Agent Capabilities (Audit)
For each UI feature, verify there's a non-UI equivalent:
- Settings modification
- Project creation/switching
- Clip selection/filtering
- Batch operations
- Preview playback control

---

## Output Format

When reviewing, produce:

```markdown
## Agent-Native Review: [Feature/Plan Name]

### Parity Score: X/10
[List of orphan UI actions if any]

### Granularity Score: X/10
[List of non-atomic tools if any]

### Composability Score: X/10
[Features requiring code vs prompt-only]

### Data Accessibility: X/10
[Data not accessible to agents]

### Recommendations
1. [Highest priority fix]
2. [Second priority]
3. [Third priority]

### Emergence Test Results
- Test 1: [Novel request] - [Pass/Fail]
- Test 2: [Novel request] - [Pass/Fail]
- Test 3: [Novel request] - [Pass/Fail]
```
