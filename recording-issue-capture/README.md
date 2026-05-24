# Recording Issue Capture

`recording-issues` turns a long recording into actionable issue drafts and can publish them to Markdown/JSON, GitHub, Notion, or Linear.

The project is standalone: it does not import Scene Ripper or any repo-local modules. It expects `ffmpeg` and `ffprobe` on `PATH` for media chunking and duration detection.

By default, generated issues must include verified provenance: an `exact_quote` copied from the transcript or Zoom chat, a source file/path, a timestamp, and a validation status. Paraphrase-only or inferred issues are written to `rejected_issues.json` and are not published unless you explicitly opt out.

## Install

```bash
python -m pip install -e .
```

For local transcription:

```bash
python -m pip install -e ".[local]"
```

## Basic Run

```bash
recording-issues capture /path/to/recording-or-folder \
  --output-dir /tmp/recording-issues/run-001 \
  --transcription-backend openai \
  --transcription-model whisper-1 \
  --extractor llm \
  --idea-model gpt-5.2 \
  --curate-issues
```

OpenAI transcription and extraction read `OPENAI_API_KEY`. Local extraction can use Ollama:

```bash
recording-issues capture /path/to/recording-or-folder \
  --skip-transcription \
  --output-dir /tmp/recording-issues/run-001 \
  --extractor llm \
  --idea-model ollama/qwen3:8b \
  --curate-issues
```

## Destinations

Markdown and JSON are always written:

```text
<output-dir>/issues/*.md
<output-dir>/issues.json
<output-dir>/rejected_issues.json
<output-dir>/provenance_report.json
<output-dir>/manifest.json
<output-dir>/notion.md
```

Create GitHub issues with the GitHub CLI:

```bash
recording-issues publish-github /tmp/recording-issues/run-001 \
  --repo owner/repo
```

Create one Notion page containing the issue batch:

```bash
export NOTION_API_KEY=...
recording-issues publish-notion /tmp/recording-issues/run-001 \
  --parent-page-id YOUR_PARENT_PAGE_ID
```

Create Linear issues:

```bash
export LINEAR_API_KEY=...
recording-issues publish-linear /tmp/recording-issues/run-001 \
  --team-id YOUR_LINEAR_TEAM_ID \
  --project-id OPTIONAL_PROJECT_ID
```

## Long Runs

Use bounded transcription passes for long recordings:

```bash
recording-issues capture /path/to/recordings \
  --output-dir /tmp/recording-issues/run-001 \
  --transcription-backend openai \
  --extractor none \
  --max-new-chunks 20
```

Resume until complete:

```bash
recording-issues capture /path/to/recordings \
  --output-dir /tmp/recording-issues/run-001 \
  --transcription-backend openai \
  --extractor none \
  --transcription-loop \
  --max-new-chunks 20
```

Then extract from cached transcripts:

```bash
recording-issues capture /path/to/recordings \
  --output-dir /tmp/recording-issues/run-001 \
  --skip-transcription \
  --extractor llm \
  --idea-model gpt-5.2 \
  --curate-issues
```

## Provenance Controls

Strict provenance is enabled by default:

```bash
recording-issues capture /path/to/recordings \
  --output-dir /tmp/recording-issues/run-001 \
  --skip-transcription \
  --require-exact-provenance \
  --no-allow-inferred
```

Each accepted issue includes:

- `exact_quote`: verbatim transcript/chat text
- `provenance_kind`: `explicit`, `observed`, or `inferred`
- `validation_status`: normally `verified_exact_quote`
- source path, source file, and timestamp narrowed to the matching transcript/chat segment when possible

Use `--allow-inferred` only when you want issues that are grounded in an exact quote but still require product judgment. Use `--no-require-exact-provenance` for exploratory drafts; publishing commands still refuse unverified issues unless their explicit allow flag is passed.

## Destination Notes

- GitHub publishing uses `gh issue create` and checks exact titles before creating.
- GitHub publishing refuses issues without `verified_exact_quote` provenance unless `--allow-unverified` is passed. The generated shell script has the same guard; set `ALLOW_UNVERIFIED=1` only for exploratory drafts.
- Notion publishing uses Notion's page-create endpoint with Markdown page content.
- Linear publishing uses Linear's GraphQL `issueCreate` mutation. Labels are included in the description; this first version does not create Linear label entities.
