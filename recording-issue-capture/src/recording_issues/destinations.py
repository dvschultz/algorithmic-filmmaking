from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.request
from pathlib import Path
from typing import Sequence

from .models import IssueCandidate, TranscriptChunk, TranscriptSegment
from .text_utils import format_timestamp, safe_slug


def render_issue_markdown(issue: IssueCandidate) -> str:
    lines = [
        "## Summary",
        issue.summary.strip(),
        "",
        "## Source",
        f"- File: `{issue.source_name}`",
        f"- Path: `{issue.source_path}`",
    ]
    if issue.start_time is not None:
        time_range = format_timestamp(issue.start_time)
        if issue.end_time is not None and issue.end_time > issue.start_time:
            time_range += f" - {format_timestamp(issue.end_time)}"
        lines.append(f"- Transcript time: `{time_range}`")
    lines.extend(
        [
            "",
            "## Provenance",
            f"- Type: `{issue.provenance_kind}`",
            f"- Validation: `{issue.validation_status}`",
        ]
    )
    if issue.validation_errors:
        lines.append(f"- Validation errors: `{'; '.join(issue.validation_errors)}`")
    if issue.provenance_notes:
        lines.extend(["", issue.provenance_notes.strip()])
    if issue.exact_quote:
        lines.extend(["", "## Exact Quote", f"> {issue.exact_quote.strip()}"])
    if issue.evidence:
        lines.extend(["", "## Evidence", f"> {issue.evidence.strip()}"])
    if issue.acceptance_criteria:
        lines.extend(["", "## Acceptance Criteria"])
        lines.extend(f"- {item}" for item in issue.acceptance_criteria)
    if issue.implementation_notes:
        lines.extend(["", "## Implementation Notes", issue.implementation_notes.strip()])
    lines.extend(["", f"_Extraction confidence: {issue.confidence:.2f}_"])
    return "\n".join(lines).strip() + "\n"


def render_notion_markdown(issues: Sequence[IssueCandidate]) -> str:
    lines = ["# Recording Issue Capture", ""]
    for index, issue in enumerate(issues, start=1):
        lines.extend([f"## {index:03d}. {issue.title}", "", issue.summary.strip(), ""])
        if issue.labels:
            lines.extend([f"Labels: {', '.join(issue.labels)}", ""])
        lines.extend([f"Provenance: {issue.provenance_kind} / {issue.validation_status}", ""])
        if issue.exact_quote:
            lines.extend([f"> {issue.exact_quote.strip()}", ""])
        if issue.evidence:
            lines.extend([f"Evidence: {issue.evidence.strip()}", ""])
    return "\n".join(lines).strip() + "\n"


def write_github_script(output_dir: Path, issues: Sequence[IssueCandidate]) -> Path:
    path = output_dir / "create_github_issues.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'repo="${1:-}"', ""]
    for issue_file, issue in zip(sorted((output_dir / "issues").glob("*.md")), issues):
        labels = []
        for label in issue.labels:
            labels.extend(["--label", label])
        repo_args = '${repo:+--repo "$repo"}'
        list_cmd = f"gh issue list {repo_args} --state all --limit 200 --json title,url"
        jq_expr = f".[] | select(.title == {json.dumps(issue.title)}) | .url"
        create_cmd = [
            "gh",
            "issue",
            "create",
            "--title",
            issue.title,
            "--body-file",
            str(issue_file),
            *labels,
        ]
        lines.append(f"if [ \"${{ALLOW_UNVERIFIED:-0}}\" != \"1\" ] && ! grep -q 'Validation: `verified_exact_quote`' {shlex.quote(str(issue_file))}; then")
        lines.append(f"  echo 'Refusing to publish unverified issue body: {shlex.quote(str(issue_file))}' >&2")
        lines.append("  exit 1")
        lines.append("fi")
        lines.append(f"existing_url=$({list_cmd} --jq {shlex.quote(jq_expr)} | head -n 1)")
        lines.append('if [ -n "$existing_url" ]; then echo "$existing_url"; else')
        lines.append("  " + " ".join(shlex.quote(part) for part in create_cmd) + ' ${repo:+--repo "$repo"}')
        lines.append("fi")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def load_manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))


def manifest_issues(output_dir: Path) -> list[IssueCandidate]:
    return [IssueCandidate(**item) for item in load_manifest(output_dir).get("issues", [])]


def publishable_manifest_issues(output_dir: Path, *, allow_unverified: bool = False) -> list[IssueCandidate]:
    """Load issues and re-verify exact quotes against cached transcripts before publish."""
    issues = manifest_issues(output_dir)
    if allow_unverified:
        return issues

    chunks = manifest_provenance_chunks(output_dir)
    if not chunks:
        raise RuntimeError(
            "Refusing to publish because no transcript/chat provenance could be loaded for re-validation."
        )
    from .core import require_publishable_issues, verify_issues

    verified = verify_issues(issues, chunks)
    require_publishable_issues(verified)
    return verified


def manifest_provenance_chunks(output_dir: Path) -> list[TranscriptChunk]:
    """Rebuild transcript/chat chunks from a capture manifest for publish-time checks."""
    from .core import (
        chunk_transcript,
        find_transcript_cache,
        load_transcript,
        parse_zoom_chat,
        profile_slug,
    )

    manifest = load_manifest(output_dir)
    config = manifest.get("run_config", {})
    backend = str(config.get("transcription_backend") or "none")
    model = str(config.get("transcription_model") or "")
    language = str(config.get("language") or "en")
    profile = profile_slug(backend, model, language)

    segments: list[TranscriptSegment] = []
    for item in manifest.get("media", []):
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path") or item.get("source_path")
        if not raw_path:
            continue
        cache = find_transcript_cache(output_dir, Path(str(raw_path)), profile)
        if cache is not None:
            segments.extend(load_transcript(cache))

    chunks = chunk_transcript(segments) if segments else []
    for raw_path in manifest.get("chat_logs", []):
        path = Path(str(raw_path))
        if path.exists():
            chunks.extend(chunk_transcript(parse_zoom_chat(path), max_chars=6000))
    return chunks


def find_transcript_cache(output_dir: Path, media_path: Path, profile: str) -> Path | None:
    """Find a final transcript cache even if the original media file moved."""
    from .core import transcript_cache_path

    cache = transcript_cache_path(output_dir, media_path, profile)
    if cache.exists():
        return cache
    transcript_dir = output_dir / "transcripts" / profile
    stem = safe_slug(media_path.stem)[:80]
    matches = sorted(transcript_dir.glob(f"{stem}-*.json")) if transcript_dir.exists() else []
    return matches[0] if matches else None


def create_github_issues(output_dir: Path, *, repo: str | None = None, allow_unverified: bool = False) -> list[str]:
    issues = publishable_manifest_issues(output_dir, allow_unverified=allow_unverified)
    issue_files = sorted((output_dir / "issues").glob("*.md"))
    if len(issue_files) != len(issues):
        raise RuntimeError(
            f"Expected {len(issues)} issue body files, found {len(issue_files)} in {output_dir / 'issues'}"
        )
    urls: list[str] = []
    for issue_file, issue in zip(issue_files, issues):
        from . import core as core_facade

        existing = core_facade.find_existing_github_issue(issue.title, repo=repo)
        if existing:
            urls.append(existing)
            continue
        issue_file.write_text(render_issue_markdown(issue), encoding="utf-8")
        command = ["gh", "issue", "create", "--title", issue.title, "--body-file", str(issue_file)]
        for label in issue.labels:
            command.extend(["--label", label])
        if repo:
            command.extend(["--repo", repo])
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        urls.append(result.stdout.strip())
    update_manifest_urls(output_dir, urls)
    return urls


def find_existing_github_issue(title: str, *, repo: str | None = None) -> str | None:
    command = ["gh", "issue", "list", "--state", "all", "--limit", "200", "--json", "title,url"]
    if repo:
        command.extend(["--repo", repo])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    for item in json.loads(result.stdout):
        if item.get("title") == title:
            return item.get("url")
    return None


def publish_linear(
    output_dir: Path,
    *,
    team_id: str,
    project_id: str | None = None,
    api_key: str | None = None,
) -> list[str]:
    api_key = api_key or os.environ.get("LINEAR_API_KEY")
    if not api_key:
        raise RuntimeError("LINEAR_API_KEY is required")
    urls: list[str] = []
    issues = publishable_manifest_issues(output_dir)
    for issue in issues:
        body = {
            "query": """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue { id identifier url }
  }
}
""",
            "variables": {
                "input": {
                    "teamId": team_id,
                    "projectId": project_id,
                    "title": issue.title,
                    "description": render_issue_markdown(issue),
                }
            },
        }
        if project_id is None:
            body["variables"]["input"].pop("projectId")
        data = post_json("https://api.linear.app/graphql", body, {"Authorization": api_key})
        urls.append(data["data"]["issueCreate"]["issue"]["url"])
    update_manifest_urls(output_dir, urls)
    return urls


def publish_notion(
    output_dir: Path,
    *,
    parent_page_id: str,
    title: str = "Recording Issue Capture",
    api_key: str | None = None,
) -> str:
    api_key = api_key or os.environ.get("NOTION_API_KEY")
    if not api_key:
        raise RuntimeError("NOTION_API_KEY is required")
    issues = publishable_manifest_issues(output_dir)
    limited_children: list[dict] = []
    for issue in issues:
        limited_children.extend(
            [
                {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rich_text(issue.title)}},
                {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_text(issue.summary[:1800])}},
            ]
        )
        if len(limited_children) >= 90:
            break
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {"title": {"title": rich_text(title)}},
        "children": limited_children,
    }
    data = post_json(
        "https://api.notion.com/v1/pages",
        payload,
        {"Authorization": f"Bearer {api_key}", "Notion-Version": "2026-03-11"},
    )
    url = data.get("url", "")
    update_manifest_urls(output_dir, [url])
    return url


def rich_text(text: str) -> list[dict]:
    return [{"type": "text", "text": {"content": text[:2000]}}]


def post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def update_manifest_urls(output_dir: Path, urls: Sequence[str]) -> None:
    path = output_dir / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    existing = data.get("published_urls", [])
    data["published_urls"] = sorted(set(existing + list(urls)))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
