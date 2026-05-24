from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from .models import (
    AUDIO_EXTENSIONS,
    ISSUE_LABELS,
    PROMPT_VERSION,
    PROVENANCE_KINDS,
    VIDEO_EXTENSIONS,
    IssueCandidate,
    MediaChunk,
    RecordingMedia,
    TranscriptChunk,
    TranscriptSegment,
)
from .destinations import (
    create_github_issues,
    find_existing_github_issue,
    find_transcript_cache,
    load_manifest,
    manifest_issues,
    manifest_provenance_chunks,
    post_json,
    publish_linear,
    publish_notion,
    publishable_manifest_issues,
    render_issue_markdown,
    render_notion_markdown,
    rich_text,
    update_manifest_urls,
    write_github_script,
)
from .text_utils import (
    format_timestamp,
    labels_from,
    looks_like_chatter,
    normalize_title,
    optional_float,
    safe_slug,
    split_sentences,
    string_list,
    timestamp_to_seconds,
    title_from_sentence,
    title_similarity,
)


def discover_media(root: Path) -> list[RecordingMedia]:
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    candidates: list[RecordingMedia] = []
    for path in paths:
        if not path.is_file() or path.name.startswith("._"):
            continue
        suffix = path.suffix.lower()
        if suffix in AUDIO_EXTENSIONS:
            candidates.append(RecordingMedia(path=path, kind="audio", dedupe_key=dedupe_key(path)))
        elif suffix in VIDEO_EXTENSIONS:
            candidates.append(RecordingMedia(path=path, kind="video", dedupe_key=dedupe_key(path)))

    chosen: dict[str, RecordingMedia] = {}
    for media in candidates:
        existing = chosen.get(media.dedupe_key)
        if existing is None or media_rank(media) < media_rank(existing):
            chosen[media.dedupe_key] = media
    return sorted(chosen.values(), key=lambda item: str(item.path).lower())


def discover_chats(root: Path) -> list[Path]:
    if root.is_file():
        return []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() == ".txt"
        and "chat" in path.name.lower()
    ]


def dedupe_key(path: Path) -> str:
    stem = re.sub(r"(_recording)?(_\d+x\d+)?(-part\d+)?$", "", path.stem.lower())
    stem = safe_slug(stem)
    return f"{path.parent.resolve()}::{stem}"


def media_rank(media: RecordingMedia) -> tuple[int, str]:
    return (0 if media.kind == "audio" else 1, str(media.path).lower())


def cache_key_for_media(path: Path) -> str:
    try:
        stat = path.stat()
        identity = f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    except FileNotFoundError:
        identity = str(path)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def profile_slug(backend: str, model: str, language: str) -> str:
    return safe_slug(f"{backend}-{model}-{language}")


def transcript_cache_path(output_dir: Path, media_path: Path, profile: str) -> Path:
    stem = safe_slug(media_path.stem)[:80]
    return output_dir / "transcripts" / profile / f"{stem}-{cache_key_for_media(media_path)}.json"


def chunk_cache_path(output_dir: Path, media_path: Path, profile: str, chunk: MediaChunk) -> Path:
    stem = safe_slug(media_path.stem)[:72]
    key = cache_key_for_media(media_path)
    return (
        output_dir
        / "transcripts"
        / "chunks"
        / profile
        / key
        / f"{stem}-{chunk.index:04d}-{int(chunk.start_time):06d}-{int(chunk.end_time):06d}.json"
    )


def load_transcript(path: Path) -> list[TranscriptSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment(**item) for item in data.get("segments", [])]


def save_transcript(path: Path, media: RecordingMedia, segments: Sequence[TranscriptSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(media.path),
        "source_name": media.path.name,
        "kind": media.kind,
        "segments": [asdict(segment) for segment in segments],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def media_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    return float(result.stdout.strip())


def split_media_chunks(duration: float, chunk_seconds: int) -> list[MediaChunk]:
    chunks: list[MediaChunk] = []
    start = 0.0
    index = 0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        chunks.append(MediaChunk(index=index, start_time=start, end_time=end))
        index += 1
        start = end
    return chunks


def extract_media_chunk(source: Path, chunk: MediaChunk, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{chunk.start_time:.3f}",
        "-to",
        f"{chunk.end_time:.3f}",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "mp3",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def transcribe_media_openai(
    media: RecordingMedia,
    *,
    model_name: str = "whisper-1",
    language: str = "en",
    client=None,
) -> list[TranscriptSegment]:
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI transcription")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

    with media.path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model_name,
            file=audio_file,
            language=None if language == "auto" else language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return openai_transcription_segments(response, media, offset=0.0)


def transcribe_media_faster_whisper(
    media: RecordingMedia,
    *,
    model_name: str = "large-v3",
    language: str = "en",
) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="auto")
    raw_segments, _info = model.transcribe(str(media.path), language=None if language == "auto" else language)
    return [
        TranscriptSegment(
            source_path=str(media.path),
            source_name=media.path.name,
            start_time=float(segment.start),
            end_time=float(segment.end),
            text=segment.text.strip(),
            confidence=0.0,
            language=None if language == "auto" else language,
        )
        for segment in raw_segments
        if segment.text.strip()
    ]


def openai_transcription_segments(response, media: RecordingMedia, *, offset: float) -> list[TranscriptSegment]:
    segments = getattr(response, "segments", None)
    if segments is None and isinstance(response, dict):
        segments = response.get("segments")
    if not segments:
        text = getattr(response, "text", "") if not isinstance(response, dict) else response.get("text", "")
        return [
            TranscriptSegment(
                source_path=str(media.path),
                source_name=media.path.name,
                start_time=offset,
                end_time=offset,
                text=str(text).strip(),
                language=None,
            )
        ] if str(text).strip() else []
    parsed: list[TranscriptSegment] = []
    for segment in segments:
        get = segment.get if isinstance(segment, dict) else lambda key, default=None: getattr(segment, key, default)
        text = str(get("text", "")).strip()
        if text:
            parsed.append(
                TranscriptSegment(
                    source_path=str(media.path),
                    source_name=media.path.name,
                    start_time=offset + float(get("start", 0.0)),
                    end_time=offset + float(get("end", 0.0)),
                    text=text,
                    language=None,
                )
            )
    return parsed


def transcribe_resumable(
    media: RecordingMedia,
    *,
    output_dir: Path,
    backend: str,
    model_name: str,
    language: str,
    chunk_seconds: int,
    max_new_chunks: int | None = None,
) -> tuple[list[TranscriptSegment], int]:
    profile = profile_slug(backend, model_name, language)
    chunks = split_media_chunks(media_duration(media.path), chunk_seconds)
    all_segments: list[TranscriptSegment] = []
    new_chunks = 0
    for chunk in chunks:
        cache_path = chunk_cache_path(output_dir, media.path, profile, chunk)
        if cache_path.exists():
            all_segments.extend(load_transcript(cache_path))
            continue
        if max_new_chunks is not None and new_chunks >= max_new_chunks:
            break
        with tempfile.TemporaryDirectory(prefix="recording-issues-") as tmp:
            chunk_media = RecordingMedia(
                path=Path(tmp) / f"{media.path.stem}-{chunk.index:04d}.mp3",
                kind="audio",
                dedupe_key=media.dedupe_key,
            )
            extract_media_chunk(media.path, chunk, chunk_media.path)
            if backend == "openai":
                segments = transcribe_media_openai(
                    chunk_media,
                    model_name=model_name,
                    language=language,
                )
            elif backend == "faster-whisper":
                segments = transcribe_media_faster_whisper(
                    chunk_media,
                    model_name=model_name,
                    language=language,
                )
            else:
                raise ValueError(f"Unsupported transcription backend: {backend}")
            adjusted = [
                TranscriptSegment(
                    source_path=str(media.path),
                    source_name=media.path.name,
                    start_time=chunk.start_time + segment.start_time,
                    end_time=chunk.start_time + segment.end_time,
                    text=segment.text,
                    confidence=segment.confidence,
                    language=segment.language,
                )
                for segment in segments
            ]
            save_transcript(cache_path, media, adjusted)
            all_segments.extend(adjusted)
            new_chunks += 1
    return all_segments, new_chunks


def assemble_cached_transcript(
    media: RecordingMedia,
    *,
    output_dir: Path,
    backend: str,
    model_name: str,
    language: str,
    chunk_seconds: int,
) -> list[TranscriptSegment] | None:
    profile = profile_slug(backend, model_name, language)
    chunks = split_media_chunks(media_duration(media.path), chunk_seconds)
    segments: list[TranscriptSegment] = []
    for chunk in chunks:
        cache = chunk_cache_path(output_dir, media.path, profile, chunk)
        if not cache.exists():
            return None
        segments.extend(load_transcript(cache))
    save_transcript(transcript_cache_path(output_dir, media.path, profile), media, segments)
    return segments


def transcript_status(
    media: Sequence[RecordingMedia],
    *,
    output_dir: Path,
    backend: str,
    model_name: str,
    language: str,
    chunk_seconds: int,
) -> dict:
    profile = profile_slug(backend, model_name, language)
    items = []
    total = complete = 0
    final_caches = 0
    for item in media:
        duration = media_duration(item.path)
        chunks = split_media_chunks(duration, chunk_seconds)
        completed = sum(
            1 for chunk in chunks if chunk_cache_path(output_dir, item.path, profile, chunk).exists()
        )
        final_exists = transcript_cache_path(output_dir, item.path, profile).exists()
        final_caches += 1 if final_exists else 0
        total += len(chunks)
        complete += completed
        items.append(
            {
                "source_path": str(item.path),
                "source_name": item.path.name,
                "kind": item.kind,
                "duration_seconds": duration,
                "chunk_count": len(chunks),
                "completed_chunk_count": completed,
                "remaining_chunk_count": len(chunks) - completed,
                "final_cache_exists": final_exists,
            }
        )
    return {
        "media_count": len(media),
        "total_chunk_count": total,
        "completed_chunk_count": complete,
        "remaining_chunk_count": total - complete,
        "final_cache_count": final_caches,
        "media": items,
    }


def parse_zoom_chat(path: Path) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    pattern = re.compile(r"^(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        seconds = timestamp_to_seconds(match.group("time"))
        text = f"{match.group('speaker')}: {match.group('text').strip()}"
        if text.strip():
            segments.append(
                TranscriptSegment(
                    source_path=str(path),
                    source_name=path.name,
                    start_time=seconds,
                    end_time=seconds,
                    text=text,
                    confidence=1.0,
                    language="chat",
                )
            )
    return segments


def chunk_transcript(segments: Sequence[TranscriptSegment], max_chars: int = 9000) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_chars = 0
    for segment in segments:
        length = len(segment.text)
        if current and current_chars + length > max_chars:
            chunks.append(make_transcript_chunk(current))
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += length
    if current:
        chunks.append(make_transcript_chunk(current))
    return chunks


def make_transcript_chunk(segments: Sequence[TranscriptSegment]) -> TranscriptChunk:
    first = segments[0]
    last = segments[-1]
    text = "\n".join(
        f"[{format_timestamp(segment.start_time)}] {segment.text}" for segment in segments
    )
    return TranscriptChunk(
        source_name=first.source_name,
        source_path=first.source_path,
        start_time=first.start_time,
        end_time=last.end_time,
        text=text,
        segments=list(segments),
    )


def chunk_chats(paths: Sequence[Path], max_chars: int = 6000) -> list[TranscriptChunk]:
    chunks: list[TranscriptChunk] = []
    for path in paths:
        chunks.extend(chunk_transcript(parse_zoom_chat(path), max_chars=max_chars))
    return chunks


def extract_ideas_heuristic(chunks: Iterable[TranscriptChunk]) -> list[IssueCandidate]:
    trigger = re.compile(
        r"\b(we should|we need|need to|could|would be good|feature|bug|problem|"
        r"issue|add|support|automate|workflow|export|import|document)\b",
        re.IGNORECASE,
    )
    issues: list[IssueCandidate] = []
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if trigger.search(sentence) and not looks_like_chatter(sentence):
                issues.append(
                    IssueCandidate(
                        title=title_from_sentence(sentence),
                        summary=sentence.strip(),
                        source_path=chunk.source_path,
                        source_name=chunk.source_name,
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        evidence=sentence.strip(),
                        exact_quote=sentence.strip(),
                        provenance_kind="explicit",
                        provenance_notes="Matched by deterministic trigger in transcript text.",
                        acceptance_criteria=[
                            "Define the expected behavior.",
                            "Implement the change with focused tests.",
                            "Document any user-facing workflow change.",
                        ],
                        implementation_notes="Extracted by deterministic heuristics.",
                        confidence=0.45,
                    )
                )
    return dedupe_issues(issues)


def extract_ideas_llm_cached(
    chunks: Iterable[TranscriptChunk],
    *,
    output_dir: Path,
    model: str,
    context: str,
    reuse_cache: bool = True,
) -> list[IssueCandidate]:
    issues: list[IssueCandidate] = []
    for chunk in chunks:
        path = issue_cache_path(output_dir, chunk, "extract", model)
        if reuse_cache and path.exists():
            issues.extend(verify_issues(load_issues(path), [chunk]))
            continue
        chunk_issues = extract_ideas_llm(chunk, model=model, context=context)
        chunk_issues = verify_issues(chunk_issues, [chunk])
        save_issues(path, chunk_issues)
        issues.extend(chunk_issues)
    return filter_issues(issues)


def extract_ideas_llm(chunk: TranscriptChunk, *, model: str, context: str) -> list[IssueCandidate]:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract concrete, implementable issue candidates from a recording transcript. "
                "Ignore meeting logistics, resolved troubleshooting, generic praise, and inspiration "
                "that does not imply a specific work item. Return strict JSON only."
            ),
        },
        {"role": "user", "content": extraction_prompt(chunk, context=context)},
    ]
    response = complete_json(model=model, messages=messages, temperature=0.1)
    return issues_from_payload(response, fallback=chunk)


def curate_issues_cached(
    issues: Sequence[IssueCandidate],
    *,
    output_dir: Path,
    model: str,
    context: str,
    target_count: int = 80,
    batch_size: int = 40,
    reuse_cache: bool = True,
) -> list[IssueCandidate]:
    filtered = filter_issues(issues)
    if not filtered:
        return []
    batches = [filtered[index : index + batch_size] for index in range(0, len(filtered), batch_size)]
    batch_target = max(5, min(batch_size, (target_count + len(batches) - 1) // len(batches) + 3))
    curated: list[IssueCandidate] = []
    for index, batch in enumerate(batches):
        path = curation_cache_path(output_dir, batch, model=model, stage=f"batch-{index:03d}", target=batch_target)
        if reuse_cache and path.exists():
            curated.extend(load_issues(path))
            continue
        batch_curated = curate_issues(batch, model=model, context=context, target_count=batch_target)
        save_issues(path, batch_curated)
        curated.extend(batch_curated)
    curated = filter_issues(curated)
    if len(curated) <= target_count:
        return curated
    path = curation_cache_path(output_dir, curated, model=model, stage="final", target=target_count)
    if reuse_cache and path.exists():
        return filter_issues(load_issues(path))
    final = curate_issues(curated, model=model, context=context, target_count=target_count)
    save_issues(path, final)
    return filter_issues(final)


def curate_issues(
    issues: Sequence[IssueCandidate],
    *,
    model: str,
    context: str,
    target_count: int,
) -> list[IssueCandidate]:
    messages = [
        {
            "role": "system",
            "content": (
                "Curate raw issue candidates. Merge duplicates, drop weak or unsupported items, "
                "and keep only actionable work. Return strict JSON only."
            ),
        },
        {"role": "user", "content": curation_prompt(issues, context=context, target_count=target_count)},
    ]
    payload = complete_json(model=model, messages=messages, temperature=0.05, timeout=300)
    return filter_issues(issues_from_payload(payload, fallback=None))


def complete_json(
    *,
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: int = 120,
) -> dict:
    if model.startswith("ollama/"):
        return complete_json_ollama(model.removeprefix("ollama/"), messages=messages, temperature=temperature, timeout=timeout)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI LLM extraction")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return parse_json_text(response.choices[0].message.content or "")


def complete_json_ollama(
    model: str,
    *,
    messages: list[dict],
    temperature: float,
    timeout: int,
) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return parse_json_text(data.get("message", {}).get("content", ""))


def parse_json_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"issues": []}
        return json.loads(match.group(0))


def issues_from_payload(payload: dict, fallback: TranscriptChunk | None) -> list[IssueCandidate]:
    raw_issues = payload.get("issues", [])
    if not isinstance(raw_issues, list):
        return []
    issues: list[IssueCandidate] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        issues.append(
            IssueCandidate(
                title=str(item.get("title", "")).strip(),
                summary=str(item.get("summary", "")).strip(),
                source_path=str(item.get("source_path") or (fallback.source_path if fallback else "Multiple sources")),
                source_name=str(item.get("source_name") or (fallback.source_name if fallback else "Multiple sources")),
                start_time=optional_float(item.get("start_time"), fallback.start_time if fallback else None),
                end_time=optional_float(item.get("end_time"), fallback.end_time if fallback else None),
                evidence=str(item.get("evidence", "")).strip(),
                exact_quote=str(item.get("exact_quote", "")).strip(),
                provenance_kind=str(item.get("provenance_kind", "unverified")).strip().lower() or "unverified",
                provenance_notes=str(item.get("provenance_notes", "")).strip(),
                validation_status=str(item.get("validation_status", "unvalidated")).strip() or "unvalidated",
                validation_errors=string_list(item.get("validation_errors")),
                labels=labels_from(item.get("labels")),
                acceptance_criteria=string_list(item.get("acceptance_criteria")),
                implementation_notes=str(item.get("implementation_notes", "")).strip(),
                confidence=optional_float(item.get("confidence"), 0.75) or 0.75,
            )
        )
    return filter_issues(issues)


def verify_issues(issues: Sequence[IssueCandidate], chunks: Sequence[TranscriptChunk]) -> list[IssueCandidate]:
    return [verify_issue(issue, chunks) for issue in issues]


def verify_issue(issue: IssueCandidate, chunks: Sequence[TranscriptChunk]) -> IssueCandidate:
    issue.validation_errors = []
    issue.validation_status = "unvalidated"
    issue.exact_quote = issue.exact_quote.strip()
    issue.provenance_kind = issue.provenance_kind.strip().lower() or "unverified"
    if issue.provenance_kind not in PROVENANCE_KINDS:
        issue.validation_errors.append(f"unknown provenance_kind: {issue.provenance_kind}")
        issue.provenance_kind = "unverified"

    if not issue.exact_quote:
        issue.validation_status = "missing_exact_quote"
        issue.validation_errors.append("exact_quote is required for verified provenance")
        return issue

    match = find_quote_match(issue, chunks)
    if match is None:
        issue.validation_status = "quote_not_found"
        issue.validation_errors.append("exact_quote was not found in the source transcript/chat segments")
        return issue

    segment, chunk = match
    issue.validation_status = "verified_exact_quote"
    if issue.provenance_kind == "unverified":
        issue.provenance_kind = "explicit"
    if not issue.evidence:
        issue.evidence = issue.exact_quote
    issue.source_path = segment.source_path if segment else chunk.source_path
    issue.source_name = segment.source_name if segment else chunk.source_name
    issue.start_time = segment.start_time if segment else chunk.start_time
    issue.end_time = segment.end_time if segment else chunk.end_time
    return issue


def find_quote_match(
    issue: IssueCandidate,
    chunks: Sequence[TranscriptChunk],
) -> tuple[TranscriptSegment | None, TranscriptChunk] | None:
    quote = normalize_quote(issue.exact_quote)
    if not quote:
        return None
    for chunk in chunks:
        if not source_matches(issue, chunk):
            continue
        for segment in chunk.segments:
            if quote in normalize_quote(segment.text):
                return segment, chunk
        if quote in normalize_quote(chunk.text):
            return None, chunk
    return None


def source_matches(issue: IssueCandidate, chunk: TranscriptChunk) -> bool:
    issue_sources = {issue.source_path, issue.source_name}
    if "Multiple sources" in issue_sources:
        return True
    return chunk.source_path in issue_sources or chunk.source_name in issue_sources


def normalize_quote(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def keep_verified_issues(issues: Sequence[IssueCandidate]) -> list[IssueCandidate]:
    return [issue for issue in issues if issue.validation_status == "verified_exact_quote"]


def keep_provenance_approved(
    issues: Sequence[IssueCandidate],
    *,
    require_exact: bool = True,
    allow_inferred: bool = False,
) -> list[IssueCandidate]:
    approved: list[IssueCandidate] = []
    for issue in issues:
        if require_exact and issue.validation_status != "verified_exact_quote":
            continue
        if not allow_inferred and issue.provenance_kind == "inferred":
            continue
        approved.append(issue)
    return approved


def provenance_report(issues: Sequence[IssueCandidate]) -> dict:
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for issue in issues:
        by_status[issue.validation_status] = by_status.get(issue.validation_status, 0) + 1
        by_kind[issue.provenance_kind] = by_kind.get(issue.provenance_kind, 0) + 1
    return {
        "total": len(issues),
        "by_validation_status": by_status,
        "by_provenance_kind": by_kind,
        "invalid": [
            {
                "title": issue.title,
                "source_name": issue.source_name,
                "validation_status": issue.validation_status,
                "validation_errors": issue.validation_errors,
                "provenance_kind": issue.provenance_kind,
                "exact_quote": issue.exact_quote,
            }
            for issue in issues
            if issue.validation_status != "verified_exact_quote"
        ],
    }


def issue_cache_path(output_dir: Path, chunk: TranscriptChunk, stage: str, model: str) -> Path:
    identity = f"{PROMPT_VERSION}:{chunk.source_path}:{chunk.start_time}:{chunk.end_time}:{stage}:{model}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return output_dir / "idea-caches" / safe_slug(model) / f"{safe_slug(chunk.source_name)[:64]}-{digest}.json"


def curation_cache_path(
    output_dir: Path,
    issues: Sequence[IssueCandidate],
    *,
    model: str,
    stage: str,
    target: int,
) -> Path:
    identity = {
        "prompt_version": PROMPT_VERSION,
        "stage": stage,
        "model": model,
        "target": target,
        "issues": [
            {
                "title": issue.title,
                "summary": issue.summary,
                "exact_quote": issue.exact_quote,
                "source_path": issue.source_path,
            }
            for issue in issues
        ],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return output_dir / "issue-curation-caches" / safe_slug(model) / f"{stage}-{digest}.json"


def save_issues(path: Path, issues: Sequence[IssueCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"issues": [asdict(issue) for issue in issues]}, indent=2), encoding="utf-8")


def load_issues(path: Path) -> list[IssueCandidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [IssueCandidate(**item) for item in data.get("issues", [])]


def filter_issues(issues: Sequence[IssueCandidate]) -> list[IssueCandidate]:
    return dedupe_issues([issue for issue in issues if issue.title and issue.summary and not looks_like_chatter(issue.summary)])


def dedupe_issues(issues: Sequence[IssueCandidate]) -> list[IssueCandidate]:
    deduped: list[IssueCandidate] = []
    seen: set[str] = set()
    for issue in issues:
        key = normalize_title(issue.title)
        if not key or key in seen:
            continue
        if any(title_similarity(key, normalize_title(existing.title)) > 0.88 for existing in deduped):
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def write_outputs(
    output_dir: Path,
    media: Sequence[RecordingMedia],
    chats: Sequence[Path],
    issues: Sequence[IssueCandidate],
    *,
    status: dict | None,
    config: dict,
    published_urls: Sequence[str] | None = None,
    rejected_issues: Sequence[IssueCandidate] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    issue_dir = output_dir / "issues"
    issue_dir.mkdir(parents=True, exist_ok=True)
    for stale in issue_dir.glob("*.md"):
        stale.unlink()
    issue_files: list[str] = []
    for index, issue in enumerate(issues, start=1):
        path = issue_dir / f"{index:03d}-{safe_slug(issue.title)[:80]}.md"
        path.write_text(render_issue_markdown(issue), encoding="utf-8")
        issue_files.append(str(path))
    (output_dir / "issues.json").write_text(json.dumps({"issues": [asdict(issue) for issue in issues]}, indent=2), encoding="utf-8")
    (output_dir / "rejected_issues.json").write_text(
        json.dumps({"issues": [asdict(issue) for issue in rejected_issues or []]}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "provenance_report.json").write_text(
        json.dumps(
            {
                "accepted": provenance_report(issues),
                "rejected": provenance_report(rejected_issues or []),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "notion.md").write_text(render_notion_markdown(issues), encoding="utf-8")
    manifest = {
        "media": [asdict(item) | {"path": str(item.path)} for item in media],
        "chat_logs": [str(path) for path in chats],
        "transcript_status": status,
        "issue_files": issue_files,
        "published_urls": list(published_urls or []),
        "run_config": config,
        "issues": [asdict(issue) for issue in issues],
        "rejected_issue_count": len(rejected_issues or []),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_github_script(output_dir, issues)
    return manifest


def require_publishable_issues(issues: Sequence[IssueCandidate]) -> None:
    invalid = [issue for issue in issues if issue.validation_status != "verified_exact_quote"]
    if invalid:
        preview = "; ".join(f"{issue.title} ({issue.validation_status})" for issue in invalid[:5])
        more = f" and {len(invalid) - 5} more" if len(invalid) > 5 else ""
        raise RuntimeError(
            "Refusing to publish issues without verified exact provenance: "
            f"{preview}{more}. Re-run capture with strict provenance or pass the explicit allow flag."
        )

def extraction_prompt(chunk: TranscriptChunk, *, context: str) -> str:
    return f"""Destination context: {context}

Source file: {chunk.source_name}
Transcript range: {format_timestamp(chunk.start_time)} - {format_timestamp(chunk.end_time)}

Rules:
- Create issues only from the transcript/chat text below.
- exact_quote must be copied verbatim from the transcript/chat text, excluding the leading timestamp.
- evidence may summarize why the quote matters, but exact_quote must contain the source words.
- provenance_kind must be one of:
  - explicit: the speaker/chat directly asks for or identifies the work.
  - observed: the transcript describes a failure or UI behavior that implies the work.
  - inferred: the work is a product inference beyond what the text directly supports.
- Prefer explicit/observed issues. Use inferred only when the quote strongly supports the inference.
- If you cannot provide an exact_quote from the transcript/chat, do not return that issue.

Return JSON:
{{"issues":[{{"title":"imperative title","summary":"what should change and why","exact_quote":"verbatim transcript/chat excerpt","provenance_kind":"explicit|observed|inferred","provenance_notes":"why the quote supports this issue","evidence":"short explanation of the evidence","labels":["idea","recording-capture"],"acceptance_criteria":["done condition"],"implementation_notes":"optional","confidence":0.0}}]}}

Transcript:
{chunk.text}
"""


def curation_prompt(issues: Sequence[IssueCandidate], *, context: str, target_count: int) -> str:
    compact = [
        {
            "title": issue.title,
            "summary": issue.summary,
            "source_path": issue.source_path,
            "source_name": issue.source_name,
            "start_time": issue.start_time,
            "end_time": issue.end_time,
            "exact_quote": issue.exact_quote,
            "provenance_kind": issue.provenance_kind,
            "provenance_notes": issue.provenance_notes,
            "validation_status": issue.validation_status,
            "evidence": issue.evidence,
            "labels": issue.labels,
            "acceptance_criteria": issue.acceptance_criteria,
            "implementation_notes": issue.implementation_notes,
            "confidence": issue.confidence,
        }
        for issue in issues
    ]
    return f"""Destination context: {context}

Curate these raw candidates into at most {target_count} non-duplicative issues.
Merge related items, drop weak/non-actionable items, and preserve exact provenance.
Do not invent evidence. Every returned issue must keep an exact_quote from one of the raw candidates.
Drop candidates with validation_status other than "verified_exact_quote" unless the output is explicitly marked provenance_kind="inferred"; inferred items will be held as drafts and not published by default.
Return JSON with an "issues" array using the same fields.

Raw candidates:
{json.dumps(compact, indent=2)}
"""
