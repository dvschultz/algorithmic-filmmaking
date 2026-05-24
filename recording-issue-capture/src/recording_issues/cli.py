from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .core import (
    assemble_cached_transcript,
    chunk_chats,
    chunk_transcript,
    create_github_issues,
    curate_issues_cached,
    discover_chats,
    discover_media,
    extract_ideas_heuristic,
    extract_ideas_llm_cached,
    keep_provenance_approved,
    load_transcript,
    manifest_issues,
    profile_slug,
    publish_linear,
    publish_notion,
    transcribe_media_faster_whisper,
    transcribe_media_openai,
    transcribe_resumable,
    transcript_cache_path,
    transcript_status,
    verify_issues,
    write_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recording-issues")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Transcribe recordings and write issue drafts")
    capture.add_argument("root", type=Path)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--transcription-backend", choices=["openai", "faster-whisper", "none"], default="openai")
    capture.add_argument("--transcription-model", default="whisper-1")
    capture.add_argument("--language", default="en")
    capture.add_argument("--chunk-seconds", type=int, default=300)
    capture.add_argument("--skip-transcription", action="store_true")
    capture.add_argument("--extractor", choices=["llm", "heuristic", "none"], default="llm")
    capture.add_argument("--idea-model", default="gpt-5.2")
    capture.add_argument("--context", default="Software/product project receiving issues from a recording")
    capture.add_argument("--include-chat", action=argparse.BooleanOptionalAction, default=True)
    capture.add_argument("--curate-issues", action=argparse.BooleanOptionalAction, default=False)
    capture.add_argument("--curation-model", default=None)
    capture.add_argument("--curation-target", type=int, default=80)
    capture.add_argument("--curation-batch-size", type=int, default=40)
    capture.add_argument("--require-exact-provenance", action=argparse.BooleanOptionalAction, default=True)
    capture.add_argument("--allow-inferred", action=argparse.BooleanOptionalAction, default=False)
    capture.add_argument("--max-media", type=int, default=None)
    capture.add_argument("--max-new-chunks", type=int, default=None)
    capture.add_argument("--transcription-loop", action="store_true")
    capture.add_argument("--loop-sleep-seconds", type=float, default=0.0)
    capture.add_argument("--reuse-caches", action=argparse.BooleanOptionalAction, default=True)
    capture.add_argument("--status", action="store_true")
    capture.add_argument("--status-json", action="store_true")

    github = subparsers.add_parser("publish-github", help="Create GitHub issues from a run")
    github.add_argument("output_dir", type=Path)
    github.add_argument("--repo", default=None)
    github.add_argument("--allow-unverified", action="store_true")

    notion = subparsers.add_parser("publish-notion", help="Create a Notion page from a run")
    notion.add_argument("output_dir", type=Path)
    notion.add_argument("--parent-page-id", required=True)
    notion.add_argument("--title", default="Recording Issue Capture")

    linear = subparsers.add_parser("publish-linear", help="Create Linear issues from a run")
    linear.add_argument("output_dir", type=Path)
    linear.add_argument("--team-id", required=True)
    linear.add_argument("--project-id", default=None)

    inspect = subparsers.add_parser("inspect", help="Print issue titles from a run")
    inspect.add_argument("output_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        return capture(args)
    if args.command == "publish-github":
        for url in create_github_issues(args.output_dir, repo=args.repo, allow_unverified=args.allow_unverified):
            print(url)
        return 0
    if args.command == "publish-notion":
        print(publish_notion(args.output_dir, parent_page_id=args.parent_page_id, title=args.title))
        return 0
    if args.command == "publish-linear":
        for url in publish_linear(args.output_dir, team_id=args.team_id, project_id=args.project_id):
            print(url)
        return 0
    if args.command == "inspect":
        for index, issue in enumerate(manifest_issues(args.output_dir), start=1):
            print(f"{index:03d}. {issue.title}")
        return 0
    return 2


def capture(args: argparse.Namespace) -> int:
    media = discover_media(args.root)
    if args.max_media is not None:
        media = media[: args.max_media]
    if not media:
        print(f"No media found under {args.root}", file=sys.stderr)
        return 2
    chats = discover_chats(args.root) if args.include_chat else []

    if args.status or args.status_json:
        status = run_status(media, args)
        if args.status_json:
            print(json.dumps(status, indent=2))
        else:
            print_status(status)
        return 0

    if args.transcription_loop:
        while True:
            status = run_status(media, args)
            if status["remaining_chunk_count"] == 0:
                break
            run_transcription_pass(media, args)
            if args.loop_sleep_seconds:
                time.sleep(args.loop_sleep_seconds)

    all_segments = []
    if args.skip_transcription or args.transcription_backend == "none":
        profile = profile_slug(args.transcription_backend, args.transcription_model, args.language)
        for item in media:
            cache = transcript_cache_path(args.output_dir, item.path, profile)
            if not cache.exists() and args.transcription_backend != "none":
                assembled = assemble_cached_transcript(
                    item,
                    output_dir=args.output_dir,
                    backend=args.transcription_backend,
                    model_name=args.transcription_model,
                    language=args.language,
                    chunk_seconds=args.chunk_seconds,
                )
                if assembled is None:
                    print(f"Missing transcript cache for {item.path}", file=sys.stderr)
                    return 1
            if cache.exists():
                all_segments.extend(load_transcript(cache))
    else:
        for item in media:
            cache = transcript_cache_path(
                args.output_dir,
                item.path,
                profile_slug(args.transcription_backend, args.transcription_model, args.language),
            )
            if args.reuse_caches and cache.exists():
                all_segments.extend(load_transcript(cache))
                continue
            if args.max_new_chunks is not None:
                segments, _new_count = transcribe_resumable_item(item, args)
            else:
                segments = transcribe_whole_item(item, args)
            if args.max_new_chunks is None:
                from .core import save_transcript

                save_transcript(cache, item, segments)
            all_segments.extend(segments)

    status = run_status(media, args) if args.transcription_backend != "none" else None
    chunks = chunk_transcript(all_segments) + chunk_chats(chats)
    raw_issues = extract_issues(chunks, args)
    verified_raw_issues = verify_issues(raw_issues, chunks)
    issues = keep_provenance_approved(
        verified_raw_issues,
        require_exact=args.require_exact_provenance,
        allow_inferred=args.allow_inferred,
    )
    rejected = [issue for issue in verified_raw_issues if issue not in issues]
    if args.curate_issues and issues:
        raw_count = len(issues)
        issues = curate_issues_cached(
            issues,
            output_dir=args.output_dir,
            model=args.curation_model or args.idea_model,
            context=args.context,
            target_count=args.curation_target,
            batch_size=args.curation_batch_size,
            reuse_cache=args.reuse_caches,
        )
        issues = verify_issues(issues, chunks)
        rejected.extend(
            issue
            for issue in issues
            if issue
            not in keep_provenance_approved(
                [issue],
                require_exact=args.require_exact_provenance,
                allow_inferred=args.allow_inferred,
            )
        )
        issues = keep_provenance_approved(
            issues,
            require_exact=args.require_exact_provenance,
            allow_inferred=args.allow_inferred,
        )
        print(f"Issues after curation: {len(issues)} (from {raw_count} raw candidates)")
    manifest = write_outputs(
        args.output_dir,
        media,
        chats,
        issues,
        status=status,
        config=vars(args),
        rejected_issues=rejected,
    )
    print(f"Output: {args.output_dir}")
    print(f"Media files: {len(media)}")
    print(f"Issues generated: {len(manifest['issues'])}")
    return 0


def run_transcription_pass(media, args: argparse.Namespace) -> None:
    remaining = args.max_new_chunks
    for item in media:
        if remaining is not None and remaining <= 0:
            return
        _segments, new_count = transcribe_resumable_item(item, args, max_new_chunks=remaining)
        if remaining is not None:
            remaining -= new_count
    for item in media:
        assemble_cached_transcript(
            item,
            output_dir=args.output_dir,
            backend=args.transcription_backend,
            model_name=args.transcription_model,
            language=args.language,
            chunk_seconds=args.chunk_seconds,
        )


def transcribe_resumable_item(item, args: argparse.Namespace, max_new_chunks: int | None = None):
    return transcribe_resumable(
        item,
        output_dir=args.output_dir,
        backend=args.transcription_backend,
        model_name=args.transcription_model,
        language=args.language,
        chunk_seconds=args.chunk_seconds,
        max_new_chunks=args.max_new_chunks if max_new_chunks is None else max_new_chunks,
    )


def transcribe_whole_item(item, args: argparse.Namespace):
    if args.transcription_backend == "openai":
        return transcribe_media_openai(
            item,
            model_name=args.transcription_model,
            language=args.language,
        )
    if args.transcription_backend == "faster-whisper":
        return transcribe_media_faster_whisper(
            item,
            model_name=args.transcription_model,
            language=args.language,
        )
    raise ValueError(f"Unsupported transcription backend: {args.transcription_backend}")


def extract_issues(chunks, args: argparse.Namespace):
    if args.extractor == "none":
        return []
    if args.extractor == "heuristic":
        return extract_ideas_heuristic(chunks)
    return extract_ideas_llm_cached(
        chunks,
        output_dir=args.output_dir,
        model=args.idea_model,
        context=args.context,
        reuse_cache=args.reuse_caches,
    )


def run_status(media, args: argparse.Namespace) -> dict:
    return transcript_status(
        media,
        output_dir=args.output_dir,
        backend=args.transcription_backend,
        model_name=args.transcription_model,
        language=args.language,
        chunk_seconds=args.chunk_seconds,
    )


def print_status(status: dict) -> None:
    print(
        "Transcript chunks: "
        f"{status['completed_chunk_count']}/{status['total_chunk_count']} complete "
        f"({status['remaining_chunk_count']} remaining)"
    )
    print(
        "Final transcript caches: "
        f"{status['final_cache_count']}/{status['media_count']} present"
    )
    for item in status["media"]:
        print(
            f"- {item['source_name']}: {item['completed_chunk_count']}/"
            f"{item['chunk_count']} chunks, final_cache={item['final_cache_exists']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
