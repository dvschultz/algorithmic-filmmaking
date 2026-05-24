from pathlib import Path
import json


def test_discover_media_prefers_zoom_audio_and_skips_appledouble(tmp_path: Path):
    from recording_issues.core import discover_media

    week = tmp_path / "Week 1"
    week.mkdir()
    audio = week / "GMT20260414-001602_Recording.m4a"
    video = week / "GMT20260414-001602_Recording_1920x1200.mp4"
    standalone = week / "demo.mov"
    appledouble = week / "._demo.mov"
    for path in (audio, video, standalone, appledouble):
        path.write_text("x")

    media = discover_media(tmp_path)

    assert [item.path for item in media] == [standalone, audio]
    assert [item.kind for item in media] == ["video", "audio"]


def test_openai_transcription_segments_offsets_verbose_response(tmp_path: Path):
    from recording_issues.core import RecordingMedia, openai_transcription_segments

    media_path = tmp_path / "lesson.m4a"
    media_path.write_text("x")
    media = RecordingMedia(path=media_path, kind="audio", dedupe_key="lesson")
    response = {
        "segments": [
            {"start": 1.5, "end": 3.0, "text": " First idea "},
            {"start": 3.0, "end": 4.0, "text": ""},
        ]
    }

    segments = openai_transcription_segments(response, media, offset=300.0)

    assert len(segments) == 1
    assert segments[0].start_time == 301.5
    assert segments[0].end_time == 303.0
    assert segments[0].text == "First idea"


def test_heuristic_extractor_writes_issue_candidate():
    from recording_issues.core import TranscriptChunk, extract_ideas_heuristic

    chunk = TranscriptChunk(
        source_name="lesson.m4a",
        source_path="/tmp/lesson.m4a",
        start_time=0.0,
        end_time=30.0,
        text="We should add a Linear export workflow.",
    )

    issues = extract_ideas_heuristic([chunk])

    assert len(issues) == 1
    assert "Linear export" in issues[0].title
    assert issues[0].exact_quote == "We should add a Linear export workflow."
    assert issues[0].provenance_kind == "explicit"


def test_verify_issue_provenance_requires_exact_quote():
    from recording_issues.core import IssueCandidate, TranscriptSegment, chunk_transcript, verify_issues

    segments = [
        TranscriptSegment(
            source_path="/tmp/lesson.m4a",
            source_name="lesson.m4a",
            start_time=10.0,
            end_time=12.0,
            text="The prompt box is too short for editing.",
        )
    ]
    chunks = chunk_transcript(segments)

    verified = verify_issues(
        [
            IssueCandidate(
                title="Fix prompt box",
                summary="Make prompt editing more comfortable.",
                source_path="/tmp/lesson.m4a",
                source_name="lesson.m4a",
                exact_quote="The prompt box is too short for editing.",
                provenance_kind="observed",
            )
        ],
        chunks,
    )

    assert verified[0].validation_status == "verified_exact_quote"
    assert verified[0].start_time == 10.0
    assert verified[0].end_time == 12.0


def test_verify_issue_provenance_rejects_paraphrase_only():
    from recording_issues.core import IssueCandidate, TranscriptSegment, chunk_transcript, verify_issues

    chunks = chunk_transcript(
        [
            TranscriptSegment(
                source_path="/tmp/lesson.m4a",
                source_name="lesson.m4a",
                start_time=10.0,
                end_time=12.0,
                text="The box could be bigger.",
            )
        ]
    )

    verified = verify_issues(
        [
            IssueCandidate(
                title="Fix prompt box",
                summary="Make prompt editing more comfortable.",
                source_path="/tmp/lesson.m4a",
                source_name="lesson.m4a",
                evidence="Prompt box is too short.",
                provenance_kind="observed",
            )
        ],
        chunks,
    )

    assert verified[0].validation_status == "missing_exact_quote"
    assert "exact_quote is required" in verified[0].validation_errors[0]


def test_write_outputs_creates_portable_artifacts(tmp_path: Path):
    from recording_issues.core import IssueCandidate, RecordingMedia, write_outputs

    media_path = tmp_path / "lesson.m4a"
    media_path.write_text("x")
    issue = IssueCandidate(
        title="Add destination adapter",
        summary="Support another issue destination.",
        source_path=str(media_path),
        source_name=media_path.name,
        evidence="We should ship this.",
        exact_quote="We should ship this.",
        provenance_kind="explicit",
        validation_status="verified_exact_quote",
    )

    manifest = write_outputs(
        tmp_path / "out",
        [RecordingMedia(path=media_path, kind="audio", dedupe_key="lesson")],
        [],
        [issue],
        status={"completed_chunk_count": 1},
        config={"extractor": "heuristic"},
    )

    assert len(manifest["issues"]) == 1
    assert (tmp_path / "out" / "issues.json").exists()
    assert (tmp_path / "out" / "rejected_issues.json").exists()
    assert (tmp_path / "out" / "provenance_report.json").exists()
    assert (tmp_path / "out" / "notion.md").exists()
    assert (tmp_path / "out" / "create_github_issues.sh").exists()
    issue_body = (tmp_path / "out" / "issues" / "001-add-destination-adapter.md").read_text()
    assert "## Provenance" in issue_body
    assert "## Exact Quote" in issue_body
    github_script = (tmp_path / "out" / "create_github_issues.sh").read_text()
    assert "ALLOW_UNVERIFIED" in github_script


def test_publish_rejects_unverified_manifest(tmp_path: Path):
    from recording_issues.core import IssueCandidate, require_publishable_issues

    issue = IssueCandidate(
        title="Add destination adapter",
        summary="Support another issue destination.",
        source_path="/tmp/lesson.m4a",
        source_name="lesson.m4a",
        validation_status="missing_exact_quote",
    )

    try:
        require_publishable_issues([issue])
    except RuntimeError as error:
        assert "Refusing to publish" in str(error)
    else:
        raise AssertionError("Expected unverified issue to be rejected")


def test_publish_revalidates_manifest_against_cached_transcript(tmp_path: Path, monkeypatch):
    from recording_issues.core import (
        IssueCandidate,
        RecordingMedia,
        TranscriptSegment,
        create_github_issues,
        save_transcript,
        transcript_cache_path,
        write_outputs,
    )

    media_path = tmp_path / "lesson.m4a"
    media_path.write_text("x")
    media = RecordingMedia(path=media_path, kind="audio", dedupe_key="lesson")
    transcript = [
        TranscriptSegment(
            source_path=str(media_path),
            source_name=media_path.name,
            start_time=1.0,
            end_time=2.0,
            text="We should add verified publishing.",
        )
    ]
    cache = transcript_cache_path(tmp_path / "out", media_path, "openai-whisper-1-en")
    save_transcript(cache, media, transcript)
    issue = IssueCandidate(
        title="Add verified publishing",
        summary="Publish only issues with checked quotes.",
        source_path=str(media_path),
        source_name=media_path.name,
        exact_quote="We should add verified publishing.",
        validation_status="missing_exact_quote",
    )
    write_outputs(
        tmp_path / "out",
        [media],
        [],
        [issue],
        status={},
        config={
            "transcription_backend": "openai",
            "transcription_model": "whisper-1",
            "language": "en",
        },
    )
    monkeypatch.setattr(
        "recording_issues.core.find_existing_github_issue",
        lambda *_args, **_kwargs: "https://github.com/example/repo/issues/1",
    )

    assert create_github_issues(tmp_path / "out") == ["https://github.com/example/repo/issues/1"]


def test_publish_rejects_tampered_verified_manifest(tmp_path: Path):
    from recording_issues.core import (
        IssueCandidate,
        RecordingMedia,
        TranscriptSegment,
        create_github_issues,
        save_transcript,
        transcript_cache_path,
        write_outputs,
    )

    media_path = tmp_path / "lesson.m4a"
    media_path.write_text("x")
    media = RecordingMedia(path=media_path, kind="audio", dedupe_key="lesson")
    cache = transcript_cache_path(tmp_path / "out", media_path, "openai-whisper-1-en")
    save_transcript(
        cache,
        media,
        [
            TranscriptSegment(
                source_path=str(media_path),
                source_name=media_path.name,
                start_time=1.0,
                end_time=2.0,
                text="This transcript says something else.",
            )
        ],
    )
    issue = IssueCandidate(
        title="Add verified publishing",
        summary="Publish only issues with checked quotes.",
        source_path=str(media_path),
        source_name=media_path.name,
        exact_quote="We should add verified publishing.",
        validation_status="verified_exact_quote",
    )
    write_outputs(
        tmp_path / "out",
        [media],
        [],
        [issue],
        status={},
        config={
            "transcription_backend": "openai",
            "transcription_model": "whisper-1",
            "language": "en",
        },
    )

    try:
        create_github_issues(tmp_path / "out")
    except RuntimeError as error:
        assert "quote_not_found" in str(error)
    else:
        raise AssertionError("Expected tampered manifest to be rejected")


def test_cached_curation_reuses_json_response(tmp_path: Path, monkeypatch):
    from recording_issues import core
    from recording_issues.core import IssueCandidate

    calls = []

    def fake_complete_json(**_kwargs):
        calls.append(True)
        return {
            "issues": [
                {
                    "title": "Add reusable issue destination adapters",
                    "summary": "Merge destination-specific publishing work.",
                    "source_name": "Multiple sources",
                    "source_path": "Multiple sources",
                    "exact_quote": "We should add adapters.",
                    "provenance_kind": "explicit",
                    "labels": ["idea", "recording-capture", "integration"],
                    "acceptance_criteria": ["Adapters are documented."],
                }
            ]
        }

    monkeypatch.setattr(core, "complete_json", fake_complete_json)
    raw = [
        IssueCandidate(
            title=f"Add destination adapter {index}",
            summary="Support publishing captured issues.",
            source_name="lesson.m4a",
            source_path="/tmp/lesson.m4a",
        )
        for index in range(3)
    ]

    first = core.curate_issues_cached(raw, output_dir=tmp_path, model="gpt-test", context="test", target_count=1)
    second = core.curate_issues_cached(raw, output_dir=tmp_path, model="gpt-test", context="test", target_count=1)

    assert [issue.title for issue in first] == ["Add reusable issue destination adapters"]
    assert [issue.title for issue in second] == ["Add reusable issue destination adapters"]
    assert len(calls) == 1


def test_cli_inspect_prints_manifest_titles(tmp_path: Path, capsys):
    from recording_issues.cli import main

    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "issues": [
                    {
                        "title": "Add Notion publish",
                        "summary": "Publish captured issues to Notion.",
                        "source_path": "/tmp/lesson.m4a",
                        "source_name": "lesson.m4a",
                    }
                ]
            }
        )
    )

    assert main(["inspect", str(output)]) == 0
    assert "001. Add Notion publish" in capsys.readouterr().out
