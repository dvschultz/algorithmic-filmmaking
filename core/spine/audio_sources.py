"""Audio source spine impls.

Project-only audio source operations shared between the chat-tools agent and
the MCP server. The heavy ``ffmpeg`` / ``audio_formats`` imports are deferred
into ``import_audio_source`` so importing this module stays cheap.
"""

from __future__ import annotations


def list_audio_sources(project) -> dict:
    """List all imported audio sources in the project."""
    from core.analysis_availability import audio_transcription_is_complete

    audio_sources = []
    for a in project.audio_sources:
        audio_sources.append(
            {
                "id": a.id,
                "filename": a.filename,
                "duration": a.duration_seconds,
                "duration_str": a.duration_str,
                "sample_rate": a.sample_rate,
                "channels": a.channels,
                "transcribed": audio_transcription_is_complete(a),
                "transcript_segment_count": len(a.transcript) if a.transcript else 0,
            }
        )

    return {
        "success": True,
        "audio_sources": audio_sources,
        "count": len(audio_sources),
    }


def get_audio_source(project, audio_source_id: str) -> dict:
    """Return detailed information about a single audio source."""
    audio = project.get_audio_source(audio_source_id)
    if audio is None:
        return {
            "success": False,
            "error": (
                f"Audio source '{audio_source_id}' not found. "
                "Use list_audio_sources to see available IDs."
            ),
        }

    transcript_payload = None
    if audio.transcript is not None:
        transcript_payload = [seg.to_dict() for seg in audio.transcript]

    return {
        "success": True,
        "audio_source": {
            "id": audio.id,
            "filename": audio.filename,
            "file_path": str(audio.file_path),
            "duration": audio.duration_seconds,
            "duration_str": audio.duration_str,
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "transcript": transcript_payload,
        },
    }


def import_audio_source(project, file_path: str) -> dict:
    """Import an audio file into ``project``."""
    from pathlib import Path

    from core.operations.audio_import import (
        AudioImportTask,
        AudioImportApplication,
        run_audio_import,
    )

    path = Path(file_path).expanduser()
    if not path.is_absolute() and getattr(project, "path", None):
        path = project.path.parent / path
    task = AudioImportTask.from_path(path)
    application = AudioImportApplication(project, task)
    outcome = run_audio_import(task)
    if outcome.status != "succeeded":
        return {"success": False, "error": outcome.message or "Audio import failed"}
    application.apply(project, outcome)
    audio = application.audio
    if audio is None:
        return {"success": False, "error": "Audio import target changed"}
    return {
        "success": True,
        "audio_source_id": audio.id,
        "filename": audio.filename,
        "duration": audio.duration_seconds,
    }


__all__ = ["get_audio_source", "import_audio_source", "list_audio_sources"]
