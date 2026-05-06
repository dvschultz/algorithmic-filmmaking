"""Audio source spine impls.

Project-only audio source operations shared between the chat-tools agent and
the MCP server. The heavy ``ffmpeg`` / ``audio_formats`` imports are deferred
into ``import_audio_source`` so importing this module stays cheap.
"""

from __future__ import annotations


def list_audio_sources(project) -> dict:
    """List all imported audio sources in the project."""
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
                "transcribed": bool(a.transcript),
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
    if audio.transcript:
        transcript_payload = [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            for seg in audio.transcript
        ]

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

    from core.audio_formats import is_audio_file
    from core.ffmpeg import FFmpegProcessor
    from models.audio_source import AudioSource

    path = Path(file_path).expanduser()
    if not path.is_absolute() and getattr(project, "path", None):
        path = project.path.parent / path
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}
    if not is_audio_file(path):
        return {
            "success": False,
            "error": (
                f"Unsupported audio format: {path.suffix or '<no extension>'}. "
                "Supported: .mp3, .wav, .flac, .m4a, .aac, .ogg."
            ),
        }

    try:
        processor = FFmpegProcessor()
    except RuntimeError as exc:
        return {"success": False, "error": f"FFmpeg unavailable: {exc}"}

    if not processor.ffprobe_available:
        return {"success": False, "error": "FFprobe is not available"}

    try:
        info = processor.get_audio_info(path)
    except ValueError:
        return {"success": False, "error": f"Not an audio file: {path.name}"}
    except RuntimeError as exc:
        return {"success": False, "error": f"Failed to probe audio: {exc}"}

    duration = info.get("duration", 0.0)
    if duration <= 0:
        return {
            "success": False,
            "error": f"Audio file has zero duration: {path.name}",
        }

    audio = AudioSource(
        file_path=path,
        duration_seconds=duration,
        sample_rate=info.get("sample_rate", 0),
        channels=info.get("channels", 0),
    )
    project.add_audio_source(audio)

    return {
        "success": True,
        "audio_source_id": audio.id,
        "filename": audio.filename,
        "duration": audio.duration_seconds,
    }


__all__ = ["get_audio_source", "import_audio_source", "list_audio_sources"]
