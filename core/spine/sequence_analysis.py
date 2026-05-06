"""Sequence-analysis spine impls.

Audio analysis, beat alignment, continuity checks, and report generation —
all read-only operations on a ``Project`` instance, callable from any
GUI-free environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.spine._agent_formatting import summarize_report_for_agent


def detect_audio_beats(audio_path: str, include_onsets: bool = True) -> dict:
    """Analyze an audio or video file for beats, tempo, and onsets."""
    from core.analysis.audio import (
        analyze_audio_from_video,
        analyze_music_file,
        has_audio_track,
    )

    path = Path(audio_path)

    if not path.exists():
        return {"success": False, "error": f"File not found: {audio_path}"}

    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    try:
        if path.suffix.lower() in audio_extensions:
            analysis = analyze_music_file(path, include_onsets=include_onsets)
        elif path.suffix.lower() in video_extensions:
            if not has_audio_track(path):
                return {
                    "success": False,
                    "error": f"Video file has no audio track: {audio_path}",
                }
            analysis = analyze_audio_from_video(path, include_onsets=include_onsets)
        else:
            return {
                "success": False,
                "error": (
                    f"Unsupported file type: {path.suffix}. "
                    f"Supported: {audio_extensions | video_extensions}"
                ),
            }

        return {
            "success": True,
            "file": str(path),
            "tempo_bpm": round(analysis.tempo_bpm, 1),
            "beat_count": len(analysis.beat_times),
            "beat_times": [round(t, 3) for t in analysis.beat_times[:20]],
            "beat_times_truncated": len(analysis.beat_times) > 20,
            "downbeat_count": len(analysis.downbeat_times),
            "downbeat_times": [round(t, 3) for t in analysis.downbeat_times[:10]],
            "onset_count": len(analysis.onset_times) if include_onsets else 0,
            "onset_times": (
                [round(t, 3) for t in analysis.onset_times[:20]]
                if include_onsets
                else []
            ),
            "duration_seconds": round(analysis.duration_seconds, 2),
            "message": (
                f"Detected {analysis.tempo_bpm:.1f} BPM with "
                f"{len(analysis.beat_times)} beats over "
                f"{analysis.duration_seconds:.1f}s"
            ),
        }

    except Exception as e:
        return {"success": False, "error": f"Audio analysis failed: {str(e)}"}


def align_sequence_to_audio(
    project,
    audio_path: str,
    strategy: str = "nearest",
    max_adjustment: float = 0.5,
) -> dict:
    """Suggest beat-aligned cut points for the current sequence."""
    from core.analysis.audio import (
        analyze_audio_from_video,
        analyze_music_file,
        has_audio_track,
    )
    from core.remix.audio_sync import suggest_beat_aligned_cuts

    valid_strategies = ("nearest", "downbeat", "onset")
    if strategy not in valid_strategies:
        return {
            "success": False,
            "error": f"Invalid strategy '{strategy}'. Use: {valid_strategies}",
        }

    if not project.sequence or not project.sequence.clips:
        return {
            "success": False,
            "error": "No clips in sequence. Add clips to the sequence first.",
        }

    path = Path(audio_path)
    if not path.exists():
        return {"success": False, "error": f"Audio file not found: {audio_path}"}

    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    try:
        if path.suffix.lower() in audio_extensions:
            audio_analysis = analyze_music_file(path)
        elif path.suffix.lower() in video_extensions:
            if not has_audio_track(path):
                return {
                    "success": False,
                    "error": f"Video file has no audio track: {audio_path}",
                }
            audio_analysis = analyze_audio_from_video(path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {path.suffix}",
            }

        clip_end_times = []
        current_time = 0.0

        for seq_clip in project.sequence.clips:
            source_clip = project.clips_by_id.get(seq_clip.source_clip_id)
            source = project.sources_by_id.get(seq_clip.source_id)

            if source_clip and source:
                duration = source_clip.duration_seconds(source.fps)
                current_time += duration
                clip_end_times.append((seq_clip.id, current_time))

        if not clip_end_times:
            return {
                "success": False,
                "error": "Could not calculate clip durations. "
                "Check source clips exist.",
            }

        suggestions = suggest_beat_aligned_cuts(
            clip_end_times=clip_end_times,
            audio_analysis=audio_analysis,
            strategy=strategy,
            max_adjustment=max_adjustment,
        )

        return {
            "success": True,
            "audio_file": str(path),
            "tempo_bpm": round(audio_analysis.tempo_bpm, 1),
            "strategy": strategy,
            "max_adjustment": max_adjustment,
            "sequence_clip_count": len(project.sequence.clips),
            "suggestions_count": len(suggestions),
            "suggestions": [s.to_dict() for s in suggestions],
            "message": (
                f"Found {len(suggestions)} clips that could be adjusted to align with "
                f"{audio_analysis.tempo_bpm:.1f} BPM beats (strategy: {strategy})"
            ),
        }

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Alignment analysis failed: {str(e)}"}


def get_sequence_analysis(
    project, genre_comparison: Optional[str] = None
) -> dict:
    """Analyze the current sequence for pacing, continuity, and consistency."""
    from core.analysis.sequence import analyze_sequence, get_pacing_curve
    from models.sequence_analysis import GENRE_PACING_NORMS

    if not project.sequence:
        return {
            "success": False,
            "error": "No sequence exists. Create a sequence first.",
        }

    all_clips = project.sequence.get_all_clips()
    if not all_clips:
        return {
            "success": False,
            "error": "Sequence is empty. Add clips to the sequence first.",
        }

    try:
        analysis = analyze_sequence(project.sequence, project)

        result = {
            "success": True,
            "sequence_name": project.sequence.name,
            "clip_count": analysis.pacing.clip_count,
            "pacing": analysis.pacing.to_dict(),
            "visual_consistency": analysis.visual_consistency.to_dict(),
            "continuity_warning_count": len(analysis.continuity_warnings),
            "continuity_warnings": [
                w.to_dict() for w in analysis.continuity_warnings
            ],
            "suggestions": analysis.suggestions,
        }

        if genre_comparison:
            genre_lower = genre_comparison.lower()
            if genre_lower in GENRE_PACING_NORMS:
                comparison = analysis.compare_to_genre(genre_lower)
                if comparison:
                    result["genre_comparison"] = comparison.to_dict()
            else:
                result["genre_comparison_error"] = (
                    f"Unknown genre '{genre_comparison}'. "
                    f"Valid genres: {list(GENRE_PACING_NORMS.keys())}"
                )

        result["pacing_curve"] = get_pacing_curve(project.sequence, project)
        return result

    except Exception as e:
        return {"success": False, "error": f"Sequence analysis failed: {str(e)}"}


def check_continuity_issues(project) -> dict:
    """Run continuity checks on the current sequence."""
    from core.analysis.sequence import _resolve_source_clips, check_continuity

    if not project.sequence:
        return {
            "success": False,
            "error": "No sequence exists. Create a sequence first.",
        }

    all_clips = project.sequence.get_all_clips()
    if len(all_clips) < 2:
        return {
            "success": True,
            "message": "Need at least 2 clips to check continuity",
            "warning_count": 0,
            "warnings": [],
        }

    try:
        resolved = _resolve_source_clips(all_clips, project)
        warnings = check_continuity(resolved)

        by_severity = {"low": 0, "medium": 0, "high": 0}
        for w in warnings:
            by_severity[w.severity] = by_severity.get(w.severity, 0) + 1

        return {
            "success": True,
            "sequence_name": project.sequence.name,
            "clip_count": len(all_clips),
            "warning_count": len(warnings),
            "warnings_by_severity": by_severity,
            "warnings": [w.to_dict() for w in warnings],
            "message": (
                (
                    f"Found {len(warnings)} potential continuity issues "
                    f"({by_severity['high']} high, {by_severity['medium']} medium, "
                    f"{by_severity['low']} low)"
                )
                if warnings
                else "No continuity issues detected"
            ),
        }

    except Exception as e:
        return {"success": False, "error": f"Continuity check failed: {str(e)}"}


def generate_analysis_report(
    project,
    sections: Optional[list[str]] = None,
    include_clip_details: bool = False,
    output_format: str = "markdown",
    clip_ids: Optional[list[str]] = None,
) -> dict:
    """Generate a film analysis report."""
    from core.scene_report import (
        DEFAULT_SECTIONS,
        REPORT_SECTIONS,
        generate_clips_report,
        generate_sequence_report,
        report_to_html,
    )

    valid_sections = list(REPORT_SECTIONS.keys())
    if sections:
        invalid = [s for s in sections if s not in valid_sections]
        if invalid:
            return {
                "success": False,
                "error": f"Invalid sections: {invalid}. Valid sections: {valid_sections}",
            }
    else:
        sections = DEFAULT_SECTIONS

    if output_format not in ("markdown", "html"):
        return {
            "success": False,
            "error": (
                f"Invalid output_format: {output_format}. Use 'markdown' or 'html'"
            ),
        }

    try:
        if clip_ids:
            clips = [project.clips_by_id.get(cid) for cid in clip_ids]
            clips = [c for c in clips if c is not None]
            if not clips:
                return {
                    "success": False,
                    "error": "No valid clips found for the provided IDs",
                }
            report = generate_clips_report(
                clips, project, title="Selected Clips Analysis"
            )
        else:
            if not project.sequence:
                return {
                    "success": False,
                    "error": "No sequence exists. Create a sequence first.",
                }
            if not project.sequence.get_all_clips():
                return {
                    "success": False,
                    "error": "Sequence is empty. Add clips to the sequence first.",
                }
            report = generate_sequence_report(
                project.sequence,
                project,
                sections=sections,
                include_clip_details=include_clip_details,
            )

        if output_format == "html":
            report = report_to_html(report)

        report_summary = summarize_report_for_agent(report, sections, output_format)
        result = {
            "success": True,
            "format": output_format,
            "sections_included": sections,
            "word_count": report_summary["word_count"],
            "report_summary": report_summary,
            "message": (
                f"Generated {output_format} report with {len(sections)} sections"
            ),
        }
        if not report_summary["is_truncated"]:
            result["report"] = report
        return result

    except Exception as e:
        return {"success": False, "error": f"Report generation failed: {str(e)}"}


__all__ = [
    "align_sequence_to_audio",
    "check_continuity_issues",
    "detect_audio_beats",
    "generate_analysis_report",
    "get_sequence_analysis",
]
