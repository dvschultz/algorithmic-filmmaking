"""Clip-level spine impls.

Read-only clip queries shared between the chat-tools agent and the MCP
server. Heavy imports (``numpy``, color helpers) are lazy.
"""

from __future__ import annotations

from typing import Optional

from core.spine._agent_formatting import ASPECT_RATIO_RANGES, append_gaze_fields


def filter_clips(
    project,
    shot_type: Optional[str] = None,
    has_speech: Optional[bool] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    aspect_ratio: Optional[str] = None,
    search_query: Optional[str] = None,
    has_object: Optional[str] = None,
    min_people: Optional[int] = None,
    max_people: Optional[int] = None,
    search_description: Optional[str] = None,
    has_faces: Optional[bool] = None,
    gaze_category: Optional[str] = None,
    min_brightness: Optional[float] = None,
    max_brightness: Optional[float] = None,
    search_ocr_text: Optional[str] = None,
    min_volume: Optional[float] = None,
    max_volume: Optional[float] = None,
    search_tags: Optional[str] = None,
    search_notes: Optional[str] = None,
    cinematography_shot_size: Optional[str] = None,
    cinematography_camera_angle: Optional[str] = None,
    cinematography_camera_movement: Optional[str] = None,
    cinematography_lighting_style: Optional[str] = None,
    cinematography_subject_count: Optional[str] = None,
    cinematography_emotional_intensity: Optional[str] = None,
    cinematography_suggested_pacing: Optional[str] = None,
    similar_to_clip_id: Optional[str] = None,
):
    """Filter clips by metadata, content, and analysis fields."""
    import numpy as np

    anchor_embedding = None
    if similar_to_clip_id:
        anchor_clip = project.clips_by_id.get(similar_to_clip_id)
        if not anchor_clip:
            return []
        emb = anchor_clip.embedding
        if emb is not None:
            anchor_arr = np.array(emb, dtype=np.float64)
            if np.linalg.norm(anchor_arr) > 0:
                anchor_embedding = anchor_arr

    results = []

    for clip in project.clips:
        source = project.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0
        duration = (clip.end_frame - clip.start_frame) / fps

        if shot_type and getattr(clip, "shot_type", None) != shot_type:
            continue

        if has_speech is not None:
            clip_has_speech = bool(getattr(clip, "transcript", None))
            if clip_has_speech != has_speech:
                continue

        if min_duration is not None and duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue

        if aspect_ratio and aspect_ratio in ASPECT_RATIO_RANGES:
            if not source or source.width == 0 or source.height == 0:
                continue
            source_aspect = source.width / source.height
            min_ratio, max_ratio = ASPECT_RATIO_RANGES[aspect_ratio]
            if not (min_ratio <= source_aspect <= max_ratio):
                continue

        if search_query:
            transcript_text = clip.get_transcript_text()
            if not transcript_text:
                continue
            if search_query.lower() not in transcript_text.lower():
                continue

        if search_description:
            description = getattr(clip, "description", None)
            if not description:
                continue
            if search_description.lower() not in description.lower():
                continue

        if has_object is not None:
            object_labels = getattr(clip, "object_labels", None) or []
            detected_objects = getattr(clip, "detected_objects", None) or []
            detected_labels = [d.get("label", "") for d in detected_objects]
            all_labels = object_labels + detected_labels
            search_lower = has_object.lower()
            if not any(search_lower in label.lower() for label in all_labels):
                continue

        if min_people is not None:
            person_count = getattr(clip, "person_count", None) or 0
            if person_count < min_people:
                continue

        if max_people is not None:
            person_count = getattr(clip, "person_count", None) or 0
            if person_count > max_people:
                continue

        if has_faces is not None:
            clip_has_faces = bool(clip.face_embeddings)
            if clip_has_faces != has_faces:
                continue

        if gaze_category and getattr(clip, "gaze_category", None) != gaze_category:
            continue

        if min_brightness is not None:
            brightness = getattr(clip, "average_brightness", None)
            if brightness is None or brightness < min_brightness:
                continue
        if max_brightness is not None:
            brightness = getattr(clip, "average_brightness", None)
            if brightness is None or brightness > max_brightness:
                continue

        if min_volume is not None:
            volume = getattr(clip, "rms_volume", None)
            if volume is None or volume < min_volume:
                continue
        if max_volume is not None:
            volume = getattr(clip, "rms_volume", None)
            if volume is None or volume > max_volume:
                continue

        if search_ocr_text:
            combined = clip.combined_text
            if not combined:
                continue
            if search_ocr_text.lower() not in combined.lower():
                continue

        if search_tags:
            clip_tags = getattr(clip, "tags", None) or []
            if not clip_tags:
                continue
            tags_text = " ".join(clip_tags)
            if search_tags.lower() not in tags_text.lower():
                continue

        if search_notes:
            clip_notes = getattr(clip, "notes", None) or ""
            if not clip_notes:
                continue
            if search_notes.lower() not in clip_notes.lower():
                continue

        if cinematography_shot_size:
            cine = getattr(clip, "cinematography", None)
            if not cine or cine.shot_size != cinematography_shot_size:
                continue
        if cinematography_camera_angle:
            cine = getattr(clip, "cinematography", None)
            if not cine or cine.camera_angle != cinematography_camera_angle:
                continue
        if cinematography_camera_movement:
            cine = getattr(clip, "cinematography", None)
            if not cine or cine.camera_movement != cinematography_camera_movement:
                continue
        if cinematography_lighting_style:
            cine = getattr(clip, "cinematography", None)
            if not cine or cine.lighting_style != cinematography_lighting_style:
                continue
        if cinematography_subject_count:
            cine = getattr(clip, "cinematography", None)
            if not cine or cine.subject_count != cinematography_subject_count:
                continue
        if cinematography_emotional_intensity:
            cine = getattr(clip, "cinematography", None)
            if (
                not cine
                or cine.emotional_intensity != cinematography_emotional_intensity
            ):
                continue
        if cinematography_suggested_pacing:
            cine = getattr(clip, "cinematography", None)
            if not cine or cine.suggested_pacing != cinematography_suggested_pacing:
                continue

        clip_aspect_ratio = None
        if source and source.height > 0:
            clip_aspect_ratio = round(source.width / source.height, 3)

        cine_data = None
        cine = getattr(clip, "cinematography", None)
        if cine:
            cine_data = {
                "shot_size": cine.shot_size,
                "camera_angle": cine.camera_angle,
                "camera_movement": cine.camera_movement,
                "lighting_style": cine.lighting_style,
                "subject_count": cine.subject_count,
                "emotional_intensity": cine.emotional_intensity,
                "suggested_pacing": cine.suggested_pacing,
            }

        clip_data = {
            "id": clip.id,
            "source_id": clip.source_id,
            "source_name": source.file_path.name if source else "Unknown",
            "duration_seconds": round(duration, 2),
            "shot_type": getattr(clip, "shot_type", None),
            "has_speech": bool(getattr(clip, "transcript", None)),
            "dominant_colors": getattr(clip, "dominant_colors", None),
            "object_labels": getattr(clip, "object_labels", None),
            "person_count": getattr(clip, "person_count", None),
            "description": getattr(clip, "description", None),
            "width": source.width if source else None,
            "height": source.height if source else None,
            "aspect_ratio": clip_aspect_ratio,
            "average_brightness": clip.average_brightness,
            "rms_volume": clip.rms_volume,
            "tags": clip.tags or [],
            "notes": clip.notes or "",
            "extracted_text": clip.combined_text,
            "cinematography": cine_data,
        }
        append_gaze_fields(clip, clip_data)
        results.append(clip_data)

    if similar_to_clip_id and anchor_embedding is None:
        return {
            "success": False,
            "error": (
                f"Anchor clip '{similar_to_clip_id}' has no valid embedding. "
                "Run embedding analysis first (analyze_clips with 'embeddings')."
            ),
        }

    if similar_to_clip_id and anchor_embedding is not None:
        scored_results = []
        for clip_data in results:
            clip_obj = project.clips_by_id.get(clip_data["id"])
            emb = clip_obj.embedding if clip_obj else None
            if emb is not None:
                clip_arr = np.array(emb, dtype=np.float64)
                if clip_arr.shape != anchor_embedding.shape:
                    continue
                norm = np.linalg.norm(clip_arr)
                if norm > 0:
                    score = float(np.dot(anchor_embedding, clip_arr))
                    clip_data["similarity_score"] = round(score, 4)
                    scored_results.append(clip_data)
        scored_results.sort(
            key=lambda d: d.get("similarity_score", 0.0), reverse=True
        )
        return scored_results

    return results


def find_similar_clips(
    project,
    clip_id: str,
    criteria: Optional[list[str]] = None,
    limit: int = 10,
) -> dict:
    """Rank clips by similarity to a reference clip."""
    from core.analysis.color import get_primary_hue

    if criteria is None:
        criteria = ["color", "shot_type"]

    valid_criteria = ["color", "shot_type", "duration"]
    for c in criteria:
        if c not in valid_criteria:
            return {
                "success": False,
                "error": (
                    f"Invalid criterion '{c}'. "
                    f"Valid criteria: {', '.join(valid_criteria)}"
                ),
            }

    reference = project.clips_by_id.get(clip_id)
    if not reference:
        return {"success": False, "error": f"Clip '{clip_id}' not found"}

    ref_source = project.sources_by_id.get(reference.source_id)
    ref_fps = ref_source.fps if ref_source else 30.0
    ref_duration = reference.duration_seconds(ref_fps)

    scores = []
    for clip in project.clips:
        if clip.id == clip_id:
            continue

        score = 0.0
        source = project.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0

        if "shot_type" in criteria:
            if clip.shot_type and reference.shot_type:
                if clip.shot_type == reference.shot_type:
                    score += 1.0
                elif clip.shot_type and reference.shot_type:
                    close_types = {
                        "close_up": ["extreme_close_up", "medium_close_up"],
                        "medium_shot": ["medium_close_up", "medium_long_shot"],
                        "wide_shot": ["medium_long_shot", "extreme_wide_shot"],
                    }
                    if reference.shot_type in close_types:
                        if clip.shot_type in close_types[reference.shot_type]:
                            score += 0.5

        if "color" in criteria and reference.dominant_colors and clip.dominant_colors:
            ref_hue = get_primary_hue(reference.dominant_colors)
            clip_hue = get_primary_hue(clip.dominant_colors)
            hue_diff = abs(ref_hue - clip_hue)
            if hue_diff > 180:
                hue_diff = 360 - hue_diff
            score += max(0, 1.0 - hue_diff / 60)

        if "duration" in criteria:
            clip_duration = clip.duration_seconds(fps)
            if ref_duration > 0 and clip_duration > 0:
                duration_ratio = (
                    min(ref_duration, clip_duration)
                    / max(ref_duration, clip_duration)
                )
                score += duration_ratio

        if score > 0:
            scores.append((clip, score, source))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for clip, score, source in scores[:limit]:
        fps = source.fps if source else 30.0
        results.append(
            {
                "clip_id": clip.id,
                "source_name": source.file_path.name if source else "Unknown",
                "similarity_score": round(score, 2),
                "shot_type": clip.shot_type,
                "duration_seconds": round(clip.duration_seconds(fps), 2),
                "has_speech": bool(clip.transcript),
            }
        )

    return {
        "success": True,
        "reference_clip_id": clip_id,
        "criteria": criteria,
        "similar_clips": results,
    }


def group_clips_by(project, criterion: str) -> dict:
    """Group clips by a single criterion."""
    from core.analysis.color import classify_color_palette

    valid_criteria = ["color", "shot_type", "duration", "source"]
    if criterion not in valid_criteria:
        return {
            "success": False,
            "error": (
                f"Invalid criterion '{criterion}'. "
                f"Valid criteria: {', '.join(valid_criteria)}"
            ),
        }

    groups: dict[str, list[str]] = {}

    for clip in project.clips:
        source = project.sources_by_id.get(clip.source_id)
        fps = source.fps if source else 30.0

        if criterion == "shot_type":
            key = clip.shot_type if clip.shot_type else "unknown"
        elif criterion == "color":
            if clip.dominant_colors:
                key = classify_color_palette(clip.dominant_colors)
            else:
                key = "unanalyzed"
        elif criterion == "duration":
            duration = clip.duration_seconds(fps)
            if duration < 2:
                key = "short (<2s)"
            elif duration < 10:
                key = "medium (2-10s)"
            else:
                key = "long (>10s)"
        elif criterion == "source":
            key = source.file_path.name if source else "unknown"
        else:
            key = "unknown"

        if key not in groups:
            groups[key] = []
        groups[key].append(clip.id)

    formatted_groups = {
        k: {"clip_ids": v, "count": len(v)}
        for k, v in sorted(groups.items())
    }

    return {
        "success": True,
        "criterion": criterion,
        "group_count": len(groups),
        "groups": formatted_groups,
    }


def get_clip_cinematography(project, clip_id: str) -> dict:
    """Return the full cinematography analysis for one clip."""
    clip = project.clips_by_id.get(clip_id)
    if clip is None:
        return {
            "success": False,
            "error": (
                f"Clip '{clip_id}' not found. "
                "Use list_clips to see available clips."
            ),
        }

    cinematography = getattr(clip, "cinematography", None)
    if cinematography is None:
        return {
            "success": False,
            "error": (
                f"Clip '{clip_id}' has no cinematography analysis. "
                "Run cinematography analysis first via the Analyze tab."
            ),
        }

    return {
        "success": True,
        "clip_id": clip_id,
        "cinematography": cinematography.to_dict(),
    }


__all__ = [
    "filter_clips",
    "find_similar_clips",
    "get_clip_cinematography",
    "group_clips_by",
]
