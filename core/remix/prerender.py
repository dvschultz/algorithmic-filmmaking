"""Pre-render transformed clips via FFmpeg.

Bakes hflip/vflip/reverse into standalone video files so playback
and export just work without runtime filter juggling.
"""

import logging
import os
import subprocess
import concurrent.futures
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Optional

from core.binary_resolver import find_binary, get_subprocess_kwargs

logger = logging.getLogger(__name__)

# Maximum clip duration (seconds) for the reverse filter.
# reverse buffers the entire clip in RAM (~900 MB for 5s of 1080p30).
_REVERSE_MAX_DURATION = 15.0


def get_transform_cache_dir() -> Path:
    """Get the directory for cached pre-rendered clips."""
    from core.settings import load_settings
    settings = load_settings()
    return settings.thumbnail_cache_dir.parent / "transformed_clips"


def prerender_clip(
    source_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    hflip: bool,
    vflip: bool,
    reverse: bool,
    output_dir: Path,
    clip_id: str,
) -> Optional[Path]:
    """Pre-render a single clip with baked transforms.

    Args:
        source_path: Path to the source video file.
        start_frame: Clip start frame in source.
        end_frame: Clip end frame in source.
        fps: Source video frame rate.
        hflip: Apply horizontal flip.
        vflip: Apply vertical flip.
        reverse: Apply reverse playback.
        output_dir: Directory to write the output file.
        clip_id: Unique clip identifier for the filename.

    Returns:
        Path to the pre-rendered file, or None if no transforms needed.
    """
    if not (hflip or vflip or reverse):
        return None

    h = "1" if hflip else "0"
    v = "1" if vflip else "0"
    r = "1" if reverse else "0"
    output_path = output_dir / f"{clip_id}_{h}_{v}_{r}.mp4"

    # Idempotent: skip if file already exists and is non-empty
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = find_binary("ffmpeg")
    if not ffmpeg_path:
        logger.error("FFmpeg not found, cannot pre-render clip")
        return None

    start_seconds = start_frame / fps
    duration_seconds = (end_frame - start_frame) / fps

    # Check reverse safety limit
    apply_reverse = reverse
    if reverse and duration_seconds > _REVERSE_MAX_DURATION:
        logger.warning(
            "Skipping reverse for clip %s (%.1fs > %.0fs limit)",
            clip_id, duration_seconds, _REVERSE_MAX_DURATION,
        )
        apply_reverse = False

    # Build video filter chain
    vf_parts = []
    if hflip:
        vf_parts.append("hflip")
    if vflip:
        vf_parts.append("vflip")
    if apply_reverse:
        vf_parts.append("reverse")

    # If all requested transforms were skipped (e.g. reverse-only on long clip),
    # we still need to produce a file for consistency when other transforms exist.
    # But if literally nothing to apply, skip.
    if not vf_parts and not apply_reverse:
        return None

    cmd = [
        ffmpeg_path,
        "-y",
        "-ss", str(start_seconds),
        "-i", str(source_path),
        "-t", str(duration_seconds),
    ]

    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    if apply_reverse:
        cmd.extend(["-af", "areverse"])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        str(output_path),
    ])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            **get_subprocess_kwargs(),
        )
        if result.returncode != 0:
            logger.error(
                "FFmpeg pre-render failed for clip %s: %s",
                clip_id, result.stderr[-500:] if result.stderr else "unknown error",
            )
            return None
        return output_path
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg pre-render timed out for clip %s", clip_id)
        return None
    except Exception:
        logger.error("FFmpeg pre-render error for clip %s", clip_id, exc_info=True)
        return None


def _process_single_clip(
    clip,
    source,
    transforms: dict,
    output_dir: Path,
) -> Optional[Path]:
    """Process a single clip for use in the thread pool.

    Returns the pre-rendered path or None if no transforms needed.
    """
    hflip = transforms.get("hflip", False)
    vflip = transforms.get("vflip", False)
    reverse = transforms.get("reverse", False)

    if not (hflip or vflip or reverse):
        return None

    return prerender_clip(
        source_path=source.file_path,
        start_frame=clip.start_frame,
        end_frame=clip.end_frame,
        fps=source.fps,
        hflip=hflip,
        vflip=vflip,
        reverse=reverse,
        output_dir=output_dir,
        clip_id=clip.id,
    )


def prerender_batch(
    clips_with_transforms: list,
    output_dir: Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[Event] = None,
) -> list:
    """Pre-render a batch of clips with transforms.

    Runs up to 4 FFmpeg processes concurrently via a ThreadPoolExecutor.

    Args:
        clips_with_transforms: List of (Clip, Source, transforms_dict) where
            transforms_dict has keys 'hflip', 'vflip', 'reverse' (bool).
        output_dir: Directory for pre-rendered output files.
        progress_cb: Optional callback(current, total) for progress reporting.
        cancel_event: Optional threading.Event to check for cancellation.

    Returns:
        List of (Clip, Source, Optional[Path]) — the path is the pre-rendered
        file or None if no transforms were applied.
    """
    total = len(clips_with_transforms)
    if total == 0:
        return []

    # Report initial progress
    if progress_cb:
        progress_cb(0, total)

    # Pre-check cancellation before submitting any work
    if cancel_event and cancel_event.is_set():
        return []

    results: list = [None] * total
    max_workers = min(4, os.cpu_count() or 2)
    completed_count = 0
    counter_lock = Lock()

    def _on_done(idx: int, future: concurrent.futures.Future) -> None:
        """Callback invoked when a future completes; updates progress."""
        nonlocal completed_count
        with counter_lock:
            completed_count += 1
            current = completed_count
        if progress_cb:
            progress_cb(current, total)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[concurrent.futures.Future, int] = {}

        for i, (clip, source, transforms) in enumerate(clips_with_transforms):
            if cancel_event and cancel_event.is_set():
                break

            future = executor.submit(
                _process_single_clip, clip, source, transforms, output_dir,
            )
            future.add_done_callback(lambda f, idx=i: _on_done(idx, f))
            futures[future] = i

        # Collect results as they complete, checking for cancellation
        for future in concurrent.futures.as_completed(futures):
            if cancel_event and cancel_event.is_set():
                # Cancel remaining pending futures
                for f in futures:
                    f.cancel()
                break

            idx = futures[future]
            clip, source, transforms = clips_with_transforms[idx]
            try:
                prerendered = future.result()
            except Exception:
                logger.error(
                    "Unexpected error pre-rendering clip %s",
                    clip.id, exc_info=True,
                )
                prerendered = None
            results[idx] = (clip, source, prerendered)

    # Report final progress
    if progress_cb:
        progress_cb(total, total)

    # Filter out None slots left by cancellation
    return [r for r in results if r is not None]
