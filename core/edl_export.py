"""CMX 3600 adapter for validated render plans."""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.render_plan import compile_render_plan
from models.clip import Source
from models.frame import Frame
from models.media_time import VideoRange, frame_rate
from models.sequence import Sequence

logger = logging.getLogger(__name__)


@dataclass
class EDLExportConfig:
    output_path: Path
    title: str = "Scene Ripper Export"
    error_message: str | None = field(default=None, init=False)


def _sanitize_edl_string(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ")[:255]


def frames_to_timecode(frames: int, fps: float) -> str:
    """Number frames at the nominal rate for non-drop timecode."""
    nominal = round(frame_rate(fps))
    if nominal < 1 or frames < 0:
        raise ValueError("Invalid timecode frame or rate")
    seconds, remainder = divmod(frames, nominal)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{remainder:02d}"


def export_edl(
    sequence: Sequence,
    sources: dict[str, Source],
    config: EDLExportConfig,
    frames: dict[str, Frame] | None = None,
    *,
    clips: dict[str, tuple] | None = None,
) -> bool:
    """Atomically publish a supported EDL; expose failures on the config."""
    config.error_message = None
    staged = None
    try:
        plan = compile_render_plan(sequence, sources, clips, frames=frames)
        plan.validate_edl()
        lines = [f"TITLE: {_sanitize_edl_string(config.title)}", "FCM: NON-DROP FRAME", ""]
        edit = 0
        for segment in plan.segments:
            if segment.kind == "gap":
                continue
            assert isinstance(segment.media, VideoRange) and segment.path is not None
            edit += 1
            name = _sanitize_edl_string(segment.path.name)
            reel = _sanitize_edl_string(segment.path.stem)[:8].ljust(8)
            source_rate = float(segment.media.rate)
            src_in = frames_to_timecode(segment.media.start_frame, source_rate)
            src_out = frames_to_timecode(segment.media.end_frame, source_rate)
            rec_in = frames_to_timecode(segment.start_frame, float(plan.output_rate))
            rec_out = frames_to_timecode(segment.end_frame, float(plan.output_rate))
            lines.extend([
                f"{edit:03d}  {reel} V     C        {src_in} {src_out} {rec_in} {rec_out}",
                f"* FROM CLIP NAME: {name}",
                f"* SOURCE FILE: {_sanitize_edl_string(str(segment.path))}", "",
            ])
        output = config.output_path
        if output.suffix.lower() != ".edl":
            output = output.with_suffix(".edl")
        if any(segment.path is not None and segment.path.resolve() == output.resolve() for segment in plan.segments):
            raise ValueError("EDL output must not overwrite source media")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".edl_", delete=False) as handle:
            staged = Path(handle.name)
            handle.write("\n".join(lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, output)
        return True
    except (ValueError, OSError) as exc:
        config.error_message = str(exc)
        logger.error("EDL export failed: %s", exc)
        return False
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
