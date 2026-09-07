"""Detached OCR computation and owner-bound publication for clips and frames."""

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from typing import Callable, Literal, TYPE_CHECKING

from core.jobs.media import media_stamp
from core.operations.contracts import OutcomeStatus

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, ExtractedText, Source
    from models.frame import Frame

_inference_lock = Lock()


@dataclass(frozen=True)
class OcrTask:
    clip_id: str
    path: Path | None
    target_type: Literal["clip", "frame"] = "clip"
    start_frame: int = 0
    end_frame: int = 0
    fps: float = 0.0
    skip: bool = False

    @classmethod
    def from_clip(
        cls, clip: "Clip", source: "Source | None", *, skip: bool = False
    ) -> "OcrTask":
        return cls(
            clip.id,
            source.file_path if source else None,
            "clip",
            clip.start_frame,
            clip.end_frame,
            source.fps if source else 0.0,
            skip,
        )

    @property
    def key(self) -> tuple[str, str]:
        return self.target_type, self.clip_id


@dataclass(frozen=True)
class OcrOptions:
    num_keyframes: int = 3
    use_vlm_fallback: bool = True
    vlm_model: str | None = None
    vlm_only: bool = False
    use_text_detection: bool = True


@dataclass(frozen=True)
class OcrText:
    frame_number: int
    text: str
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if (
            type(self.frame_number) is not int
            or self.frame_number < 0
            or not isinstance(self.text, str)
            or not self.text.strip()
            or isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
            or self.source not in ("paddleocr", "tesseract", "vlm")
        ):
            raise ValueError("Invalid OCR text observation")

    def to_model(self) -> "ExtractedText":
        from models.clip import ExtractedText

        return ExtractedText(self.frame_number, self.text, self.confidence, self.source)


@dataclass(frozen=True)
class OcrOutcome:
    clip_id: str
    status: OutcomeStatus
    target_type: Literal["clip", "frame"] = "clip"
    texts: tuple[OcrText, ...] = ()
    code: str | None = None
    message: str | None = None

    def to_models(self) -> list["ExtractedText"]:
        return [text.to_model() for text in self.texts]

    @classmethod
    def from_dict(cls, data: dict) -> "OcrOutcome":
        if data["target_type"] not in ("clip", "frame") or data["status"] not in (
            "succeeded",
            "failed",
            "skipped",
            "unprocessed",
        ):
            raise ValueError("Invalid OCR outcome")
        return cls(
            **{
                **data,
                "texts": tuple(OcrText(**value) for value in data.get("texts", ())),
            }
        )


def run_ocr(
    tasks: tuple[OcrTask, ...],
    options: OcrOptions,
    *,
    cancel_event: Event | None = None,
    on_outcome: Callable[[OcrOutcome], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[OcrOutcome, ...]:
    """Run one provider loop with immutable inputs and serialized model access."""
    cancel = cancel_event or Event()
    stamps = {task.key: media_stamp(task.path) if task.path else None for task in tasks}
    outcomes = []
    model_failed = False
    for index, task in enumerate(tasks):

        def result(status: OutcomeStatus, **kwargs) -> OcrOutcome:
            return OcrOutcome(task.clip_id, status, task.target_type, **kwargs)

        acquired = False
        try:
            if cancel.is_set():
                outcome = result("unprocessed", code="cancelled")
            elif task.skip:
                outcome = result("skipped", code="already_populated")
            elif model_failed:
                outcome = result("unprocessed", code="model_unavailable")
            elif task.path is None or not task.path.is_file():
                outcome = result(
                    "failed",
                    code="source_file_missing"
                    if task.target_type == "clip"
                    else "image_missing",
                )
            else:
                if progress:
                    progress(index + 1, len(tasks), task.clip_id)
                while not cancel.is_set() and not acquired:
                    acquired = _inference_lock.acquire(timeout=0.05)
                if cancel.is_set():
                    outcome = result("unprocessed", code="cancelled")
                else:
                    if media_stamp(task.path) != stamps[task.key]:
                        raise ValueError("OCR input media changed")
                    from core.analysis.ocr import (
                        extract_text_from_clip,
                        extract_text_from_frame,
                    )

                    if task.target_type == "clip":
                        if (
                            type(task.start_frame) is not int
                            or type(task.end_frame) is not int
                            or task.start_frame < 0
                            or task.end_frame <= task.start_frame
                            or not isfinite(task.fps)
                            or task.fps <= 0
                        ):
                            raise ValueError("Invalid OCR source range")
                        raw = extract_text_from_clip(
                            clip=SimpleNamespace(
                                id=task.clip_id,
                                start_frame=task.start_frame,
                                end_frame=task.end_frame,
                            ),
                            source=SimpleNamespace(file_path=task.path, fps=task.fps),
                            num_keyframes=options.num_keyframes,
                            use_text_detection=options.use_text_detection,
                            use_vlm_fallback=options.use_vlm_fallback,
                            vlm_model=options.vlm_model,
                            vlm_only=options.vlm_only,
                            cancel_event=cancel,
                            raise_errors=True,
                        )
                        texts = tuple(
                            OcrText(
                                item.frame_number,
                                item.text,
                                item.confidence,
                                item.source,
                            )
                            for item in raw
                        )
                        if any(
                            not task.start_frame <= text.frame_number < task.end_frame
                            for text in texts
                        ):
                            raise ValueError("OCR returned text outside the clip")
                    elif task.target_type == "frame":
                        text, confidence, method = extract_text_from_frame(
                            frame_path=task.path,
                            skip_detection=not options.use_text_detection,
                            use_vlm_fallback=options.use_vlm_fallback,
                            vlm_model=options.vlm_model,
                            vlm_only=options.vlm_only,
                            cancel_event=cancel,
                            raise_errors=True,
                        )
                        if (
                            not isinstance(text, str)
                            or isinstance(confidence, bool)
                            or not isinstance(confidence, (int, float))
                            or not isfinite(confidence)
                            or not 0 <= confidence <= 1
                            or method not in ("paddleocr", "tesseract", "vlm", "none")
                        ):
                            raise ValueError("Invalid OCR frame observation")
                        texts = (
                            (OcrText(0, text, confidence, method),)
                            if text and text.strip()
                            else ()
                        )
                    else:
                        raise ValueError("Invalid OCR target type")
                    if media_stamp(task.path) != stamps[task.key]:
                        raise ValueError("OCR input media changed")
                    outcome = result("succeeded", texts=texts)
        except Exception as exc:
            from core.errors import ModelDownloadError

            model_failed = isinstance(exc, ModelDownloadError)
            outcome = result(
                "failed",
                code="model_load_failed" if model_failed else "text_extraction_failed",
                message=str(exc),
            )
        finally:
            if acquired:
                _inference_lock.release()
        if cancel.is_set():
            outcome = result("unprocessed", code="cancelled")
        outcomes.append(outcome)
        if not cancel.is_set() and on_outcome:
            on_outcome(outcome)
    return tuple(outcomes)


class OcrApplication:
    """Apply each observation once to its original, unchanged target."""

    def __init__(self, project: "Project", tasks: tuple[OcrTask, ...]) -> None:
        project.session.assert_owner()
        self.project = project
        self.session_id = project.session.session_id
        self.tasks = {task.key: task for task in tasks}
        self.bindings = {task.key: self._binding(project, task) for task in tasks}
        self.consumed: set[tuple[str, str]] = set()

    def inputs_current(self, project: "Project") -> bool:
        """Validate an uncommitted proposal without publishing analysis."""
        project.session.assert_owner()
        if project is not self.project or project.session.session_id != self.session_id:
            return False
        for key, task in self.tasks.items():
            expected = self.bindings[key]
            current = self._binding(project, task)
            if (
                expected is None
                or current is None
                or current[0] is not expected[0]
                or current[1] is not expected[1]
                or current[2] != expected[2]
            ):
                return False
        return True

    @staticmethod
    def _binding(project: "Project", task: OcrTask) -> tuple | None:
        source = None
        target: "Clip | Frame | None"
        identity: tuple
        if task.target_type == "frame":
            target = project.frames_by_id.get(task.clip_id)
            if target is None or target.file_path != task.path:
                return None
            identity = (
                target.file_path,
                target.source_id,
                target.clip_id,
                target.frame_number,
            )
        else:
            target = project.clips_by_id.get(task.clip_id)
            if target is None:
                return None
            source = project.sources_by_id.get(target.source_id)
            if source is None or (
                source.file_path,
                target.start_frame,
                target.end_frame,
                source.fps,
            ) != (task.path, task.start_frame, task.end_frame, task.fps):
                return None
            identity = (
                target.source_id,
                source.file_path,
                target.start_frame,
                target.end_frame,
                source.fps,
            )
        stamp = media_stamp(task.path) if task.path else None
        if stamp is None:
            return None
        return target, source, (identity, stamp, deepcopy(target.extracted_texts))

    def apply(self, project: "Project", outcome: OcrOutcome) -> bool:
        if (
            project is not self.project
            or project.session.session_id != self.session_id
            or outcome.status != "succeeded"
        ):
            return False

        def publish() -> bool:
            key = outcome.target_type, outcome.clip_id
            if key in self.consumed:
                return False
            self.consumed.add(key)
            task = self.tasks.get(key)
            expected = self.bindings.get(key)
            current = self._binding(project, task) if task else None
            if (
                expected is None
                or current is None
                or current[0] is not expected[0]
                or current[1] is not expected[1]
                or current[2] != expected[2]
            ):
                return False
            if outcome.target_type == "frame":
                project.update_frame(
                    outcome.clip_id, extracted_texts=outcome.to_models()
                )
            else:
                current[0].extracted_texts = outcome.to_models()
                project.update_clips([current[0]])
            return True

        return project.session.apply_external(publish)
