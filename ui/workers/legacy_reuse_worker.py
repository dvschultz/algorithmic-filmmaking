"""Detached legacy reuse decisions with owner-thread application objects."""

from functools import partial
from collections.abc import Callable
from typing import cast

from PySide6.QtCore import Signal

from core.operations.colors import ColorApplication, color_request
from core.operations.embeddings import EmbeddingApplication, embedding_task
from core.operations.legacy_reuse import LEGACY_REUSE_OPERATIONS, accept_legacy_colors, accept_legacy_embeddings, accept_legacy_scalars, accept_legacy_visuals, accept_legacy_boundaries
from core.operations.boundary_embeddings import BoundaryEmbeddingApplication, boundary_embedding_task
from core.operations.classification import ClassificationApplication, ClassificationOptions, classification_task
from core.operations.object_detection import ObjectDetectionApplication, object_detection_task
from core.operations.scalars import ScalarBatchApplication, ScalarOperation, scalar_task
from core.project import Project
from models.analysis_record import AnalysisRecord
from ui.workers.base import CancellableWorker
from ui.workers.gui_tool_reply import GuiToolReply


class LegacyReuseWorker(CancellableWorker):
    result_ready = Signal(object)

    def __init__(self, project: Project, operation: str, clip_ids: list[str], parent=None) -> None:
        super().__init__(parent)
        self.gui_tool_reply: GuiToolReply | None = None
        project.session.assert_owner()
        if operation not in LEGACY_REUSE_OPERATIONS:
            raise ValueError("Unsupported legacy reuse operation")
        ids = list(dict.fromkeys(clip_ids))
        if not ids:
            raise ValueError("Select clips to reuse their legacy analysis")
        for cid in ids:
            clip = project.clips_by_id.get(cid)
            if clip is None:
                raise ValueError("Selected clip is no longer in the project")
            record = clip.analysis_records.get(operation)
            if record is not None and not isinstance(record, AnalysisRecord):
                raise ValueError("Unknown analysis record must be preserved; recompute analysis")
        self.operation = operation
        self.request = color_request(project, ids, skip_existing=False) if operation == "colors" else None
        self.tasks = tuple(embedding_task(project.clips_by_id[cid], project.sources_by_id.get(project.clips_by_id[cid].source_id), skip_existing=False) for cid in ids) if operation == "embeddings" else ()
        self.application: ColorApplication | EmbeddingApplication | ScalarBatchApplication | BoundaryEmbeddingApplication | ClassificationApplication | ObjectDetectionApplication
        self._compute: Callable[[], object]
        if self.request is not None:
            self.application = ColorApplication(project, self.request)
            self._compute = partial(accept_legacy_colors, self.request, cancel_event=self._cancel_event)
        elif operation == "embeddings":
            self.application = EmbeddingApplication(project, self.tasks)
            self._compute = partial(accept_legacy_embeddings, self.tasks, cancel_event=self._cancel_event)
        elif operation == "boundary_embeddings":
            boundary_tasks = tuple(boundary_embedding_task(project.clips_by_id[cid], project.sources_by_id.get(project.clips_by_id[cid].source_id)) for cid in ids)
            self.application = BoundaryEmbeddingApplication(project, boundary_tasks)
            self._compute = partial(accept_legacy_boundaries, boundary_tasks, cancel_event=self._cancel_event)
        elif operation == "classify":
            visual_tasks = tuple(classification_task(project.clips_by_id[cid], project.sources_by_id.get(project.clips_by_id[cid].source_id)) for cid in ids)
            self.application = ClassificationApplication(project, visual_tasks, ClassificationOptions())
            self._compute = partial(accept_legacy_visuals, visual_tasks, cancel_event=self._cancel_event)
        elif operation == "detect_objects":
            object_tasks = tuple(object_detection_task(project.clips_by_id[cid], project.sources_by_id.get(project.clips_by_id[cid].source_id)) for cid in ids)
            self.application = ObjectDetectionApplication(project, object_tasks)
            self._compute = partial(accept_legacy_visuals, object_tasks, cancel_event=self._cancel_event)
        else:
            tasks = tuple(scalar_task(project.clips_by_id[cid], project.sources_by_id.get(project.clips_by_id[cid].source_id), cast(ScalarOperation, operation)) for cid in ids)
            self.application = ScalarBatchApplication(project, tasks)
            self._compute = partial(accept_legacy_scalars, tasks, cancel_event=self._cancel_event)

    def run(self) -> None:
        try:
            result = self._compute()
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
