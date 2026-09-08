"""Detached legacy reuse decisions with owner-thread application objects."""

from PySide6.QtCore import Signal

from core.operations.colors import ColorApplication, color_request
from core.operations.embeddings import EmbeddingApplication, embedding_task
from core.operations.legacy_reuse import accept_legacy_colors, accept_legacy_embeddings
from core.project import Project
from models.analysis_record import AnalysisRecord
from ui.workers.base import CancellableWorker


class LegacyReuseWorker(CancellableWorker):
    result_ready = Signal(object)

    def __init__(self, project: Project, operation: str, clip_ids: list[str], parent=None) -> None:
        super().__init__(parent)
        project.session.assert_owner()
        if operation not in ("colors", "embeddings"):
            raise ValueError("Legacy reuse supports colors and embeddings")
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
        self.application = ColorApplication(project, self.request) if self.request is not None else EmbeddingApplication(project, self.tasks)

    def run(self) -> None:
        try:
            result = accept_legacy_colors(self.request, cancel_event=self._cancel_event) if self.request is not None else accept_legacy_embeddings(self.tasks, cancel_event=self._cancel_event)
            self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
