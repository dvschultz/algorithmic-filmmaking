"""Clip library item model.

One ``ClipLibraryModel`` per open project projects the project's clips and
sources into a Qt list model. Both the Cut and Analyze workspaces read clip
and source objects from it instead of keeping private copies; each
workspace still owns which clips it shows, its selection, and its filters.

All mutations must happen on the thread that owns the model (the GUI
thread). Mutators assert this so a worker thread can never corrupt the view.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from PySide6.QtCore import (
    QAbstractListModel, QByteArray, QModelIndex, QObject, QPersistentModelIndex, Qt, QThread, Signal,
)

from ui.models._utils import runs

if TYPE_CHECKING:
    from core.project import Project
    from models.clip import Clip, Source

logger = logging.getLogger(__name__)


class ClipLibraryModel(QAbstractListModel):
    """Project clips in project order, with their sources."""

    ClipRole = Qt.ItemDataRole.UserRole + 1
    SourceRole = Qt.ItemDataRole.UserRole + 2
    IdRole = Qt.ItemDataRole.UserRole + 3
    SourceIdRole = Qt.ItemDataRole.UserRole + 4
    ThumbnailRole = Qt.ItemDataRole.UserRole + 5

    thumbnail_changed = Signal(str, object)  # clip_id, Path
    """A thumbnail landed for a clip that is still in the library."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ids: list[str] = []
        self._clips: dict[str, Clip] = {}
        self._sources: dict[str, Source] = {}
        self._row_of: dict[str, int] = {}

    # -- Qt model ------------------------------------------------------------

    def rowCount(  # noqa: N802
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        return 0 if parent.isValid() else len(self._ids)

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self._ids):
            return None
        clip_id = self._ids[index.row()]
        clip = self._clips[clip_id]
        if role == self.ClipRole:
            return clip
        if role == self.SourceRole:
            return self._sources.get(clip.source_id)
        if role == self.IdRole:
            return clip_id
        if role == self.SourceIdRole:
            return clip.source_id
        if role == self.ThumbnailRole:
            return clip.thumbnail_path
        if role == Qt.ItemDataRole.DisplayRole:
            return clip.name or clip_id
        return None

    def roleNames(self) -> dict[int, QByteArray]:  # noqa: N802
        return {
            self.ClipRole: QByteArray(b"clip"), self.SourceRole: QByteArray(b"source"),
            self.IdRole: QByteArray(b"id"), self.SourceIdRole: QByteArray(b"source_id"),
            self.ThumbnailRole: QByteArray(b"thumbnail"),
        }

    # -- ownership -----------------------------------------------------------

    def _assert_owner_thread(self) -> None:
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("ClipLibraryModel must be mutated on its owning (GUI) thread")

    # -- population ----------------------------------------------------------

    def set_project(self, project: Project | None) -> None:
        """Replace the whole library from a project (reset, not incremental)."""
        self._assert_owner_thread()
        self.beginResetModel()
        self._ids = []
        self._clips = {}
        self._sources = {}
        self._row_of = {}
        if project is not None:
            self._sources = {source.id: source for source in project.sources}
            for clip in project.clips:
                self._ids.append(clip.id)
                self._clips[clip.id] = clip
            self._reindex()
        self.endResetModel()

    def sync_project(self, project: Project | None) -> tuple[list[str], list[str]]:
        """Reconcile with a project incrementally (no model reset).

        Returns ``(added_ids, removed_ids)``. Views keep selection for rows
        that survive. Restored clips are appended, so row order can differ
        from ``project.clips`` after an undo; browsers order their own
        membership, and ``set_project`` is the wholesale reset when project
        order matters more than selection.
        """
        self._assert_owner_thread()
        if project is None:
            self.set_project(None)
            return [], []
        self.set_sources(project.sources)
        wanted = {clip.id for clip in project.clips}
        removed = self.remove([clip_id for clip_id in self._ids if clip_id not in wanted])
        pairs = [
            (clip, self._sources[clip.source_id])
            for clip in project.clips if clip.source_id in self._sources
        ]
        added, _ = self.upsert(pairs)
        return added, removed

    def set_sources(self, sources: Iterable[Source]) -> None:
        """Refresh source objects (paths, fps) without touching clip rows."""
        self._assert_owner_thread()
        self._sources = {source.id: source for source in sources}
        if self._ids:
            self.dataChanged.emit(self.index(0), self.index(len(self._ids) - 1), [self.SourceRole])

    def upsert(self, pairs: Iterable[tuple[Clip, Source]]) -> tuple[list[str], list[str]]:
        """Insert unknown clips at the end and refresh known ones.

        Returns ``(added_ids, updated_ids)``.
        """
        self._assert_owner_thread()
        added: dict[str, Clip] = {}
        updated: list[str] = []
        for clip, source in pairs:
            self._sources.setdefault(source.id, source)
            if clip.id in self._clips:
                self._clips[clip.id] = clip
                updated.append(clip.id)
            else:
                added[clip.id] = clip  # a repeated id in one batch keeps the last object
        if added:
            first = len(self._ids)
            self.beginInsertRows(QModelIndex(), first, first + len(added) - 1)
            for clip_id, clip in added.items():
                self._ids.append(clip_id)
                self._clips[clip_id] = clip
                self._row_of[clip_id] = len(self._ids) - 1
            self.endInsertRows()
        self._emit_changed(updated)
        return list(added), updated

    def refresh(self, clips: Iterable[Clip]) -> list[str]:
        """Replace clip objects that already exist and announce the change."""
        self._assert_owner_thread()
        changed = []
        for clip in clips:
            if clip.id in self._clips:
                self._clips[clip.id] = clip
                changed.append(clip.id)
        self._emit_changed(changed)
        return changed

    def remove(self, clip_ids: Iterable[str]) -> list[str]:
        """Remove clips by id; contiguous runs are removed in one signal each."""
        self._assert_owner_thread()
        doomed = {clip_id for clip_id in clip_ids if clip_id in self._clips}
        if not doomed:
            return []
        rows = sorted(self._row_of[clip_id] for clip_id in doomed)
        removed = [self._ids[row] for row in rows]  # project order
        # Walk runs from the end so earlier row numbers stay valid.
        for first, last in reversed(runs(rows)):
            self.beginRemoveRows(QModelIndex(), first, last)
            for clip_id in self._ids[first:last + 1]:
                self._clips.pop(clip_id, None)
            del self._ids[first:last + 1]
            self.endRemoveRows()
        self._reindex()
        return removed

    def remove_source(self, source_id: str) -> list[str]:
        removed = self.remove([cid for cid in self._ids if self._clips[cid].source_id == source_id])
        self._sources.pop(source_id, None)
        return removed

    def thumbnail_ready(self, clip_id: str, path: Path | str) -> bool:
        """Record a thumbnail; results for clips no longer in the library are ignored."""
        self._assert_owner_thread()
        clip = self._clips.get(clip_id)
        if clip is None:
            logger.debug("Ignoring thumbnail for removed clip %s", clip_id)
            return False
        clip.thumbnail_path = Path(path)
        row = self._row_of[clip_id]
        self.dataChanged.emit(self.index(row), self.index(row), [self.ThumbnailRole])
        self.thumbnail_changed.emit(clip_id, Path(path))
        return True

    # -- queries -------------------------------------------------------------

    def __contains__(self, clip_id: object) -> bool:
        return isinstance(clip_id, str) and clip_id in self._clips

    def __len__(self) -> int:
        return len(self._ids)

    def ids(self) -> list[str]:
        return list(self._ids)

    def clip(self, clip_id: str) -> Clip | None:
        return self._clips.get(clip_id)

    def source(self, source_id: str) -> Source | None:
        return self._sources.get(source_id)

    def source_for(self, clip_id: str) -> Source | None:
        clip = self._clips.get(clip_id)
        return self._sources.get(clip.source_id) if clip is not None else None

    def entry(self, clip_id: str) -> tuple[Clip, Source] | None:
        clip = self._clips.get(clip_id)
        if clip is None:
            return None
        source = self._sources.get(clip.source_id)
        return (clip, source) if source is not None else None

    def entries(self, clip_ids: Iterable[str] | None = None) -> list[tuple[Clip, Source]]:
        """(Clip, Source) pairs in project order, or in the given id order."""
        ids = self._ids if clip_ids is None else clip_ids
        result = []
        for clip_id in ids:
            pair = self.entry(clip_id)
            if pair is not None:
                result.append(pair)
        return result

    def row_for(self, clip_id: str) -> int | None:
        return self._row_of.get(clip_id)

    # -- internals -----------------------------------------------------------

    def _reindex(self) -> None:
        self._row_of = {clip_id: row for row, clip_id in enumerate(self._ids)}

    def _emit_changed(self, clip_ids: list[str]) -> None:
        if not clip_ids:
            return
        rows = sorted(self._row_of[c] for c in clip_ids if c in self._row_of)
        for first, last in runs(rows):
            self.dataChanged.emit(self.index(first), self.index(last), [self.ClipRole])
