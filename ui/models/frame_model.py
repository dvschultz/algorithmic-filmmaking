"""Frame library item model projecting a project's extracted frames.

The Frames workspace's ``QListView`` renders straight from this model; the
view keeps selection and zoom, the model keeps the one in-process copy of
frame data. All mutations must happen on the owning (GUI) thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

from PySide6.QtCore import (
    QAbstractListModel, QByteArray, QModelIndex, QObject, QPersistentModelIndex, Qt, QThread,
)

if TYPE_CHECKING:
    from core.project import Project
    from models.frame import Frame


class FrameLibraryModel(QAbstractListModel):
    """Project frames in project order."""

    FrameRole = Qt.ItemDataRole.UserRole + 1
    IdRole = Qt.ItemDataRole.UserRole + 2
    AnalyzedRole = Qt.ItemDataRole.UserRole + 3
    ThumbnailPathRole = Qt.ItemDataRole.UserRole + 4

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._ids: list[str] = []
        self._frames: dict[str, Frame] = {}

    # -- Qt model ------------------------------------------------------------

    def rowCount(  # noqa: N802
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex(),
    ) -> int:
        return 0 if parent.isValid() else len(self._ids)

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        frame = self.frame_at(index)
        if frame is None:
            return None
        if role == self.FrameRole:
            return frame
        if role == self.IdRole:
            return frame.id
        if role == self.AnalyzedRole:
            return frame.analyzed
        if role == self.ThumbnailPathRole:
            return str(frame.thumbnail_path) if frame.thumbnail_path else None
        if role == Qt.ItemDataRole.DecorationRole:
            if frame.thumbnail_path and frame.thumbnail_path.exists():
                return str(frame.thumbnail_path)
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return frame.display_name()
        return None

    def roleNames(self) -> dict[int, QByteArray]:  # noqa: N802
        return {
            self.FrameRole: QByteArray(b"frame"), self.IdRole: QByteArray(b"id"),
            self.AnalyzedRole: QByteArray(b"analyzed"), self.ThumbnailPathRole: QByteArray(b"thumbnail"),
        }

    # -- ownership -----------------------------------------------------------

    def _assert_owner_thread(self) -> None:
        if QThread.currentThread() is not self.thread():
            raise RuntimeError("FrameLibraryModel must be mutated on its owning (GUI) thread")

    # -- population ----------------------------------------------------------

    def set_project(self, project: Project | None) -> None:
        self.set_frames(list(project.frames) if project is not None else [])

    def set_frames(self, frames: Iterable[Frame]) -> None:
        """Replace the whole library (model reset; views drop selection)."""
        self._assert_owner_thread()
        self.beginResetModel()
        self._ids = []
        self._frames = {}
        for frame in frames:
            if frame.id in self._frames:
                continue
            self._ids.append(frame.id)
            self._frames[frame.id] = frame
        self.endResetModel()

    def append(self, frames: Iterable[Frame]) -> list[str]:
        """Insert unknown frames at the end; known ids are refreshed in place."""
        self._assert_owner_thread()
        fresh = []
        refreshed = []
        for frame in frames:
            if frame.id in self._frames:
                refreshed.append(frame)
            else:
                fresh.append(frame)
        if fresh:
            first = len(self._ids)
            self.beginInsertRows(QModelIndex(), first, first + len(fresh) - 1)
            for frame in fresh:
                self._ids.append(frame.id)
                self._frames[frame.id] = frame
            self.endInsertRows()
        self.refresh(refreshed)
        return [frame.id for frame in fresh]

    def refresh(self, frames: Iterable[Frame]) -> list[str]:
        """Replace frame objects that already exist and announce the rows."""
        self._assert_owner_thread()
        changed = []
        for frame in frames:
            if frame.id in self._frames:
                self._frames[frame.id] = frame
                changed.append(frame.id)
        for frame_id in changed:
            row = self._ids.index(frame_id)
            self.dataChanged.emit(self.index(row), self.index(row))
        return changed

    def remove(self, frame_ids: Iterable[str]) -> list[str]:
        self._assert_owner_thread()
        doomed = {frame_id for frame_id in frame_ids if frame_id in self._frames}
        if not doomed:
            return []
        rows = sorted(row for row, frame_id in enumerate(self._ids) if frame_id in doomed)
        for first, last in reversed(_runs(rows)):
            self.beginRemoveRows(QModelIndex(), first, last)
            for frame_id in self._ids[first:last + 1]:
                self._frames.pop(frame_id, None)
            del self._ids[first:last + 1]
            self.endRemoveRows()
        return [frame_id for frame_id in doomed]

    # -- queries -------------------------------------------------------------

    def frame(self, frame_id: str) -> Frame | None:
        return self._frames.get(frame_id)

    def frame_at(self, index: QModelIndex | QPersistentModelIndex) -> Frame | None:
        if not index.isValid() or not 0 <= index.row() < len(self._ids):
            return None
        return self._frames[self._ids[index.row()]]

    def frames(self) -> list[Frame]:
        return [self._frames[frame_id] for frame_id in self._ids]

    def ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, frame_id: object) -> bool:
        return isinstance(frame_id, str) and frame_id in self._frames


def _runs(rows: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted row numbers into inclusive contiguous (first, last) runs."""
    runs: list[tuple[int, int]] = []
    if not rows:
        return runs
    start = prev = rows[0]
    for row in rows[1:]:
        if row == prev + 1:
            prev = row
            continue
        runs.append((start, prev))
        start = prev = row
    runs.append((start, prev))
    return runs
