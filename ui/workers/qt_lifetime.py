"""Safe deferred deletion for GUI coordinators that retain their Qt owner."""

from PySide6.QtCore import QObject, Slot


class RetiringQObject(QObject):
    """Keep normal parent ownership until a coordinator has finished its work."""

    @Slot()
    def retire(self) -> None:
        """Detach before deletion can release the last Python parent reference.

        Otherwise the parent's destructor can recursively delete this child
        while the child's destructor is still running. Call only after owned
        work has finished, or when abandoning an unstarted coordinator.
        """
        self.setParent(None)
        self.deleteLater()
