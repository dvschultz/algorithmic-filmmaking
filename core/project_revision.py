"""Content revisions for headless models kept between project operations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path


class ProjectRevisionConflict(RuntimeError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Project changed externally: {path}. Reload before editing or saving."
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "code": "project_modified_externally",
            "path": str(self.path),
            "message": str(self),
        }


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


@dataclass
class ProjectFileRevision:
    path: Path
    digest: str
    invalidated: bool = False

    @classmethod
    def capture(cls, path: Path) -> ProjectFileRevision:
        canonical = path.expanduser().resolve()
        return cls(canonical, _digest(canonical))

    def verify(self) -> None:
        """Fail closed on replacement, deletion, unreadability, or prior conflict."""
        try:
            unchanged = not self.invalidated and _digest(self.path) == self.digest
        except OSError:
            unchanged = False
        if not unchanged:
            self.invalidated = True
            raise ProjectRevisionConflict(self.path)
