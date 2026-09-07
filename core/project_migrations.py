"""Ordered, non-destructive JSON project schema upgrades."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

SCHEMA_VERSION = "1.4"


def schema_version(value: Any) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value):
        raise ValueError(f"Invalid project schema version: {value!r}")
    parts = tuple(int(part) for part in value.split("."))
    while len(parts) > 1 and parts[-1] == 0:
        parts = parts[:-1]
    return parts


def is_future_schema(version: Any) -> bool:
    return schema_version(version) > schema_version(SCHEMA_VERSION)


def _multiple_sequences(data: dict) -> None:
    """Preserve old timeline values verbatim; do not infer trim coordinates."""
    if not data.get("sequences") and data.get("sequence") is not None:
        data["sequences"] = [deepcopy(data["sequence"])]
        data["active_sequence_index"] = 0


MIGRATIONS = (("1.4", _multiple_sequences),)


def migrate_project_data(data: dict) -> dict:
    """Upgrade a detached document, leaving future schemas intact for inspection."""
    version = schema_version(data.get("version"))
    if version < schema_version("1.0"):
        raise ValueError("Project schemas older than 1.0 are unsupported")
    result = deepcopy(data)
    for target, migration in MIGRATIONS:
        if version < schema_version(target):
            migration(result)
            result["version"] = target
            version = schema_version(target)
    return result


def prepare_project_write(path: Path, version: str) -> None:
    """Reject unknown schemas and back up exact bytes before an upgrade write.

    This is schema protection, not cross-process writer ownership. The stable
    backup name identifies its content and existing backups are never replaced.
    """
    if is_future_schema(version):
        raise ValueError(
            f"Project schema {version} is newer than {SCHEMA_VERSION}; read-only"
        )
    if schema_version(version) < schema_version("1.0"):
        raise ValueError("Project schemas older than 1.0 are unsupported")
    if not path.exists():
        return
    original = path.read_bytes()
    data = json.loads(original)
    if not isinstance(data, dict):
        raise ValueError("Existing project must be a JSON object")
    old_version = data.get("version")
    if is_future_schema(old_version):
        raise ValueError(
            f"Destination schema {old_version} is newer than {SCHEMA_VERSION}; read-only"
        )
    if schema_version(old_version) < schema_version("1.0"):
        raise ValueError("Destination schema is older than supported 1.0")
    if schema_version(old_version) >= schema_version(SCHEMA_VERSION):
        return
    backup = path.with_name(
        f"{path.name}.pre-v{SCHEMA_VERSION}-{sha256(original).hexdigest()[:16]}.bak"
    )
    fd, temporary_name = tempfile.mkstemp(prefix=".project_backup_", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Link publishes complete content atomically without replacing an
            # existing backup. The temporary inode is in the same directory.
            os.link(temporary, backup)
        except FileExistsError:
            if backup.read_bytes() != original:
                raise OSError(f"Migration backup differs from original: {backup}")
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
