"""References carried in structured job data and embedded JSON documents."""

import json
from typing import Iterable

from models.analysis_record import ArtifactRef


def referenced_artifacts(columns: Iterable[str | None]) -> tuple[ArtifactRef, ...]:
    """Keep referenced files, including records encoded in ``*_json`` fields.

    Ordinary user strings are not decoded. Old opaque receipt strings remain
    readable; malformed references do not become ownership claims.
    """
    pending: list[object] = []
    for column in columns:
        if column is not None:
            try:
                pending.append(json.loads(column))
            except (ValueError, TypeError):
                pass
    refs: dict[str, ArtifactRef] = {}
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if {"sha256", "size", "media_type"} <= value.keys():
                try:
                    ref = ArtifactRef.from_dict(value)
                    refs[ref.digest] = ref
                except (ValueError, KeyError, TypeError):
                    pass
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    pending.append(child)
                elif isinstance(key, str) and key.endswith("_json") and isinstance(child, str):
                    try:
                        pending.append(json.loads(child))
                    except ValueError:
                        pass
        elif isinstance(value, list):
            pending.extend(value)
    return tuple(refs.values())
