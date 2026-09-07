"""Streaming media fingerprints shared by durable analysis jobs."""

from hashlib import sha256
from pathlib import Path
from threading import Event

from core.jobs.commits import StaleJobResult


def media_stamp(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    except OSError:
        return None


class FingerprintCancelled(Exception):
    pass


class MediaFingerprints:
    def __init__(self, cancel: Event) -> None:
        self.cancel = cancel
        self._cache: dict[Path, tuple[tuple[int, ...], dict]] = {}

    def get(self, media: Path | None) -> dict | None:
        if media is None or (stamp := media_stamp(media)) is None:
            return None
        cached = self._cache.get(media)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        digest = sha256()
        with media.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                if self.cancel.is_set():
                    raise FingerprintCancelled()
                digest.update(block)
        if media_stamp(media) != stamp:
            raise StaleJobResult("Media changed during fingerprinting")
        value = {"stamp": list(stamp), "sha256": digest.hexdigest()}
        self._cache[media] = (stamp, value)
        return value
