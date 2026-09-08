"""Content identities for the ONNX files used to initialize a face model."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event

from core.jobs.media import MediaFingerprints, media_stamp


@dataclass(frozen=True)
class FaceWeights:
    directory: Path
    files: tuple[tuple[Path, tuple[int, ...], str], ...]

    @classmethod
    def capture(cls, directory: Path) -> "FaceWeights":
        """Hash a staged model pack before ONNX sessions open its files."""
        directory = directory.resolve()
        fingerprints = MediaFingerprints(Event())
        files = []
        for path in sorted(directory.glob("*.onnx")):
            value = fingerprints.get(path)
            if value is None:
                raise ValueError("Face model weight file is missing")
            files.append((path, tuple(value["stamp"]), value["sha256"]))
        if not files:
            raise ValueError("Face model pack has no ONNX weight files")
        result = cls(directory, tuple(files))
        if not result.unchanged():
            raise ValueError("Face model weights changed during fingerprinting")
        return result

    def unchanged(self) -> bool:
        """Check loaded-weight bindings without rehashing or running inference."""
        return tuple(sorted(self.directory.glob("*.onnx"))) == tuple(
            path for path, _, _ in self.files
        ) and all(media_stamp(path) == stamp for path, stamp, _ in self.files)

    def component(self, path: Path) -> dict:
        """Return the fingerprint captured before this component was loaded."""
        path = path.resolve()
        for captured, stamp, digest in self.files:
            if captured.resolve() == path:
                return {"sha256": digest, "stamp": list(stamp)}
        raise ValueError("Face component was not part of the staged model pack")
