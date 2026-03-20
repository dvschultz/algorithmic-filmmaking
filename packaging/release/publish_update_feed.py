#!/usr/bin/env python3
"""Publish generated update-feed artifacts into a target directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def publish_directory(source_dir: Path, target_dir: Path) -> None:
    """Copy update-feed artifacts into the publish directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.rglob("*"):
        if path.is_dir():
            continue
        relative_path = path.relative_to(source_dir)
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publish_directory(Path(args.source_dir), Path(args.target_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
