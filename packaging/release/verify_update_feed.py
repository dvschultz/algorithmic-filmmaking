#!/usr/bin/env python3
"""Validate generated update-feed XML files."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

SPARKLE_NS = {"sparkle": "http://www.andymatuschak.org/xml-namespaces/sparkle"}


def verify_feed(path: Path, require_signature: bool = False) -> None:
    """Raise ValueError if the feed is invalid."""
    tree = ET.parse(path)
    item = tree.find("./channel/item")
    if item is None:
        raise ValueError("Appcast missing channel/item entry")

    enclosure = item.find("enclosure")
    if enclosure is None:
        raise ValueError("Appcast missing enclosure")

    for attr in ("url", "length", "type"):
        if not enclosure.get(attr):
            raise ValueError(f"Appcast enclosure missing {attr}")

    for attr in ("sparkle:version", "sparkle:shortVersionString"):
        local_name = attr.split(":", 1)[1]
        if not enclosure.get(f"{{{SPARKLE_NS['sparkle']}}}{local_name}"):
            raise ValueError(f"Appcast enclosure missing {attr}")

    if require_signature and not enclosure.get(f"{{{SPARKLE_NS['sparkle']}}}edSignature"):
        raise ValueError("Appcast enclosure missing sparkle:edSignature")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feed_path")
    parser.add_argument("--require-signature", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    verify_feed(Path(args.feed_path), require_signature=args.require_signature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
