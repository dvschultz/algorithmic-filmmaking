#!/usr/bin/env python3
"""Generate a single-item Sparkle/WinSparkle-compatible appcast feed."""

from __future__ import annotations

import argparse
import email.utils
import html
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from xml.etree import ElementTree as ET

SPARKLE_NS = "http://www.andymatuschak.org/xml-namespaces/sparkle"
ET.register_namespace("sparkle", SPARKLE_NS)


@dataclass(frozen=True)
class AppcastItem:
    platform: str
    version: str
    build_version: str
    release_tag: str
    release_url: str
    enclosure_url: str
    enclosure_length: int
    enclosure_type: str
    pub_date: str
    signature: str = ""
    notes_url: str = ""
    minimum_system_version: str = ""


def build_appcast(item: AppcastItem) -> ET.ElementTree:
    """Build an RSS appcast for one release item."""
    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = f"Scene Ripper {item.platform.title()} Updates"
    ET.SubElement(channel, "link").text = item.release_url
    ET.SubElement(channel, "description").text = (
        f"Latest Scene Ripper updates for {item.platform.title()}."
    )
    ET.SubElement(channel, "language").text = "en"

    release_item = ET.SubElement(channel, "item")
    ET.SubElement(release_item, "title").text = f"Scene Ripper {item.version}"
    ET.SubElement(release_item, "pubDate").text = item.pub_date
    ET.SubElement(release_item, "link").text = item.release_url

    if item.notes_url:
        ET.SubElement(release_item, f"{{{SPARKLE_NS}}}releaseNotesLink").text = item.notes_url

    enclosure_attrs = {
        "url": item.enclosure_url,
        "length": str(item.enclosure_length),
        "type": item.enclosure_type,
        f"{{{SPARKLE_NS}}}version": item.build_version,
        f"{{{SPARKLE_NS}}}shortVersionString": item.version,
        f"{{{SPARKLE_NS}}}os": item.platform,
    }
    if item.signature:
        enclosure_attrs[f"{{{SPARKLE_NS}}}edSignature"] = item.signature
    if item.minimum_system_version:
        enclosure_attrs[f"{{{SPARKLE_NS}}}minimumSystemVersion"] = item.minimum_system_version

    ET.SubElement(release_item, "enclosure", enclosure_attrs)
    return ET.ElementTree(rss)


def write_appcast(item: AppcastItem, output_path: Path) -> None:
    """Write the appcast XML to disk."""
    tree = build_appcast(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    output_path.write_bytes(xml_bytes)


def write_release_notes(item: AppcastItem, output_path: Path) -> None:
    """Write simple HTML release notes for the release feed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html lang=\"en\">",
                "<head>",
                "  <meta charset=\"utf-8\">",
                f"  <title>Scene Ripper {html.escape(item.version)} Release Notes</title>",
                "</head>",
                "<body>",
                f"  <h1>Scene Ripper {html.escape(item.version)}</h1>",
                f"  <p>Platform: {html.escape(item.platform.title())}</p>",
                f"  <p>Release tag: {html.escape(item.release_tag)}</p>",
                f"  <p><a href=\"{html.escape(item.release_url)}\">View GitHub release</a></p>",
                f"  <p><a href=\"{html.escape(item.enclosure_url)}\">Download update</a></p>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )


def _default_enclosure_type(platform: str) -> str:
    return {
        "macos": "application/x-apple-diskimage",
        "windows": "application/x-msdownload",
    }.get(platform, "application/octet-stream")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("macos", "windows"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--build-version")
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-url", required=True)
    parser.add_argument("--enclosure-url", required=True)
    parser.add_argument("--enclosure-length", type=int, required=True)
    parser.add_argument("--enclosure-type")
    parser.add_argument("--signature", default="")
    parser.add_argument("--notes-url", default="")
    parser.add_argument("--minimum-system-version", default="")
    parser.add_argument("--published-at", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--notes-output", default="")
    return parser.parse_args()


def _published_at_to_rfc2822(value: str) -> str:
    if not value:
        return email.utils.format_datetime(datetime.now(UTC))
    return email.utils.format_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))


def main() -> int:
    args = _parse_args()
    item = AppcastItem(
        platform=args.platform,
        version=args.version,
        build_version=args.build_version or args.version,
        release_tag=args.release_tag,
        release_url=args.release_url,
        enclosure_url=args.enclosure_url,
        enclosure_length=args.enclosure_length,
        enclosure_type=args.enclosure_type or _default_enclosure_type(args.platform),
        pub_date=_published_at_to_rfc2822(args.published_at),
        signature=args.signature,
        notes_url=args.notes_url,
        minimum_system_version=args.minimum_system_version,
    )
    write_appcast(item, Path(args.output))
    if args.notes_output:
        write_release_notes(item, Path(args.notes_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
