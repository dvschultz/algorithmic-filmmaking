"""Tests for update-feed generation and verification."""

import importlib.util
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def _load_module(name: str, relative_path: str):
    project_root = Path(__file__).resolve().parent.parent
    module_path = project_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_appcast = _load_module("scene_ripper_generate_appcast", "packaging/release/generate_appcast.py")
publish_update_feed = _load_module("scene_ripper_publish_update_feed", "packaging/release/publish_update_feed.py")
verify_update_feed = _load_module("scene_ripper_verify_update_feed", "packaging/release/verify_update_feed.py")

AppcastItem = generate_appcast.AppcastItem
write_appcast = generate_appcast.write_appcast
write_release_notes = generate_appcast.write_release_notes
publish_directory = publish_update_feed.publish_directory
verify_feed = verify_update_feed.verify_feed

SPARKLE_NS = "http://www.andymatuschak.org/xml-namespaces/sparkle"


def test_write_appcast_contains_expected_enclosure_fields(tmp_path):
    """Generated feeds should contain the metadata Sparkle and WinSparkle expect."""
    output = tmp_path / "appcast.xml"
    write_appcast(
        AppcastItem(
            platform="macos",
            version="0.2.0",
            build_version="20260320.1",
            release_tag="v0.2.0",
            release_url="https://example.com/release",
            enclosure_url="https://example.com/app.dmg",
            enclosure_length=123,
            enclosure_type="application/x-apple-diskimage",
            pub_date="Thu, 20 Mar 2026 10:00:00 +0000",
            signature="abc123",
            notes_url="https://example.com/notes.html",
            minimum_system_version="13.0",
        ),
        output,
    )

    tree = ET.parse(output)
    enclosure = tree.find("./channel/item/enclosure")
    assert enclosure is not None
    assert enclosure.get("url") == "https://example.com/app.dmg"
    assert enclosure.get(f"{{{SPARKLE_NS}}}version") == "20260320.1"
    assert enclosure.get(f"{{{SPARKLE_NS}}}shortVersionString") == "0.2.0"
    assert enclosure.get(f"{{{SPARKLE_NS}}}edSignature") == "abc123"


def test_verify_feed_accepts_valid_appcast(tmp_path):
    """Verification should succeed for a valid generated feed."""
    output = tmp_path / "appcast.xml"
    write_appcast(
        AppcastItem(
            platform="windows",
            version="0.2.0",
            build_version="20260320.1",
            release_tag="v0.2.0",
            release_url="https://example.com/release",
            enclosure_url="https://example.com/setup.exe",
            enclosure_length=456,
            enclosure_type="application/x-msdownload",
            pub_date="Thu, 20 Mar 2026 10:00:00 +0000",
            signature="sig",
        ),
        output,
    )

    verify_feed(output, require_signature=True)


def test_publish_directory_copies_generated_feed(tmp_path):
    """Publishing should mirror generated update artifacts into a target directory."""
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    (source_dir / "appcast.xml").write_text("<rss />", encoding="utf-8")
    write_release_notes(
        AppcastItem(
            platform="windows",
            version="0.2.0",
            build_version="20260320.1",
            release_tag="v0.2.0",
            release_url="https://example.com/release",
            enclosure_url="https://example.com/setup.exe",
            enclosure_length=456,
            enclosure_type="application/x-msdownload",
            pub_date="Thu, 20 Mar 2026 10:00:00 +0000",
        ),
        source_dir / "notes.html",
    )

    publish_directory(source_dir, target_dir)

    assert (target_dir / "appcast.xml").exists()
    assert (target_dir / "notes.html").exists()
