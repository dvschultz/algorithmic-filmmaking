"""Measure shared library model + clip browser behavior at 1k and 10k clips.

Records the KTD13 / plan U15 budgets: initial population, edit/update
latency, removal, filter toggling, scroll (virtual window re-realization),
and peak RSS. Run headless:

    QT_QPA_PLATFORM=offscreen python scripts/library_model_benchmark.py [--sizes 1000,10000]

Output is one JSON document per size plus a markdown table on stderr.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def _timed(fn):
    start = time.perf_counter()
    fn()
    return round((time.perf_counter() - start) * 1000, 2)


def run(size: int, sources: int = 20) -> dict:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    from core.project import Project
    from models.clip import Clip, Source
    from ui.clip_browser import ClipBrowser
    from ui.project_adapter import ProjectSignalAdapter

    project = Project.new()
    for s in range(sources):
        project.add_source(Source(
            id=f"src-{s}", file_path=Path(f"/bench/video-{s}.mp4"), duration_seconds=3600.0,
            fps=30.0, width=1920, height=1080,
        ))
    clips = [
        Clip(
            id=f"clip-{i}", source_id=f"src-{i % sources}", start_frame=i * 30, end_frame=i * 30 + 29,
            shot_type=("wide", "medium", "close-up")[i % 3],
        )
        for i in range(size)
    ]
    rss_before = _rss_mb()
    adapter = ProjectSignalAdapter(project)
    results: dict = {"clips": size, "sources": sources}

    results["model_populate_ms"] = _timed(lambda: project.add_clips(clips))
    assert len(adapter.clip_model) == size

    cut = ClipBrowser()
    cut.attach_model(adapter.clip_model)
    cut.resize(1400, 900)
    cut.show()
    pairs = [(clip, project.sources_by_id[clip.source_id]) for clip in project.clips]
    results["browser_populate_virtual_ms"] = _timed(lambda: (cut.set_virtual_clips(pairs), app.processEvents()))
    results["realized_cards"] = cut.get_realized_clip_count()

    analyze = ClipBrowser()
    analyze.attach_model(adapter.clip_model)
    analyze.resize(1400, 900)
    analyze.show()
    results["second_workspace_populate_ms"] = _timed(
        lambda: (analyze.set_virtual_clips(pairs), app.processEvents())
    )

    edited = project.clips[: 50]
    for clip in edited:
        clip.shot_type = "extreme close-up"
    cut.selected_clips = {c.id for c in project.clips[:10]}

    def update():
        project.update_clips(edited)
        cut.update_clips(edited, preserve_layout=True)
        analyze.update_clips(edited, preserve_layout=True)
        app.processEvents()

    results["update_50_clips_ms"] = _timed(update)
    assert cut.selected_clips == {c.id for c in project.clips[:10]}

    def scroll():
        bar = cut.scroll.verticalScrollBar()
        for fraction in (0.25, 0.5, 0.75, 1.0, 0.0):
            bar.setValue(int(bar.maximum() * fraction))
            app.processEvents()
            deadline = time.perf_counter() + 0.5
            while cut._rebuild_pending and time.perf_counter() < deadline:
                app.processEvents()

    results["scroll_five_positions_ms"] = _timed(scroll)

    def filter_toggle():
        cut._filter_state.shot_type = {"wide"}
        app.processEvents()
        cut._filter_state.shot_type = set()
        app.processEvents()

    results["filter_toggle_ms"] = _timed(filter_toggle)

    doomed = [clip.id for clip in project.clips[100:200]]

    def remove():
        project.remove_clips(doomed)
        cut.remove_clips_by_ids(doomed)
        analyze.remove_clips_by_ids(doomed)
        app.processEvents()

    results["remove_100_clips_ms"] = _timed(remove)
    assert len(adapter.clip_model) == size - 100

    results["thumbnail_ignore_1000_ms"] = _timed(
        lambda: [adapter.clip_model.thumbnail_ready(cid, "/bench/x.jpg") for cid in doomed * 10]
    )
    results["peak_rss_mb"] = round(_rss_mb(), 1)
    results["rss_growth_mb"] = round(_rss_mb() - rss_before, 1)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1000,10000")
    args = parser.parse_args()
    meta = {
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(),
    }
    rows = [run(int(s)) for s in args.sizes.split(",")]
    print(json.dumps({"meta": meta, "runs": rows}, indent=2))
    keys = [k for k in rows[0] if k not in ("clips", "sources")]
    print("| metric | " + " | ".join(f"{r['clips']} clips" for r in rows) + " |", file=sys.stderr)
    print("|---|" + "---|" * len(rows), file=sys.stderr)
    for key in keys:
        print(f"| {key} | " + " | ".join(str(r[key]) for r in rows) + " |", file=sys.stderr)


if __name__ == "__main__":
    main()
