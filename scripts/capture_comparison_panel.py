"""Render the Compare A/B panel offscreen to a PNG (U16 verification aid).

    QT_QPA_PLATFORM=offscreen python scripts/capture_comparison_panel.py docs/user-guide/images/compare-ab.png
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(output: Path) -> None:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from core.project import Project
    from core.spine.sequences import generate_sequence, regenerate_sequence
    from models.clip import Clip, Source
    from ui.widgets.sequence_comparison import SequenceComparisonPanel

    project = Project.new()
    media = Path(tempfile.mkdtemp()) / "interview.mp4"
    media.write_bytes(b"0")
    project.add_source(Source(id="s", file_path=media, fps=25.0, duration_seconds=120))
    project.add_clips([
        Clip(id=f"c{i}", source_id="s", start_frame=i * 50, end_frame=(i + 1) * 50 + i * 7, dominant_colors=[(i * 30, 40, 60)])
        for i in range(8)
    ])
    first = generate_sequence(project, "shuffle", seed=42, name="Shuffle A")
    a = project.sequence
    regenerate_sequence(project, a.id, parameters={"hflip": True, "max_consecutive_same_source": 2}, name="Shuffle B (flipped)")
    b = project.sequence
    panel = SequenceComparisonPanel()
    panel.set_project(project)
    panel.set_preview_probe(lambda sequence: sequence.id == a.id)
    panel.select("a", a.id)
    panel.select("b", b.id)
    panel.resize(900, 320)
    panel.show()
    app.processEvents()
    output.parent.mkdir(parents=True, exist_ok=True)
    assert panel.grab().save(str(output)), f"could not write {output}"
    print(f"wrote {output} ({first['clip_count']} clips per side)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "compare-ab.png"))
