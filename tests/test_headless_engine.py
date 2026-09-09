"""U17 scenario 1: the engine runs a synthetic headless workflow without Qt.

The workflow (synthetic video -> scene detection -> registry generation ->
recipe inspection -> EDL export -> save/load -> MCP tool call) runs in a
child interpreter so this process's imports cannot leak in. It fails if
PySide6, mpv or any optional ML runtime ends up in ``sys.modules``.

In the ``headless-engine`` CI job the environment has no PySide6 at all;
``SCENE_RIPPER_HEADLESS_ENV=1`` makes the child assert that too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKFLOW = r'''
import asyncio, importlib.util, json, os, sys, tempfile
from pathlib import Path

if os.environ.get("SCENE_RIPPER_HEADLESS_ENV") == "1":
    assert importlib.util.find_spec("PySide6") is None, "headless env must not ship PySide6"

import cv2
import numpy as np

from core.project import Project
from core.spine.detect import detect_scenes_for_video
from core.spine.sequences import compare_sequences, generate_sequence, get_sequence_recipe, regenerate_sequence
from core.edl_export import EDLExportConfig, export_edl

work = Path(tempfile.mkdtemp())
video = work / "synthetic.mp4"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (96, 64))
assert writer.isOpened(), "cv2 cannot encode mp4v here"
for color in ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)):
    frame = np.zeros((64, 96, 3), dtype=np.uint8); frame[:] = color
    for _ in range(24):
        writer.write(frame)
writer.release()
assert video.stat().st_size > 0

project = Project.new()
detected = detect_scenes_for_video(project, video, sensitivity=3.0)
assert detected["success"], detected
assert len(project.clips) >= 2, len(project.clips)
first = generate_sequence(project, "sequential", name="A")
assert first["success"], first
second = regenerate_sequence(project, first["sequence_id"], name="B")
assert second["success"], second
recipe = get_sequence_recipe(project, first["sequence_id"])
assert recipe["success"] and recipe["reconstructable"]
comparison = compare_sequences(project, first["sequence_id"], second["sequence_id"])
assert comparison["success"] and comparison["related"]

sequence = next(s for s in project.sequences if s.id == first["sequence_id"])
sources = dict(project.sources_by_id)
clips = {c.id: (c, sources[c.source_id]) for c in project.clips}
config = EDLExportConfig(output_path=work / "a.edl", title="headless")
assert export_edl(sequence, sources, config, clips=clips), config.error_message
placed = len(sequence.get_all_clips())
assert placed >= 2 and (work / "a.edl").read_text().count("V     C") == placed

path = work / "headless.sceneripper"
assert project.save(path)
reloaded = Project.load(path)
assert [s.name for s in reloaded.sequences] == ["A", "B"]

from scene_ripper_mcp.tools import sequence as mcp_sequence
listing = json.loads(asyncio.run(mcp_sequence.list_sequences(str(path))))
assert listing["success"] and len(listing["sequences"]) == 2, listing

forbidden = [m for m in ("PySide6", "mpv", "av", "faster_whisper", "paddleocr", "mlx_vlm", "torch", "ultralytics")
             if m in sys.modules or any(k.startswith(m + ".") for k in sys.modules)]
print(json.dumps({"clips": len(project.clips), "forbidden": forbidden}))
'''


def test_engine_runs_a_synthetic_workflow_without_gui_or_ml_runtimes():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("SCENE_RIPPER_NATIVE_WORKERS", "0")
    # Keep the workflow's managed runtime, config and cache out of the developer's real dirs.
    sandbox = Path(tempfile.mkdtemp(prefix="scene-ripper-headless-"))
    env["SCENE_RIPPER_APP_SUPPORT_DIR"] = str(sandbox / "support")
    env["SCENE_RIPPER_CACHE_DIR"] = str(sandbox / "cache")
    env["SCENE_RIPPER_CONFIG"] = str(sandbox / "config.json")
    proc = subprocess.run(
        [sys.executable, "-c", WORKFLOW], capture_output=True, text=True, cwd=str(ROOT), env=env, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["clips"] >= 2
    assert result["forbidden"] == [], f"headless workflow imported {result['forbidden']}"
