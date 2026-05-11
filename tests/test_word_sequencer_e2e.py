"""End-to-end smoke test for the Word Sequencer pipeline.

Covers the path U6 wires up:

    aligned (Clip, Source) inputs
        → ``WordSequencerDialog`` programmatic accept
        → ``generate_word_sequence`` (spine + remix)
        → ``SequenceClip[]`` emitted on ``sequence_ready``

This is a *Python-level* smoke that proves the dialog → worker → spine →
SequenceClip wiring is intact. The optional MP4 render step is gated behind
``SCENE_RIPPER_E2E_RENDER=1`` because rendering pulls in FFmpeg and real
source media that aren't available in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.transcription import TranscriptSegment, WordTimestamp
from models.clip import Clip, Source

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_50_word_source() -> tuple[Clip, Source]:
    """Build a 30-second synthetic clip with 50 aligned words.

    Each word spans 600ms (50 × 0.6 = 30s). Word texts are chosen so that
    alphabetical order produces a deterministic 50-cut timeline.
    """
    source = Source(
        id="src-e2e",
        file_path=Path("/test/e2e.mp4"),
        duration_seconds=30.0,
        fps=24.0,
        width=640,
        height=360,
    )
    clip = Clip(
        id="clip-e2e",
        source_id=source.id,
        start_frame=0,
        end_frame=int(30.0 * 24.0),  # 720
    )
    word_texts = [
        # 50 deliberately distinct lowercase words.
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "xray", "yankee", "zulu", "amber", "blossom",
        "cinder", "dune", "ember", "flint", "gable", "harbor", "ivy",
        "jasper", "kestrel", "loam", "moss", "nimbus", "ochre", "pine",
        "quartz", "ridge", "stone", "tide", "umber", "vale", "willow", "yew",
    ]
    assert len(word_texts) == 50

    words = []
    for i, text in enumerate(word_texts):
        start = i * 0.6
        end = start + 0.6
        words.append(WordTimestamp(start=start, end=end, text=text))

    clip.transcript = [
        TranscriptSegment(
            start_time=0.0,
            end_time=30.0,
            text=" ".join(word_texts),
            confidence=0.95,
            words=words,
            language="en",
        )
    ]
    return clip, source


def test_alphabetical_50_word_smoke(qapp):
    """The integration gate: 30-second seeded source → 50 SequenceClip entries."""
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_50_word_source()

    captured: list = []
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)
    dialog.sequence_ready.connect(captured.append)

    # Alphabetical mode is the default (index 0).
    assert dialog._mode_combo.currentIndex() == 0

    dialog._on_accept()

    assert len(captured) == 1, "dialog must emit sequence_ready exactly once"
    sequence_clips = captured[0]
    assert len(sequence_clips) == 50, (
        f"expected 50 SequenceClips from 50 aligned words, got {len(sequence_clips)}"
    )

    # Frame coords are clip-relative integers within the clip bounds.
    for sc in sequence_clips:
        assert sc.source_clip_id == clip.id
        assert sc.source_id == source.id
        assert sc.in_point >= 0
        assert sc.out_point > sc.in_point
        assert sc.out_point <= clip.end_frame - clip.start_frame

    # Materializing into a Sequence/Track must produce a contiguous timeline.
    from models.sequence import Sequence, Track

    track = Track(name="Video 1")
    current_frame = 0
    for sc in sequence_clips:
        sc.start_frame = current_frame
        track.clips.append(sc)
        current_frame += sc.out_point - sc.in_point

    sequence = Sequence(
        name="e2e",
        fps=source.fps,
        tracks=[track],
        algorithm="word_sequencer",
    )
    assert sequence.duration_frames == current_frame
    assert sequence.duration_frames > 0


@pytest.mark.skipif(
    os.environ.get("SCENE_RIPPER_E2E_RENDER") != "1",
    reason="optional MP4 render gate; set SCENE_RIPPER_E2E_RENDER=1 to enable",
)
def test_render_produces_mp4(tmp_path, qapp):
    """When opted in, render the 50-clip sequence to an MP4 and check the
    file exists with non-zero size."""
    from ui.dialogs.word_sequencer_dialog import WordSequencerDialog

    clip, source = _make_50_word_source()

    captured: list = []
    dialog = WordSequencerDialog(clips=[(clip, source)], project=None)
    dialog.sequence_ready.connect(captured.append)
    dialog._on_accept()

    assert captured, "no sequence emitted; render step cannot run"
    sequence_clips = captured[0]

    # The renderer lives in core/export; pull it lazily so the smoke test
    # doesn't import heavy dependencies unless this branch runs.
    try:
        from core.export import render_sequence_to_mp4  # type: ignore
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"render entrypoint unavailable: {exc}")

    output = tmp_path / "e2e.mp4"
    render_sequence_to_mp4(  # type: ignore[misc]
        sequence_clips=sequence_clips,
        sources=[source],
        clips=[clip],
        output_path=output,
    )
    assert output.exists()
    assert output.stat().st_size > 0
