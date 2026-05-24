"""Tests for editable transcript search and edit behavior."""

from PySide6.QtWidgets import QApplication

from core.transcription import TranscriptSegment
from ui.widgets.editable_transcript import EditableTranscriptWidget


def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_transcript_search_jumps_to_matching_segment():
    _qapp()
    widget = EditableTranscriptWidget()
    selected = []
    widget.segment_selected.connect(lambda index, start: selected.append((index, start)))
    widget.setSegments(
        [
            TranscriptSegment(start_time=0.0, end_time=1.0, text="alpha"),
            TranscriptSegment(start_time=1.0, end_time=2.0, text="proper noun"),
            TranscriptSegment(start_time=2.0, end_time=3.0, text="another proper noun"),
        ]
    )

    widget.search_edit.setText("proper")

    assert widget.search_count_label.text() == "2"
    assert selected[-1] == (1, 1.0)

    widget.search_edit.returnPressed.emit()

    assert selected[-1] == (2, 2.0)


def test_transcript_edit_emits_updated_segments():
    _qapp()
    widget = EditableTranscriptWidget()
    changed = []
    widget.segments_changed.connect(changed.append)
    widget.setSegments([
        TranscriptSegment(start_time=0.0, end_time=1.0, text="Dada")
    ])

    segment_widget = widget._segment_widgets[0]
    segment_widget._start_editing()
    segment_widget.text_edit.setText("DADA")
    segment_widget._finish_editing()

    assert changed
    assert changed[-1][0].text == "DADA"
