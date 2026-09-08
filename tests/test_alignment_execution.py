"""Alignment reports model execution separately from approximate fallback."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from core.analysis import alignment
from core.transcription_models import TranscriptSegment


def test_real_engine_reports_loaded_model_before_inference(tmp_path, monkeypatch):
    events = []
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="revision-123"))
    loader = Mock(return_value=(model, "tokenizer"))

    def emissions(*args):
        assert events[0]["revision"] == "revision-123"
        return "emissions", 1

    runtime = SimpleNamespace(
        load_alignment_model=loader,
        load_audio=lambda *a, **k: "audio",
        generate_emissions=emissions,
        preprocess_text=lambda *a, **k: ("tokens", "text"),
        get_alignments=lambda *a: ("segments", "scores", 0),
        get_spans=lambda *a: "spans",
        postprocess_results=lambda *a: [],
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fp32"))
    monkeypatch.setitem(sys.modules, "ctc_forced_aligner", runtime)
    assert (
        alignment._execute_alignment(
            tmp_path / "audio.wav",
            "hello",
            "en",
            scope="whole_clip",
            on_execution=events.append,
        )
        == []
    )
    loader.assert_called_once_with(
        device="cpu", dtype="fp32", model_path=alignment.ALIGNMENT_MODEL
    )
    assert events == [
        {
            "backend": "ctc",
            "model": alignment.ALIGNMENT_MODEL,
            "revision": "revision-123",
            "device": "cpu",
            "dtype": "float32",
            "language": "en",
            "romanize": True,
            "scope": "whole_clip",
            "source_range": None,
        }
    ]


def test_fallback_reports_ctc_attempts_and_approximation(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(alignment, "_check_language_supported", lambda _: None)
    calls = []

    def engine(*, wav_path, text, language, on_execution):
        on_execution(
            {"backend": "ctc", "model": alignment.ALIGNMENT_MODEL, "revision": "r1"}
        )
        calls.append(text)
        raise RuntimeError("targets length is too long for CTC")

    def extract(*a, **k):
        segment = tmp_path / "segment.wav"
        segment.write_bytes(b"segment")
        return segment

    monkeypatch.setattr(alignment, "_run_alignment_engine", engine)
    monkeypatch.setattr(alignment, "extract_audio_to_wav", extract)
    events = []
    words = alignment.align_words(
        str(audio),
        [TranscriptSegment(0, 1, "hello world", language="en")],
        extract_audio=False,
        on_execution=events.append,
    )
    assert [word.text for word in words] == ["hello", "world"]
    assert [event["backend"] for event in events] == ["ctc", "ctc", "uniform"]
    assert [event["scope"] for event in events] == ["whole_clip", "segment", "segment"]
    assert events[-1]["source_range"] == [0, 1]
    assert events[-1]["model"] is None
    assert len(calls) == 2
    assert audio.exists()
    assert not (tmp_path / "segment.wav").exists()


def test_empty_input_reports_no_model_execution():
    events = []
    assert alignment.align_words("missing.wav", [], on_execution=events.append) == []
    assert events == [
        {"backend": "empty", "scope": "whole_clip", "model": None, "revision": None}
    ]
