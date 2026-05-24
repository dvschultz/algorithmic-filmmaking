"""Tests for copyable diagnostics reports."""

from core.diagnostics import build_diagnostics_report


def test_diagnostics_report_includes_environment_and_redacts_logs():
    report = build_diagnostics_report(
        ["provider failed with sk-proj-abcdefghijklmnopqrstuvwxyz123456"]
    )

    assert "Scene Ripper Diagnostics" in report
    assert "Build:" in report
    assert "Platform:" in report
    assert "sk-proj-" not in report
    assert "[REDACTED]" in report
