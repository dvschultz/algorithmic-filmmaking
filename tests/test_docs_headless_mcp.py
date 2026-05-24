from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_headless_mcp_docs_cover_cli_mcp_and_credentials():
    doc = (PROJECT_ROOT / "docs" / "user-guide" / "headless-mcp.md").read_text(
        encoding="utf-8"
    )

    assert "scene_ripper detect" in doc
    assert "scene_ripper transcribe" in doc
    assert "scene_ripper export sequence" in doc
    assert "start_detect_scenes_new_project(" in doc
    assert "start_transcribe(" in doc
    assert "start_download_videos(" in doc
    assert "OPENAI_API_KEY" in doc
    assert "YOUTUBE_API_KEY" in doc
    assert "ollama serve" in doc
