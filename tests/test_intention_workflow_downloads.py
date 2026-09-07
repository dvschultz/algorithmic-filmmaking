"""Regression tests for intention workflow URL downloads."""

from pathlib import Path

from core.intention_workflow import IntentionWorkflowCoordinator
from ui.main_window import MainWindow


def test_download_step_starts_with_workflow_urls():
    class Workflow:
        def get_download_urls(self):
            return ["https://example.com/video"]

    window = MainWindow.__new__(MainWindow)
    window.intention_import_dialog = None
    window.intention_workflow = Workflow()
    started_with = []

    def start_downloads(urls):
        started_with.append(urls)

    window._start_intention_downloads = start_downloads

    MainWindow._on_intention_step_started(window, "downloading", 1, 4)

    assert started_with == [["https://example.com/video"]]


def test_download_urls_are_exposed_as_copy():
    workflow = IntentionWorkflowCoordinator()
    workflow.start(
        algorithm="shuffle",
        local_files=[Path("/tmp/local.mp4")],
        urls=["https://example.com/one"],
    )

    urls = workflow.get_download_urls()
    urls.append("https://example.com/two")

    assert workflow.get_download_urls() == ["https://example.com/one"]


def test_terminal_summary_records_failures_and_missing_item_signals(tmp_path):
    from core.intention_workflow import WorkflowState

    workflow = IntentionWorkflowCoordinator()
    urls = ["https://example.com/one", "https://example.com/two"]
    workflow.start("shuffle", [], urls)
    results = [
        {"url": urls[0], "success": True, "file_path": str(tmp_path / "one.mp4")},
        {"url": urls[1], "success": False, "error": "provider rejected URL"},
    ]
    workflow.on_download_all_finished(results)
    assert workflow.state == WorkflowState.DETECTING
    assert workflow.get_sources_to_detect() == [tmp_path / "one.mp4"]
    assert workflow._sources_failed == [
        {"url": urls[1], "error": "provider rejected URL"}
    ]
    workflow.on_download_all_finished(results)
    assert len(workflow._sources_failed) == 1


def test_missing_terminal_outcomes_are_counted_as_failures():
    workflow = IntentionWorkflowCoordinator()
    urls = ["https://example.com/one", "https://example.com/two"]
    workflow.start("shuffle", [], urls)
    workflow.on_download_all_finished([])
    assert [item["url"] for item in workflow._sources_failed] == urls
