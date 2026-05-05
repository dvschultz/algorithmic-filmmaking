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
