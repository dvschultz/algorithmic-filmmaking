"""End-of-batch analysis failure summary with a "Retry Failed" action."""

from PySide6.QtWidgets import QMessageBox, QWidget

from core.operations.clip_analysis import ClipAnalysisFailureReport


def ask_retry_failed_analysis(
    parent: QWidget, report: ClipAnalysisFailureReport
) -> bool:
    """Show success/failure counts and return True if the user wants a retry.

    The visible text stays bounded; the complete per-clip error list is
    available under "Show Details...".
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Analysis Incomplete")
    box.setText(report.summary)
    box.setInformativeText(report.details)
    if report.full_log and report.full_log != report.details:
        box.setDetailedText(report.full_log)
    retry = None
    if report.retry_clip_ids and report.retry_operations:
        retry = box.addButton(
            f"Retry Failed ({len(report.retry_clip_ids)})",
            QMessageBox.ButtonRole.AcceptRole,
        )
    close = box.addButton(QMessageBox.StandardButton.Close)
    box.setDefaultButton(close)
    box.exec()
    return retry is not None and box.clickedButton() is retry
