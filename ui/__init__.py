"""UI components for the application."""

__all__ = ["MainWindow"]


def __getattr__(name: str):
    if name == "MainWindow":
        from ui.main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module 'ui' has no attribute {name!r}")
