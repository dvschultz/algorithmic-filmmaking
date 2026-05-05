"""Tests that every name in lazy package ``__all__`` resolves cleanly.

``core/__init__.py`` and ``ui/__init__.py`` use ``__getattr__`` to defer
heavy submodule imports (mpv, PyAV, etc.) until first attribute access.
That pattern is fragile: a name added to ``__all__`` without a matching
``__getattr__`` branch silently fails only when the name is first used.
These tests guard against that by walking every ``__all__`` entry and
asserting it resolves without raising.

The "no mpv / no PyAV" invariants are checked in a subprocess so that
modules loaded by other tests in the same pytest run don't pollute
``sys.modules`` and produce false failures.
"""

import subprocess
import sys
import textwrap


def _run_in_clean_interpreter(script: str) -> subprocess.CompletedProcess:
    """Execute ``script`` in a fresh ``python`` subprocess and capture output."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_all_core_all_names_resolve_via_lazy_getattr():
    """Every name in ``core.__all__`` must resolve through ``__getattr__``."""
    import core

    for name in core.__all__:
        resolved = getattr(core, name)
        assert resolved is not None, f"core.{name} resolved to None"


def test_core_lazy_attrs_do_not_load_native_media_runtimes():
    """Resolving every ``core.__all__`` name must not pull mpv or PyAV in.

    Runs in a subprocess so it's immune to other tests in the pytest run
    that may legitimately load these modules.
    """
    script = """
        import sys, core
        for name in core.__all__:
            getattr(core, name)
        assert 'mpv' not in sys.modules, f'mpv loaded by core.__all__: {sorted(k for k in sys.modules if "mpv" in k)}'
        assert 'av' not in sys.modules, f'av loaded by core.__all__: {sorted(k for k in sys.modules if k == "av" or k.startswith("av."))}'
    """
    result = _run_in_clean_interpreter(script)
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_all_ui_all_names_resolve_via_lazy_getattr():
    """Every name in ``ui.__all__`` must resolve through ``__getattr__``."""
    import ui

    for name in ui.__all__:
        resolved = getattr(ui, name)
        assert resolved is not None, f"ui.{name} resolved to None"


def test_ui_lazy_attrs_do_not_load_native_media_runtimes():
    """Resolving every ``ui.__all__`` name must not pull mpv or PyAV in.

    ``MainWindow`` is imported as a class object; ``ui.video_player`` defers
    its python-mpv import until playback initializes. So the class-level
    resolve is safe — only constructing/showing the window would load mpv.
    """
    script = """
        import sys, ui
        for name in ui.__all__:
            getattr(ui, name)
        assert 'mpv' not in sys.modules, f'mpv loaded by ui.__all__: {sorted(k for k in sys.modules if "mpv" in k)}'
        assert 'av' not in sys.modules, f'av loaded by ui.__all__: {sorted(k for k in sys.modules if k == "av" or k.startswith("av."))}'
    """
    result = _run_in_clean_interpreter(script)
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_core_unknown_name_raises_attribute_error():
    """Sanity check: unknown names still raise ``AttributeError``."""
    import core

    try:
        core.DoesNotExist  # noqa: B018 - intentional attribute access
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError for unknown core attr")


def test_ui_unknown_name_raises_attribute_error():
    """Sanity check: unknown names still raise ``AttributeError``."""
    import ui

    try:
        ui.DoesNotExist  # noqa: B018 - intentional attribute access
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError for unknown ui attr")
