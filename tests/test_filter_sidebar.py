"""Tests for FilterSidebar scaffold."""

import os

import pytest

from PySide6.QtWidgets import QLabel


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_constructs_with_all_nine_sections(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar, SECTIONS

    fs = FilterState()
    sidebar = FilterSidebar(fs)

    for key, _title in SECTIONS:
        assert sidebar.section(key) is not None, f"missing section {key}"
    assert len(SECTIONS) == 9


def test_binds_to_provided_filter_state(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar

    fs = FilterState()
    sidebar = FilterSidebar(fs)
    assert sidebar.filter_state is fs


def test_section_expanded_state_persistence(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar, SECTIONS

    fs = FilterState()
    # Provide initial expand state
    initial = {SECTIONS[0][0]: False, SECTIONS[1][0]: True}
    sidebar = FilterSidebar(fs, section_expanded_state=initial)

    assert sidebar.section(SECTIONS[0][0]).expanded is False
    assert sidebar.section(SECTIONS[1][0]).expanded is True

    # Missing keys default to True
    assert sidebar.section(SECTIONS[2][0]).expanded is True


def test_section_expanded_changed_emits(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar, SECTION_SHOT

    fs = FilterState()
    sidebar = FilterSidebar(fs)

    events: list[tuple[str, bool]] = []
    sidebar.section_expanded_changed.connect(
        lambda key, exp: events.append((key, exp))
    )

    sidebar.section(SECTION_SHOT).set_expanded(False)
    qapp.processEvents()

    assert events == [(SECTION_SHOT, False)]


def test_section_expanded_state_snapshot(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar, SECTION_SHOT, SECTION_VISUAL

    fs = FilterState()
    sidebar = FilterSidebar(fs)
    sidebar.section(SECTION_SHOT).set_expanded(False)
    sidebar.section(SECTION_VISUAL).set_expanded(False)

    snapshot = sidebar.section_expanded_state()
    assert snapshot[SECTION_SHOT] is False
    assert snapshot[SECTION_VISUAL] is False


def test_set_section_content_installs_widget(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar, SECTION_SHOT

    fs = FilterState()
    sidebar = FilterSidebar(fs)

    widget = QLabel("my content")
    sidebar.set_section_content(SECTION_SHOT, widget)
    # Content widget should now be a child of the section
    section = sidebar.section(SECTION_SHOT)
    assert widget.parent() is section._content_frame


def test_set_section_content_raises_for_unknown_key(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar

    fs = FilterState()
    sidebar = FilterSidebar(fs)

    with pytest.raises(KeyError):
        sidebar.set_section_content("nonexistent", QLabel("x"))


def test_visibility_requested_on_hide_click(qapp):
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar

    fs = FilterState()
    sidebar = FilterSidebar(fs)

    events: list[bool] = []
    sidebar.visibility_requested.connect(lambda v: events.append(v))

    # Find the hide button (single QPushButton in the header) and click it
    from PySide6.QtWidgets import QPushButton
    hide_btn = sidebar.findChild(QPushButton, "FilterSidebarHideBtn")
    assert hide_btn is not None
    hide_btn.click()
    qapp.processEvents()

    assert events == [False]


def test_shared_filter_state_between_two_sidebars(qapp):
    """Two sidebars pointing at one FilterState observe the same mutations."""
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar

    fs = FilterState()
    sidebar_a = FilterSidebar(fs)
    sidebar_b = FilterSidebar(fs)

    events_a: list[set] = []
    events_b: list[set] = []
    fs.changed.connect(lambda: events_a.append(set(fs.shot_type)))
    fs.changed.connect(lambda: events_b.append(set(fs.shot_type)))

    fs.shot_type = "Close-up"
    qapp.processEvents()

    assert events_a == [{"Close-up"}]
    assert events_b == [{"Close-up"}]
    assert sidebar_a.filter_state is sidebar_b.filter_state


def test_chip_toggle_updates_filter_state(qapp):
    """Toggling a chip in the sidebar writes through to FilterState."""
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar

    fs = FilterState()
    sidebar = FilterSidebar(fs)
    group = sidebar._chip_groups["shot_type"]
    # Toggle the first chip on
    first_value = next(iter(group._buttons.keys()))
    group._buttons[first_value].setChecked(True)
    qapp.processEvents()

    assert fs.shot_type == {first_value}


def test_state_change_updates_chip_selection(qapp):
    """Mutating FilterState updates the sidebar's chip selection (round-trip)."""
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar

    fs = FilterState()
    sidebar = FilterSidebar(fs)
    group = sidebar._chip_groups["shot_type"]
    first_value = next(iter(group._buttons.keys()))

    fs.shot_type = first_value
    qapp.processEvents()

    assert group.selected_values() == {first_value}


def test_state_sync_does_not_loop(qapp):
    """State→UI→state round-trip must not cause a cascade."""
    from core.filter_state import FilterState
    from ui.widgets.filter_sidebar import FilterSidebar

    fs = FilterState()
    sidebar = FilterSidebar(fs)

    emissions: list[int] = []
    fs.changed.connect(lambda: emissions.append(1))

    group = sidebar._chip_groups["gaze_filter"]
    first_value = next(iter(group._buttons.keys()))

    # Simulate user chip click
    group._buttons[first_value].setChecked(True)
    qapp.processEvents()

    # Only one emission expected
    assert len(emissions) == 1
