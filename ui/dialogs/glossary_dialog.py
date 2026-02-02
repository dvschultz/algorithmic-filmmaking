"""Film Language Glossary dialog.

Searchable glossary of cinematography terms with category filtering.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QTextBrowser,
    QDialogButtonBox,
    QSplitter,
    QWidget,
    QFrame,
)
from PySide6.QtCore import Qt

from ui.theme import theme, UISizes
from core.film_glossary import (
    FILM_GLOSSARY,
    GLOSSARY_CATEGORIES,
    search_glossary,
    get_terms_by_category,
)


class GlossaryDialog(QDialog):
    """Dialog displaying a searchable film terminology glossary.

    Features:
    - Search input to filter terms by name or definition
    - Category dropdown to filter by term category
    - Term list with clickable items
    - Definition panel showing selected term details
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Film Language Glossary")
        self.setMinimumSize(600, 500)
        self.resize(700, 550)

        self._setup_ui()
        self._populate_terms()

        # Connect to theme changes
        if theme().changed:
            theme().changed.connect(self._refresh_theme)

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        header_label = QLabel("Film Language Glossary")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header_label)

        description = QLabel(
            "Reference guide for cinematography terminology. "
            "Search or browse by category to learn about film language concepts."
        )
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {theme().text_secondary};")
        layout.addWidget(description)

        # Search and filter row
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(12)

        # Search input
        search_label = QLabel("Search:")
        filter_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search terms...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumHeight(UISizes.LINE_EDIT_MIN_HEIGHT)
        self.search_input.textChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.search_input, stretch=1)

        filter_layout.addSpacing(16)

        # Category filter
        category_label = QLabel("Category:")
        filter_layout.addWidget(category_label)

        self.category_combo = QComboBox()
        self.category_combo.setMinimumHeight(UISizes.COMBO_BOX_MIN_HEIGHT)
        self.category_combo.setMinimumWidth(150)
        self.category_combo.addItem("All Categories")
        for category in GLOSSARY_CATEGORIES:
            self.category_combo.addItem(category)
        self.category_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.category_combo)

        layout.addLayout(filter_layout)

        # Splitter for term list and definition
        splitter = QSplitter(Qt.Horizontal)

        # Term list (left side)
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)

        list_header = QLabel("Terms")
        list_header.setStyleSheet("font-weight: bold;")
        list_layout.addWidget(list_header)

        self.term_list = QListWidget()
        self.term_list.setMinimumWidth(200)
        self.term_list.currentItemChanged.connect(self._on_term_selected)
        list_layout.addWidget(self.term_list)

        self.result_count_label = QLabel("")
        self.result_count_label.setStyleSheet(f"color: {theme().text_muted}; font-size: 11px;")
        list_layout.addWidget(self.result_count_label)

        splitter.addWidget(list_container)

        # Definition panel (right side)
        definition_container = QFrame()
        definition_container.setFrameStyle(QFrame.StyledPanel)
        definition_container.setStyleSheet(
            f"QFrame {{ background-color: {theme().background_secondary}; "
            f"border: 1px solid {theme().border_secondary}; border-radius: 4px; }}"
        )
        definition_layout = QVBoxLayout(definition_container)
        definition_layout.setContentsMargins(12, 12, 12, 12)

        definition_header = QLabel("Definition")
        definition_header.setStyleSheet("font-weight: bold;")
        definition_layout.addWidget(definition_header)

        self.definition_display = QTextBrowser()
        self.definition_display.setOpenExternalLinks(False)
        self.definition_display.setStyleSheet(
            f"QTextBrowser {{ background-color: transparent; border: none; }}"
        )
        self.definition_display.setPlaceholderText("Select a term to see its definition.")
        definition_layout.addWidget(self.definition_display)

        splitter.addWidget(definition_container)

        # Set initial splitter sizes (40% list, 60% definition)
        splitter.setSizes([280, 420])

        layout.addWidget(splitter, stretch=1)

        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _populate_terms(self, filter_text: str = "", category: str = ""):
        """Populate the term list based on filters.

        Args:
            filter_text: Text to search for in term names/definitions
            category: Category to filter by (empty or "All Categories" for all)
        """
        self.term_list.clear()

        # Determine which terms to show
        if filter_text:
            # Search mode
            cat_filter = category if category and category != "All Categories" else None
            terms = search_glossary(filter_text, cat_filter)
        elif category and category != "All Categories":
            # Category filter only
            terms = get_terms_by_category(category)
        else:
            # Show all terms
            terms = [{"key": k, **v} for k, v in sorted(FILM_GLOSSARY.items(), key=lambda x: x[1]["name"])]

        # Add items to list
        for term_data in terms:
            item = QListWidgetItem(term_data["name"])
            item.setData(Qt.UserRole, term_data)
            self.term_list.addItem(item)

        # Update result count
        total = len(FILM_GLOSSARY)
        shown = len(terms)
        if shown == total:
            self.result_count_label.setText(f"{total} terms")
        else:
            self.result_count_label.setText(f"Showing {shown} of {total} terms")

        # Select first item if available
        if self.term_list.count() > 0:
            self.term_list.setCurrentRow(0)
        else:
            self._clear_definition()

    def _on_filter_changed(self):
        """Handle search or category filter change."""
        search_text = self.search_input.text().strip()
        category = self.category_combo.currentText()
        self._populate_terms(search_text, category)

    def _on_term_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle term selection in the list."""
        if current is None:
            self._clear_definition()
            return

        term_data = current.data(Qt.UserRole)
        if term_data:
            self._show_definition(term_data)

    def _show_definition(self, term_data: dict):
        """Display the definition for a term.

        Args:
            term_data: Dict with key, name, category, definition
        """
        html = f"""
        <h2 style="margin-top: 0; color: {theme().text_primary};">{term_data['name']}</h2>
        <p style="color: {theme().text_muted}; font-style: italic; margin-bottom: 16px;">
            {term_data['category']}
        </p>
        <p style="color: {theme().text_secondary}; font-size: 14px; line-height: 1.5;">
            {term_data['definition']}
        </p>
        <p style="color: {theme().text_muted}; font-size: 11px; margin-top: 24px;">
            Internal key: <code>{term_data['key']}</code>
        </p>
        """
        self.definition_display.setHtml(html)

    def _clear_definition(self):
        """Clear the definition display."""
        self.definition_display.setHtml(
            f'<p style="color: {theme().text_muted};">Select a term to see its definition.</p>'
        )

    def _refresh_theme(self):
        """Refresh styles when theme changes."""
        # Re-apply styles that depend on theme colors
        self.result_count_label.setStyleSheet(f"color: {theme().text_muted}; font-size: 11px;")

        # If a term is selected, refresh its display
        current = self.term_list.currentItem()
        if current:
            term_data = current.data(Qt.UserRole)
            if term_data:
                self._show_definition(term_data)
