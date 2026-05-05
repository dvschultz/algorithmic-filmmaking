# Clip Browser Model/View Virtualization Plan

## Problem

Cut and Analyze both use `ui.clip_browser.ClipBrowser`. The current virtual mode keeps clip data separate from widgets, but it still realizes cards as a `QGridLayout` full of `ClipThumbnail` widgets. Scrolling large projects can still block the UI because row rebuilds run on the main thread, card widgets are created or removed during scroll, and thumbnails can be decoded and scaled repeatedly.

## Best Fix

Replace the widget grid with a Qt model/view implementation:

- Store clip/source rows in a `QAbstractListModel` or `QAbstractTableModel`.
- Render cards with a `QStyledItemDelegate` instead of one QWidget tree per card.
- Use a `QListView` or `QTableView` configured for icon/grid layout, uniform item sizes, and batched layout.
- Keep selection, drag, context menu, and details/export signals as browser-level APIs so Cut and Analyze do not need to know about the backing view.
- Keep filtering/sorting in a proxy model or in a dedicated data index, not in paint or scroll handlers.
- Use an LRU thumbnail pixmap cache shared by the delegate.

This removes most QWidget allocation, layout churn, stylesheet recalculation, and per-scroll object deletion. Qt item views are designed to scroll thousands of rows by painting visible items, which is the behavior the current custom virtual grid is approximating.

## Migration Shape

1. Introduce `ClipBrowserModel` with roles for clip id, source id, thumbnail path, duration, badges, disabled state, and selected/hover affordances.
2. Introduce `ClipCardDelegate` to paint the thumbnail, duration, badges, selected glow, disabled dimming, and optional transcript/analysis indicators.
3. Build `ModelViewClipBrowser` behind the existing `ClipBrowser` public methods.
4. Port selection, drag, double-click, context menu, source grouping, and filters.
5. Switch Cut and Analyze to the model/view browser after parity tests pass.

## Interim Patch

Until the model/view rewrite lands, the current widget grid should avoid work during scroll:

- Cache virtual display rows and invalidate them only when data, filters, sort, expansion, or column count changes.
- Debounce scroll-driven realization to roughly one frame.
- Reuse a bounded set of virtual `ClipThumbnail` widgets.
- Cache scaled thumbnail pixmaps by path, mtime, and target size.

These changes reduce pinwheels for 500+ clips while keeping the existing browser API intact.
