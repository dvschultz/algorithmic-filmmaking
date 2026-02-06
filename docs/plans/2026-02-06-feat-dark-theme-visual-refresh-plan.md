---
title: "feat: Dark Theme Visual Refresh with Gradient Accents"
type: feat
date: 2026-02-06
---

# Dark Theme Visual Refresh with Gradient Accents

## Overview

Transform Scene Ripper from a generic gray dark theme into a premium, cinematic aesthetic inspired by Frame.io and translucent dark gradient textures. **Dark mode first**, light mode updated for parity afterward. The refresh touches four layers:

1. **Foundation** — New navy-charcoal color palette, typography scale, spacing system, border-radius system
2. **Cards & Components** — Rounded corners, modern buttons/inputs, polished states
3. **Gradient Accents** — Translucent gradient glow effects on sorting cards and selected clip thumbnails
4. **Content-Aware Gradients** — Dynamic gradient colors derived from clips' dominant color palettes

## Problem Statement / Motivation

The current dark theme uses generic gray tones (`#1e1e1e`, `#2a2a2a`, `#3a3a3a`) that feel flat and utilitarian. Cards are boxy (0px border-radius from `QFrame.Panel|QFrame.Raised`), typography sizes are scattered across 84 call sites with no system, and spacing varies from 4px to 20px with no scale. The overall impression is "developer tool" rather than "creative professional tool."

Frame.io demonstrates that video tools can feel premium: deep navy-black backgrounds, electric blue/turquoise accents, generous spacing, soft rounded elements, and subtle depth through layered transparency. The translucent dark gradient reference image shows how soft diffused color blobs on dark backgrounds create depth and visual interest without overwhelming content.

## Proposed Solution

### Visual Direction

- **Default state:** Cool purple-blue gradient accents (like gradient #04/#14 from the reference)
- **Active/selected state:** Electric multicolor gradients (like #15/#16 — orange-teal-blue)
- **Gradient application:** Border glow effect on cards (not background fill) — cards keep solid dark backgrounds for text readability, with a soft diffused gradient glow around edges
- **Content-aware:** When clips have `dominant_colors` data, the gradient glow derives from those colors instead of the default purple-blue
- **Intensity:** Bold (60%+ opacity) — the gradient is a deliberate design statement

### Where Gradients Appear

| Surface | Gradient Behavior |
|---------|------------------|
| Sorting cards (7 cards in Sequence tab) | Always show gradient border glow; default cool purple-blue |
| Selected clip thumbnails | Gradient border glow replaces solid accent-blue selection border |
| Empty state backgrounds | Subtle gradient wash behind empty state text |
| Everything else | Solid colors from the new palette (no gradients) |

### Content-Aware Gradient Logic

Clips already store `dominant_colors: list[tuple[int, int, int]]` (RGB tuples) from the color analysis pipeline (`core/analysis/color.py`). When a clip has dominant colors:

1. Take the top 2-3 dominant colors from `clip.dominant_colors`
2. Create a radial gradient using those colors (softened/blurred in a `QRadialGradient` or conic gradient)
3. Apply as the border glow on that clip's thumbnail card when selected
4. Fall back to default cool purple-blue if no dominant colors are available

For sorting cards in the Sequence tab, the gradient could optionally reflect the dominant colors of the clips currently in the sequence (aggregate), but defaults to the static cool purple-blue.

## Technical Considerations

### Prerequisite: QFrame Style Migration

Three card widgets use `QFrame.Panel | QFrame.Raised` which draws a platform-native 3D frame that **ignores QSS border-radius**:
- `ui/widgets/sorting_card.py:49`
- `ui/clip_browser.py:95` (ClipThumbnail)
- `ui/source_thumbnail.py:28`

**Fix:** Change to `QFrame.NoFrame` and control all appearance via QSS + custom `paintEvent`. This is the single prerequisite for both rounded corners and gradient glows.

### Thumbnail Clipping with Border-Radius

When cards get rounded corners (8px+), the rectangular `QLabel` thumbnails (220x124px) will poke out of the card corners. QSS `border-radius` on a QLabel with a QPixmap does not clip the image.

**Fix:** Add a custom `paintEvent` override to the thumbnail QLabel that clips with `QPainterPath.addRoundedRect()` before drawing the pixmap. Only clip the top corners of the card (bottom of the thumbnail is inside the card body).

### Gradient Glow Implementation (QPainter)

QSS cannot render radial gradients or diffused glow effects. The gradient border glow must be implemented via custom `paintEvent` using QPainter:

```python
def paintEvent(self, event):
    painter = QPainter(self)
    painter.setRenderHint(QPainter.Antialiasing)

    # 1. Draw the gradient glow (slightly larger than the card)
    glow_rect = self.rect().adjusted(-4, -4, 4, 4)
    gradient = QRadialGradient(glow_rect.center(), glow_rect.width() / 2)
    gradient.setColorAt(0.0, QColor(color1_r, color1_g, color1_b, 160))  # 60%+ opacity
    gradient.setColorAt(0.5, QColor(color2_r, color2_g, color2_b, 100))
    gradient.setColorAt(1.0, QColor(0, 0, 0, 0))  # Fade to transparent
    painter.setBrush(QBrush(gradient))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(glow_rect, Radii.MD + 4, Radii.MD + 4)

    # 2. Draw the card body (solid dark background)
    card_rect = self.rect()
    painter.setBrush(QColor(theme().card_background))
    painter.setPen(QPen(QColor(theme().card_border), 1))
    painter.drawRoundedRect(card_rect, Radii.MD, Radii.MD)

    painter.end()
```

**Performance note:** QPainter gradient rendering is efficient for small numbers of widgets (7 sorting cards + a handful of selected clip thumbnails). The gradient is NOT applied to all 100+ clip cards in the grid — only selected ones. Non-selected clips use simple solid QSS borders.

### Glass-Morphism: Pragmatic Approach

True glass-morphism (backdrop-blur) is not supported in QSS. Use semi-transparent `rgba()` backgrounds on elevated surfaces (menus, tooltips, popovers) to approximate depth. This is cross-platform and performant.

### QPainter-Based Components (Timeline)

The timeline renders via `QPainter` using `theme().colors.qcolor()`, not QSS. New color tokens flow through automatically — no structural painting changes needed.

### Hardcoded rgba() Values

Six files contain hardcoded `rgba()` values outside the theme system that assume neutral gray backgrounds. These need to become theme tokens for navy backgrounds to work:
- Transcript overlay: `rgba(0, 0, 0, 0.85)` → new `overlay_dark` token
- Disabled badge: `rgba(170, 35, 35, 0.95)` → new `badge_disabled_bg` token
- Duration badge: `rgba(0, 0, 0, 0.8)` → reuse `overlay_dark`
- Success highlight: `rgba(76, 175, 80, 0.1)` → new `surface_success` token

### Existing Infrastructure to Build On

- **`Clip.dominant_colors`** (`models/clip.py:172`) — Already stores RGB tuples from color analysis
- **`ColorSwatchBar`** (`ui/clip_browser.py:38-74`) — Already renders dominant colors as painted stripes
- **`theme().changed` signal** — All 34 files already connect to theme changes; gradient colors will update automatically
- **`_update_style()` pattern** — All card widgets already have state-driven style update methods

## Acceptance Criteria

### Visual

- [x] Dark theme backgrounds shift from neutral gray to deep navy-charcoal
- [x] Primary accent color is a vibrant electric blue (`#5b8def`) replacing muted `#4a90d9`
- [x] All cards (SortingCard, ClipThumbnail, SourceThumbnail) have rounded corners (8px)
- [x] Thumbnails are clipped to match card border-radius (top corners)
- [x] **Sorting cards show gradient border glow** — cool purple-blue by default
- [x] **Selected sorting card shows electric multicolor gradient glow**
- [x] **Selected clip thumbnails show gradient border glow** — uses clip's dominant colors if available, otherwise default purple-blue
- [x] Non-selected clip thumbnails use solid borders (no gradient, no performance impact)
- [x] Gradient glow is bold (60%+ opacity) — clearly visible, not subtle
- [x] Buttons feel modern: slightly taller, more rounded, subtle hover transitions
- [x] Tab bar uses the new accent for the active indicator
- [x] Typography is consistent: all sizes come from TypeScale constants
- [x] Spacing is consistent: all padding/margins come from Spacing constants
- [x] Chat bubbles coordinate with the new palette
- [x] Dialogs look polished (all 5)
- [x] Empty states have a subtle gradient wash backdrop

### Functional

- [x] Theme toggle (dark/light/system) still works correctly
- [x] All widget states render correctly: normal, hover, selected, disabled, focus (keyboard)
- [x] Scrolling 100+ clips in ClipBrowser is smooth (gradients only on selected cards)
- [x] Text remains legible on all surfaces (WCAG AA: 4.5:1 for normal text)
- [x] Focus indicators remain clearly visible for keyboard navigation
- [x] Light theme remains functional (not broken) though less polished
- [x] Gradient gracefully falls back to default colors when `dominant_colors` is None

### Code Quality

- [x] All new values use theme tokens — no new hardcoded colors
- [x] TypeScale, Spacing, and Radii constants added to `ui/theme.py`
- [x] `QFrame.Panel|QFrame.Raised` removed from all three card widgets
- [x] Gradient rendering is isolated in a reusable `GradientGlowMixin` or helper
- [x] Existing tests pass (376 tests)

## Implementation Plan

### Phase 1: Design Tokens & Foundation

**Add new constants to `ui/theme.py`.**

#### New Dark Theme Color Palette

```python
DARK_THEME = ThemeColors(
    # Backgrounds — deep navy-charcoal instead of neutral gray
    background_primary="#0d0f14",       # Near-black with blue undertone
    background_secondary="#151820",     # Card/panel surfaces
    background_tertiary="#1c2030",      # Nested elements, hover states
    background_elevated="#252a3a",      # Menus, tooltips, popovers

    # Text — slightly warm white, not pure #ffffff
    text_primary="#e8eaf0",             # Main text (softer than pure white)
    text_secondary="#8b92a8",           # Labels, secondary info
    text_muted="#525a72",               # Disabled, hints (WCAG: verify ≥4.5:1)
    text_inverted="#ffffff",            # On accent backgrounds

    # Borders — subtle, blends with navy
    border_primary="#2a3045",           # Main borders
    border_secondary="#1e2436",         # Subtle borders
    border_focus="#5b8def",             # Focus rings

    # Accents — electric blue family
    accent_blue="#5b8def",              # Primary accent
    accent_blue_hover="#7ba3f7",        # Blue hover
    accent_red="#f04e5e",               # Errors, destructive
    accent_green="#3ecf6e",             # Success
    accent_orange="#f0a030",            # Warning
    accent_purple="#a87edb",            # Special highlights

    # Timeline
    timeline_background="#0d0f14",
    timeline_ruler="#151820",
    timeline_ruler_border="#2a3045",
    timeline_ruler_tick="#525a72",
    timeline_ruler_tick_minor="#2a3045",
    timeline_track="#131620",
    timeline_track_highlight="#1c2545",
    timeline_clip="#4a7de0",
    timeline_clip_selected="#5b8def",
    timeline_clip_border="#3a65c0",
    timeline_clip_selected_border="#e8eaf0",

    # Components
    thumbnail_background="#1a1d28",
    card_background="#151820",
    card_border="#2a3045",
    card_hover="#1c2030",
    badge_analyzed="#3ecf6e",
    badge_not_analyzed="#525a72",
    shot_type_badge="#2a3045",
    surface_highlight="#1c2545",

    # Chat
    chat_user_bubble="#5b8def",
    chat_assistant_bubble="#1c2030",
    chat_user_text="#ffffff",
    chat_assistant_text="#e8eaf0",
    plan_pending_bg="#151820",
    plan_pending_border="#2a3045",
    plan_running_bg="#14203a",
    plan_running_border="#5b8def",
    plan_completed_bg="#0f2a1a",
    plan_completed_border="#3ecf6e",
    plan_failed_bg="#2a1015",
    plan_failed_border="#f04e5e",

    # New overlay tokens
    overlay_dark="rgba(5, 5, 15, 0.85)",
    overlay_medium="rgba(5, 5, 15, 0.65)",
    surface_success="rgba(62, 207, 110, 0.1)",
    surface_error="rgba(240, 78, 94, 0.1)",
    badge_disabled_bg="rgba(170, 35, 35, 0.95)",
)
```

#### Gradient Palette (new in `theme.py`)

```python
@dataclass
class GradientPalette:
    """Default gradient colors for glow effects."""
    # Default cool purple-blue (non-selected sorting cards, fallback)
    default_color_1: str    # "#6366f1" — indigo
    default_color_2: str    # "#8b5cf6" — violet
    default_color_3: str    # "#3b82f6" — blue

    # Electric multicolor (selected/active state)
    active_color_1: str     # "#f97316" — orange
    active_color_2: str     # "#06b6d4" — cyan
    active_color_3: str     # "#5b8def" — electric blue

    # Glow intensity
    default_opacity: int    # 160 (out of 255, ~63%)
    active_opacity: int     # 200 (out of 255, ~78%)
    glow_spread: int        # 6 pixels beyond card edge
```

#### Typography Scale

```python
class TypeScale:
    """Consistent font sizes. Use for QFont.setPointSize() and QSS font-size."""
    XS = 9       # Badges, timestamps, fine print
    SM = 11      # Secondary labels, captions, metadata
    BASE = 13    # Body text, form inputs, default
    MD = 14      # Card titles, section labels
    LG = 16      # Dialog headers, prominent labels
    XL = 18      # Section headers, empty state titles
    XXL = 24     # Hero text, main empty state titles
    XXXL = 32    # Splash/about (rare)
```

#### Spacing Scale

```python
class Spacing:
    """Consistent spacing for padding and margins."""
    XXS = 2      # Tight: between badge elements
    XS = 4       # Compact: icon-text gaps
    SM = 8       # Default: between related elements
    MD = 12      # Comfortable: form field spacing
    LG = 16      # Section: card padding, panel margins
    XL = 24      # Group: grid gutter, section breaks
    XXL = 32     # Page: major section separators
    XXXL = 48    # Hero: empty state breathing room
```

#### Border-Radius Scale

```python
class Radii:
    """Consistent border-radius values."""
    SM = 4       # Badges, checkboxes, progress bars
    MD = 8       # Cards, buttons, inputs
    LG = 12      # Chat bubbles, dialogs
    XL = 16      # Featured cards, modals
    FULL = 9999  # Pill shapes
```

**Files modified:**
- `ui/theme.py` — add ThemeColors overlay fields, GradientPalette, TypeScale, Spacing, Radii; update DARK_THEME values

---

### Phase 2: QFrame Migration & Card Foundation

Fix the `QFrame.Panel|QFrame.Raised` blocker and establish rounded card rendering.

#### 2a. SortingCard (`ui/widgets/sorting_card.py`)

- Change `setFrameStyle(QFrame.Panel | QFrame.Raised)` → `setFrameStyle(QFrame.NoFrame)`
- Override `paintEvent()` to:
  1. Draw gradient glow (extends 6px beyond card edges)
  2. Draw solid card body with `Radii.MD` corners
  3. Let child widgets render on top
- Selected state: switch from default gradient palette to active (electric multicolor)
- Disabled state: no gradient glow, muted solid border

#### 2b. ClipThumbnail (`ui/clip_browser.py`)

- Change `setFrameStyle(QFrame.Panel | QFrame.Raised)` → `setFrameStyle(QFrame.NoFrame)`
- Override `paintEvent()` to draw card with `Radii.MD` rounded corners
- **Selected state only:** draw gradient glow using `clip.dominant_colors` if available, else default purple-blue
- Non-selected state: simple solid border via QSS (no QPainter gradient — performance)
- Add thumbnail QLabel `paintEvent` override for top-corner clipping via `QPainterPath.addRoundedRect()`

#### 2c. SourceThumbnail (`ui/source_thumbnail.py`)

- Same QFrame migration as ClipThumbnail
- Same thumbnail clipping approach
- No gradient glow (source cards are simpler — keep solid borders)

#### 2d. Gradient Glow Helper

Create a reusable helper (mixin or function) for gradient glow rendering:

```python
# ui/gradient_glow.py

def paint_gradient_glow(
    painter: QPainter,
    rect: QRect,
    colors: list[tuple[int, int, int]],
    opacity: int = 160,
    spread: int = 6,
    radius: int = 8,
):
    """Paint a soft radial gradient glow around a card rectangle.

    Args:
        painter: Active QPainter
        rect: Card body rectangle
        colors: 2-3 RGB tuples for gradient stops
        opacity: Alpha (0-255) for glow intensity
        spread: Pixels beyond card edge
        radius: Border-radius matching the card
    """
    glow_rect = rect.adjusted(-spread, -spread, spread, spread)
    center = glow_rect.center()
    gradient = QRadialGradient(center, max(glow_rect.width(), glow_rect.height()) / 2)

    # Use up to 3 colors with decreasing opacity
    for i, (r, g, b) in enumerate(colors[:3]):
        stop = i / max(len(colors[:3]) - 1, 1) * 0.7  # Spread across 0.0-0.7
        gradient.setColorAt(stop, QColor(r, g, b, opacity))
    gradient.setColorAt(1.0, QColor(0, 0, 0, 0))  # Fade to transparent

    painter.setBrush(QBrush(gradient))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(glow_rect, radius + spread, radius + spread)
```

**Files modified:**
- `ui/widgets/sorting_card.py`
- `ui/clip_browser.py`
- `ui/source_thumbnail.py`
- `ui/gradient_glow.py` (new)

---

### Phase 3: Global Stylesheet Update

Update `get_app_stylesheet()` in `ui/theme.py` (lines 367-686) to use new tokens:

- **Tab bar:** Padding `{Spacing.SM}px {Spacing.LG}px`, `Radii.MD` top corners, accent underline
- **Buttons:** `border-radius: {Radii.MD}px`, padding `{Spacing.SM}px {Spacing.LG}px`, min-height `32px` (up from 28)
- **Inputs/Combos:** `border-radius: {Radii.MD}px`, slightly taller
- **Checkboxes:** `border-radius: {Radii.SM}px`, 18x18 (up from 16x16)
- **Sliders:** Handle `border-radius: {Radii.MD}px`
- **Scrollbars:** Thinner (10px), more transparent, `border-radius: 5px`
- **Group boxes:** `border-radius: {Radii.MD}px`
- **Progress bars:** `border-radius: {Radii.MD}px`
- **Tooltips:** `border-radius: {Radii.MD}px`, semi-transparent `background_elevated`
- **Menus:** `border-radius: {Radii.LG}px`, semi-transparent elevated background

Also update `UISizes`:
- `BUTTON_MIN_HEIGHT = 32` (up from 28)
- `COMBO_BOX_MIN_HEIGHT = 32` (up from 28)
- `LINE_EDIT_MIN_HEIGHT = 32` (up from 28)

**Files modified:**
- `ui/theme.py` (get_app_stylesheet method, UISizes class)

---

### Phase 4: Component-Level Inline Style Updates

Update ~244 `setStyleSheet()` calls across ~34 files. Prioritize by visibility:

**Tier 1 — Highest visibility (first):**

| File | Key Updates |
|------|------------|
| `ui/clip_browser.py` | ClipThumbnail states, ColorSwatchBar, filter panel, badges |
| `ui/widgets/sorting_card.py` | Card states with gradient glow integration |
| `ui/source_thumbnail.py` | Source card states, analyzed badge |
| `ui/tabs/analyze_tab.py` | Analysis controls, combo boxes |
| `ui/tabs/cut_tab.py` | Cut controls, button styling |
| `ui/tabs/sequence_tab.py` | Sequence controls, sorting card grid backdrop |

**Tier 2 — Important surfaces:**

| File | Key Updates |
|------|------------|
| `ui/chat_widgets.py` | Bubble colors, plan step states, input area |
| `ui/chat_panel.py` | Panel container background |
| `ui/source_browser.py` | Source sidebar |
| `ui/widgets/source_group_header.py` | Collapsible headers |
| `ui/video_player.py` | Player controls |

**Tier 3 — Dialogs:**

| File | Key Updates |
|------|------------|
| `ui/dialogs/analysis_picker_dialog.py` | Phase labels, checkboxes, buttons |
| `ui/dialogs/exquisite_corpus_dialog.py` | Form fields, inline styles |
| `ui/dialogs/storyteller_dialog.py` | Form fields, inline styles |
| `ui/dialogs/glossary_dialog.py` | QTextBrowser styling |
| `ui/dialogs/intention_import_dialog.py` | Import controls |

**Tier 4 — Settings & remaining:**

| File | Key Updates |
|------|------------|
| `ui/settings_dialog.py` | PathSelector, group boxes |
| `ui/tabs/collect_tab.py` | Collection controls |
| `ui/tabs/render_tab.py` | Export controls |
| `ui/tabs/generate_tab.py` | Generation controls |

**For each file:**
1. Replace hardcoded font sizes → `TypeScale.XX`
2. Replace hardcoded padding/margins → `Spacing.XX`
3. Replace hardcoded border-radius → `Radii.XX`
4. Replace hardcoded `rgba()` → theme overlay tokens
5. Verify all widget states (normal/hover/selected/disabled/focus)

**Files modified:** ~34 files (see tier list above)

---

### Phase 5: Content-Aware Gradient Integration

Wire up `Clip.dominant_colors` to the gradient glow system:

#### 5a. ClipThumbnail Selected State

In `ClipThumbnail._update_style()`, when the clip is selected:

```python
def _update_style(self):
    if self._selected:
        # Get gradient colors from clip data
        if self._clip and self._clip.dominant_colors:
            glow_colors = self._clip.dominant_colors[:3]
        else:
            glow_colors = theme().gradient.default_colors
        self._glow_colors = glow_colors
        self._show_glow = True
    else:
        self._show_glow = False
    self.update()  # Triggers paintEvent
```

#### 5b. SortingCard Gradient

Sorting cards always show glow (not content-dependent by default):

- **Normal state:** Default cool purple-blue gradient from `GradientPalette`
- **Selected state:** Electric multicolor from `GradientPalette.active_*`
- **Disabled state:** No glow
- **Hover state:** Slightly brighter version of the normal gradient

#### 5c. Empty State Gradient Wash

Add a subtle gradient wash behind empty state widgets:

```python
# In EmptyStateWidget.paintEvent:
gradient = QLinearGradient(0, 0, self.width(), self.height())
gradient.setColorAt(0.0, QColor(99, 102, 241, 30))   # Very faint indigo
gradient.setColorAt(0.5, QColor(139, 92, 246, 20))    # Very faint violet
gradient.setColorAt(1.0, QColor(59, 130, 246, 25))    # Very faint blue
painter.fillRect(self.rect(), QBrush(gradient))
```

**Files modified:**
- `ui/clip_browser.py` (ClipThumbnail gradient integration)
- `ui/widgets/sorting_card.py` (gradient state management)
- `ui/widgets/empty_state.py` (gradient wash backdrop)

---

### Phase 6: Polish & Light Theme

#### 6a. Empty State & Badge Polish

- Verify badge colors have sufficient contrast on navy backgrounds
- Update `badge_not_analyzed` to be visible on `card_background`
- Ensure `shot_type_badge` text is readable
- Review all tooltip text

#### 6b. Light Theme Parity

Update `LIGHT_THEME` to maintain functional parity:
- Cool-white backgrounds (not pure gray): `#fafbfe`, `#f0f2f8`, `#e4e8f0`
- Same accent blue family as dark theme
- Gradient glows use lower opacity in light mode (40% instead of 60%)
- Verify all text contrast ratios meet WCAG AA (4.5:1)
- Test all widget states

#### 6c. Contrast Verification

Check these critical combinations:
- `text_muted` (#525a72) on `background_primary` (#0d0f14) — needs ≥4.5:1
- `text_secondary` (#8b92a8) on `background_primary` (#0d0f14) — needs ≥4.5:1
- `text_inverted` (#ffffff) on `accent_blue` (#5b8def) — needs ≥4.5:1
- Badge text on `badge_analyzed` (#3ecf6e) — needs ≥4.5:1

**Files modified:**
- `ui/theme.py` (LIGHT_THEME values, contrast adjustments)
- `ui/widgets/empty_state.py`
- `ui/tabs/base_tab.py`

---

## Visual QA Checklist

Since there are no visual regression tests, manually verify:

### Per-Tab Checks
- [ ] **Collect tab:** Source browser, YouTube search panel, drag-drop target
- [ ] **Cut tab:** Clip grid with 50+ clips, all card states, filter panel, sensitivity slider
- [ ] **Analyze tab:** Analysis picker dialog, progress indicators, empty state
- [ ] **Sequence tab:** Sorting card grid (7 cards with gradient glow), timeline, player, preview strip
- [ ] **Render tab:** Export controls, progress bar
- [ ] **Generate tab:** Controls and output area

### Gradient-Specific Checks
- [ ] All 7 sorting cards show cool purple-blue gradient glow in normal state
- [ ] Selected sorting card shows electric multicolor gradient glow
- [ ] Disabled sorting cards have no gradient glow
- [ ] Selected clip thumbnail shows gradient glow with clip's dominant colors
- [ ] Selected clip without dominant_colors shows default purple-blue gradient glow
- [ ] Non-selected clips have NO gradient (solid borders only)
- [ ] Empty states show subtle gradient wash
- [ ] Gradient glow is clearly visible (60%+ opacity) — bold, not subtle

### Per-Component State Checks
- [ ] Card states: normal → hover → selected → disabled → focus (Tab key)
- [ ] Button states: normal → hover → pressed → disabled
- [ ] Input states: empty → focused → filled → disabled
- [ ] Checkbox states: unchecked → checked → disabled
- [ ] Scrollbar: visible when scrolling, smooth appearance

### Cross-Cutting Checks
- [ ] Text legibility: `text_muted` on `background_primary` (≥ 4.5:1 contrast)
- [ ] Badge legibility: badge text on accent backgrounds
- [ ] Chat panel: user/assistant bubbles, plan steps (all 4 states)
- [ ] All 5 dialogs open and look correct
- [ ] Theme toggle: dark → light → system all work
- [ ] Scroll performance: 100+ clips in grid, smooth scroll (gradients only on selection)

## Dependencies & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| QFrame migration breaks card sizing | Cards render with wrong dimensions | Test each card widget independently after migration |
| Thumbnail clipping adds paint overhead | Scroll performance regression | Profile before/after; only clip on card classes, not all QLabels |
| Gradient glow on 100+ selected cards | Performance regression | Only apply gradient to selected cards (typically 1-20 at a time) |
| Navy backgrounds make muted text illegible | Accessibility regression | Check WCAG contrast ratios; adjust `text_muted` if needed |
| Partial migration looks worse than current | Visual inconsistency | Do Phases 1-3 in one commit, then Phase 4 per-tier |
| 34 files with inline styles = merge conflicts | Parallel work conflicts | Do refresh on a dedicated branch |
| `dominant_colors` is None on many clips | Gradient falls back ungracefully | Always fall back to default GradientPalette |
| Light theme gradient opacity needs tuning | Glows too bright/invisible in light mode | Use separate opacity values for light vs dark |

## Alternative Approaches Considered

| Approach | Why Rejected |
|----------|-------------|
| Gradient as card background fill | Reduces text readability; border glow preserves solid dark background for content |
| Gradients on ALL card states (not just selected) | Performance concern with 100+ cards; visual noise |
| True glass-morphism (blur) | QSS doesn't support backdrop-filter; platform-specific APIs add complexity |
| Bundled font (Inter, SF Pro) | Licensing complexity; system fonts work well across platforms |
| Gradients via QSS `qlineargradient()` | QSS gradients are linear only, verbose syntax, can't do radial/glow effects |
| Per-tab gradient identity colors | Adds complexity, reduces content-awareness; clips' own colors are more meaningful |

## References

### Internal
- Theme system: `ui/theme.py` (full file, 718 lines)
- Card styling: `ui/widgets/sorting_card.py:99-140`
- Clip thumbnail: `ui/clip_browser.py:77-330`
- ColorSwatchBar: `ui/clip_browser.py:38-74`
- Clip model dominant_colors: `models/clip.py:172`
- Color analysis: `core/analysis/color.py`
- Global stylesheet: `ui/theme.py:367-686`
- Existing sequence tab redesign plan: `docs/plans/2026-01-28-feat-sequence-tab-ui-redesign-plan.md`

### External
- Frame.io design inspiration: https://frame.io/
- Translucent dark gradient reference: https://framerusercontent.com/images/OZkxWMWl3TXCHVL5TEnYDCRD4.jpg
- Qt6 QSS reference: https://doc.qt.io/qt-6/stylesheet-reference.html
- Qt6 QPainter gradients: https://doc.qt.io/qt-6/qradialgradient.html
- WCAG contrast checker: https://webaim.org/resources/contrastchecker/
