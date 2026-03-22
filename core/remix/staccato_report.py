"""Staccato debug report — interactive HTML visualization of algorithm decisions.

Generates a self-contained HTML file with Chart.js charts showing:
- Onset strength at each beat slot
- Cosine distance between consecutive clips vs target distance
- Per-slot metadata on hover
"""

import json
import logging
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.remix.staccato import StaccatoDebugInfo

logger = logging.getLogger(__name__)

_VENDOR_DIR = Path(__file__).parent / "vendor"


def _read_vendor_file(filename: str) -> str:
    """Read a vendored JS file, returning empty string if missing."""
    path = _VENDOR_DIR / filename
    if not path.exists():
        logger.warning(f"Vendor file missing: {path}")
        return ""
    return path.read_text(encoding="utf-8")


def _debug_to_json(debug: "StaccatoDebugInfo") -> str:
    """Serialize debug info to JSON for embedding in HTML."""
    slots_data = []
    for s in debug.slots:
        slots_data.append({
            "index": s.slot_index,
            "startTime": round(s.start_time, 3),
            "endTime": round(s.end_time, 3),
            "duration": round(s.end_time - s.start_time, 3),
            "onsetStrength": round(s.onset_strength, 4),
            "clipId": s.clip_id[:8],
            "clipName": s.clip_name,
            "sourceFile": s.source_filename,
            "cosineDistance": round(s.cosine_distance, 4) if s.cosine_distance is not None else None,
            "targetDistance": round(s.target_distance, 4),
            "distanceScore": round(s.distance_score, 4),
            "needsLoop": s.needs_loop,
        })

    return json.dumps({
        "strategy": debug.strategy,
        "totalSlots": debug.total_slots,
        "totalClips": debug.total_clips_available,
        "slots": slots_data,
    }, indent=None)


def generate_staccato_report(debug: "StaccatoDebugInfo") -> str:
    """Generate a self-contained HTML debug report.

    Args:
        debug: StaccatoDebugInfo from a Staccato generation run

    Returns:
        Complete HTML string
    """
    chart_js = _read_vendor_file("chart.umd.min.js")
    hammer_js = _read_vendor_file("hammer.min.js")
    zoom_js = _read_vendor_file("chartjs-plugin-zoom.min.js")
    data_json = _debug_to_json(debug)

    has_charts = bool(chart_js)

    # Build the chart section or fallback table
    if has_charts:
        content_section = _build_chart_section()
        scripts_section = _build_chart_scripts(chart_js, hammer_js, zoom_js, data_json)
    else:
        content_section = _build_fallback_table(debug)
        scripts_section = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Staccato Debug Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>Staccato Debug Report</h1>
  <div class="stats">
    <span class="stat">Strategy: <strong>{debug.strategy}</strong></span>
    <span class="stat">Slots: <strong>{debug.total_slots}</strong></span>
    <span class="stat">Clips available: <strong>{debug.total_clips_available}</strong></span>
  </div>
  {content_section}
  <div class="legend">
    <p><strong>How to read:</strong> Each slot is a beat interval in the music.
    Onset strength (blue bars) drives clip selection — stronger onsets target
    clips with greater visual distance from the previous clip.
    The bottom chart shows how well the actual cosine distance (orange)
    matched the target distance (blue dashed).</p>
    <p><strong>Controls:</strong> Scroll to zoom, drag to pan, double-click to reset.</p>
  </div>
</div>
{scripts_section}
</body>
</html>"""


def _build_chart_section() -> str:
    """Build the HTML for the chart canvases."""
    return """
  <div class="chart-section">
    <h2>Onset Strength & Clip Assignments</h2>
    <div class="chart-wrapper">
      <canvas id="onsetChart"></canvas>
    </div>
  </div>
  <div class="chart-section">
    <h2>Visual Distance: Actual vs Target</h2>
    <div class="chart-wrapper">
      <canvas id="distanceChart"></canvas>
    </div>
  </div>
  <div class="chart-section">
    <h2>Slot Details</h2>
    <div id="detailTable"></div>
  </div>"""


def _build_chart_scripts(chart_js: str, hammer_js: str, zoom_js: str, data_json: str) -> str:
    """Build the script block with Chart.js and chart initialization."""
    return f"""
<script>{chart_js}</script>
<script>{hammer_js}</script>
<script>{zoom_js}</script>
<script>
const DATA = {data_json};

const zoomOpts = {{
  zoom: {{
    wheel: {{ enabled: true }},
    pinch: {{ enabled: true }},
    mode: 'x',
  }},
  pan: {{
    enabled: true,
    mode: 'x',
  }},
}};

// --- Onset Strength Chart ---
const onsetCtx = document.getElementById('onsetChart').getContext('2d');
const slotLabels = DATA.slots.map(s =>
  s.startTime.toFixed(1) + 's'
);
const clipLabels = DATA.slots.map(s => s.clipName);

new Chart(onsetCtx, {{
  type: 'bar',
  data: {{
    labels: slotLabels,
    datasets: [{{
      label: 'Onset Strength',
      data: DATA.slots.map(s => s.onsetStrength),
      backgroundColor: 'rgba(59, 130, 246, 0.7)',
      borderColor: 'rgba(59, 130, 246, 1)',
      borderWidth: 1,
    }}],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      zoom: zoomOpts,
      tooltip: {{
        callbacks: {{
          title: (items) => {{
            const i = items[0].dataIndex;
            const s = DATA.slots[i];
            return s.clipName + ' (' + s.startTime.toFixed(2) + 's - ' + s.endTime.toFixed(2) + 's)';
          }},
          afterBody: (items) => {{
            const i = items[0].dataIndex;
            const s = DATA.slots[i];
            const lines = [
              'Source: ' + s.sourceFile,
              'Onset: ' + s.onsetStrength.toFixed(3),
            ];
            if (s.cosineDistance !== null) {{
              lines.push('Cosine dist: ' + s.cosineDistance.toFixed(3));
              lines.push('Score: ' + s.distanceScore.toFixed(3));
            }}
            if (s.needsLoop) lines.push('Needs loop');
            return lines;
          }},
        }},
      }},
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Time (slot start)', color: '#94a3b8' }},
        ticks: {{ color: '#94a3b8', maxRotation: 45 }},
        grid: {{ color: 'rgba(148, 163, 184, 0.1)' }},
      }},
      y: {{
        title: {{ display: true, text: 'Onset Strength', color: '#94a3b8' }},
        min: 0, max: 1,
        ticks: {{ color: '#94a3b8' }},
        grid: {{ color: 'rgba(148, 163, 184, 0.1)' }},
      }},
    }},
  }},
}});

// --- Distance Chart ---
const distCtx = document.getElementById('distanceChart').getContext('2d');
// Skip first slot (no previous clip)
const distSlots = DATA.slots.filter(s => s.cosineDistance !== null);
const distLabels = distSlots.map(s => s.startTime.toFixed(1) + 's');

new Chart(distCtx, {{
  type: 'line',
  data: {{
    labels: distLabels,
    datasets: [
      {{
        label: 'Target Distance (onset strength)',
        data: distSlots.map(s => s.targetDistance),
        borderColor: 'rgba(59, 130, 246, 0.6)',
        borderDash: [5, 5],
        pointRadius: 3,
        pointBackgroundColor: 'rgba(59, 130, 246, 0.8)',
        fill: false,
        tension: 0.1,
      }},
      {{
        label: 'Actual Cosine Distance',
        data: distSlots.map(s => s.cosineDistance),
        borderColor: 'rgba(249, 115, 22, 1)',
        pointRadius: 4,
        pointBackgroundColor: 'rgba(249, 115, 22, 0.9)',
        fill: false,
        tension: 0.1,
      }},
    ],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      zoom: zoomOpts,
      tooltip: {{
        callbacks: {{
          title: (items) => {{
            const i = items[0].dataIndex;
            const s = distSlots[i];
            return s.clipName + ' (' + s.startTime.toFixed(2) + 's)';
          }},
          afterBody: (items) => {{
            const i = items[0].dataIndex;
            const s = distSlots[i];
            return [
              'Target: ' + s.targetDistance.toFixed(3),
              'Actual: ' + (s.cosineDistance !== null ? s.cosineDistance.toFixed(3) : 'n/a'),
              'Score: ' + s.distanceScore.toFixed(3),
            ];
          }},
        }},
      }},
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Time (slot start)', color: '#94a3b8' }},
        ticks: {{ color: '#94a3b8', maxRotation: 45 }},
        grid: {{ color: 'rgba(148, 163, 184, 0.1)' }},
      }},
      y: {{
        title: {{ display: true, text: 'Cosine Distance', color: '#94a3b8' }},
        min: 0,
        ticks: {{ color: '#94a3b8' }},
        grid: {{ color: 'rgba(148, 163, 184, 0.1)' }},
      }},
    }},
  }},
}});

// --- Detail Table ---
const esc = (t='') => t.replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const tableDiv = document.getElementById('detailTable');
let html = '<table><thead><tr>';
html += '<th>#</th><th>Time</th><th>Duration</th><th>Onset</th>';
html += '<th>Clip</th><th>Source</th><th>Cos Dist</th><th>Target</th><th>Score</th><th>Loop</th>';
html += '</tr></thead><tbody>';
DATA.slots.forEach((s, i) => {{
  html += '<tr>';
  html += '<td>' + (i + 1) + '</td>';
  html += '<td>' + s.startTime.toFixed(2) + 's</td>';
  html += '<td>' + s.duration.toFixed(2) + 's</td>';
  html += '<td>' + s.onsetStrength.toFixed(3) + '</td>';
  html += '<td>' + esc(s.clipName) + '</td>';
  html += '<td>' + esc(s.sourceFile) + '</td>';
  html += '<td>' + (s.cosineDistance !== null ? s.cosineDistance.toFixed(3) : '—') + '</td>';
  html += '<td>' + s.targetDistance.toFixed(3) + '</td>';
  html += '<td>' + s.distanceScore.toFixed(3) + '</td>';
  html += '<td>' + (s.needsLoop ? 'Yes' : '') + '</td>';
  html += '</tr>';
}});
html += '</tbody></table>';
tableDiv.innerHTML = html;
</script>"""


def _build_fallback_table(debug: "StaccatoDebugInfo") -> str:
    """Build a plain HTML table when Chart.js is unavailable."""
    rows = []
    for s in debug.slots:
        cos_dist = f"{s.cosine_distance:.3f}" if s.cosine_distance is not None else "—"
        loop = "Yes" if s.needs_loop else ""
        rows.append(
            f"<tr><td>{s.slot_index + 1}</td>"
            f"<td>{s.start_time:.2f}s</td>"
            f"<td>{s.end_time - s.start_time:.2f}s</td>"
            f"<td>{s.onset_strength:.3f}</td>"
            f"<td>{html_escape(s.clip_name)}</td>"
            f"<td>{html_escape(s.source_filename)}</td>"
            f"<td>{cos_dist}</td>"
            f"<td>{s.target_distance:.3f}</td>"
            f"<td>{s.distance_score:.3f}</td>"
            f"<td>{loop}</td></tr>"
        )

    return f"""
  <p style="color: #f97316;">Chart.js vendor files not found — showing table only.</p>
  <table>
    <thead><tr>
      <th>#</th><th>Time</th><th>Duration</th><th>Onset</th>
      <th>Clip</th><th>Source</th><th>Cos Dist</th><th>Target</th><th>Score</th><th>Loop</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>"""


def save_staccato_report(debug: "StaccatoDebugInfo", output_path: Path) -> Path:
    """Generate and save the debug report to an HTML file.

    Args:
        debug: StaccatoDebugInfo from a Staccato generation run
        output_path: Path to write the HTML file

    Returns:
        The output_path (for convenience)
    """
    html = generate_staccato_report(debug)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Staccato debug report saved to {output_path}")
    return output_path


_CSS = """
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --border: #475569;
  --accent-blue: #3b82f6;
  --accent-orange: #f97316;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.5;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 32px 24px;
}

h1 {
  font-size: 24px;
  margin-bottom: 16px;
}

h2 {
  font-size: 16px;
  color: var(--text-secondary);
  margin-bottom: 12px;
}

.stats {
  display: flex;
  gap: 24px;
  margin-bottom: 32px;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
  border: 1px solid var(--border);
}

.stat {
  color: var(--text-secondary);
  font-size: 14px;
}

.stat strong {
  color: var(--text-primary);
}

.chart-section {
  margin-bottom: 32px;
}

.chart-wrapper {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  height: 300px;
}

.legend {
  margin-top: 24px;
  padding: 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.legend p { margin-bottom: 8px; }
.legend p:last-child { margin-bottom: 0; }

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

thead {
  background: var(--bg-tertiary);
}

th, td {
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

th {
  color: var(--text-secondary);
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

td {
  color: var(--text-primary);
}

tr:last-child td {
  border-bottom: none;
}

tr:hover td {
  background: rgba(59, 130, 246, 0.05);
}
"""
