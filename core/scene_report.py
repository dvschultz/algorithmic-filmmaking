"""Scene/sequence analysis report generation.

Generates human-readable film analysis reports synthesizing clip metadata,
cinematography analysis, pacing metrics, and advisory suggestions.

Output formats:
- Markdown (default): Plain text markdown
- HTML: Rendered markdown with basic styling
"""

import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.clip import Clip
    from models.sequence import Sequence
    from core.project import Project

logger = logging.getLogger(__name__)

# Report section templates
REPORT_SECTIONS = {
    "overview": "Overview",
    "cinematography": "Cinematography Analysis",
    "pacing": "Pacing & Rhythm",
    "visual_consistency": "Visual Consistency",
    "continuity": "Continuity Notes",
    "recommendations": "Suggestions",
    "clip_details": "Clip Details",
}

DEFAULT_SECTIONS = ["overview", "cinematography", "pacing", "recommendations"]


def generate_sequence_report(
    sequence: "Sequence",
    project: "Project",
    sections: Optional[list[str]] = None,
    include_clip_details: bool = False,
) -> str:
    """Generate a film analysis report for a sequence.

    Args:
        sequence: The sequence to analyze
        project: Project containing clips and sources
        sections: Which sections to include (default: overview, cinematography, pacing, recommendations)
        include_clip_details: Whether to include per-clip breakdown

    Returns:
        Markdown-formatted report string
    """
    from core.analysis.sequence import analyze_sequence

    sections = sections or DEFAULT_SECTIONS
    all_clips = sequence.get_all_clips()

    if not all_clips:
        return _generate_empty_report(sequence.name)

    # Get sequence analysis
    analysis = analyze_sequence(sequence, project)

    # Resolve source clips for metadata
    resolved_clips = []
    for seq_clip in all_clips:
        source_clip = project.clips_by_id.get(seq_clip.source_clip_id)
        source = project.sources_by_id.get(seq_clip.source_id)
        if source_clip:
            resolved_clips.append((seq_clip, source_clip, source))

    # Build report
    report_parts = []

    # Header
    report_parts.append(f"# Film Analysis Report: {sequence.name}")
    report_parts.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    report_parts.append(f"\n**Clips:** {len(all_clips)} | **Duration:** {sequence.duration_seconds:.1f}s")
    report_parts.append("")

    # Generate requested sections
    if "overview" in sections:
        report_parts.append(_generate_overview(sequence, analysis, resolved_clips))

    if "cinematography" in sections:
        report_parts.append(_generate_cinematography_section(resolved_clips))

    if "pacing" in sections:
        report_parts.append(_generate_pacing_section(analysis))

    if "visual_consistency" in sections:
        report_parts.append(_generate_visual_consistency_section(analysis))

    if "continuity" in sections and analysis.continuity_warnings:
        report_parts.append(_generate_continuity_section(analysis))

    if "recommendations" in sections and analysis.suggestions:
        report_parts.append(_generate_recommendations_section(analysis))

    if include_clip_details or "clip_details" in sections:
        report_parts.append(_generate_clip_details(resolved_clips))

    return "\n".join(report_parts)


def generate_clips_report(
    clips: list["Clip"],
    project: "Project",
    title: str = "Clip Analysis Report",
) -> str:
    """Generate a report for a selection of clips.

    Args:
        clips: List of clips to analyze
        project: Project containing sources
        title: Report title

    Returns:
        Markdown-formatted report string
    """
    if not clips:
        return f"# {title}\n\n*No clips to analyze.*"

    report_parts = []

    # Header
    report_parts.append(f"# {title}")
    report_parts.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    report_parts.append(f"\n**Clips analyzed:** {len(clips)}")
    report_parts.append("")

    # Resolve sources
    resolved = []
    for clip in clips:
        source = project.sources_by_id.get(clip.source_id)
        resolved.append((None, clip, source))

    # Cinematography summary
    report_parts.append(_generate_cinematography_section(resolved))

    # Individual clip details
    report_parts.append(_generate_clip_details(resolved))

    return "\n".join(report_parts)


def _generate_empty_report(name: str) -> str:
    """Generate report for empty sequence."""
    return f"""# Film Analysis Report: {name}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*

**Status:** No clips in sequence

Add clips to the sequence to generate analysis.
"""


def _generate_overview(
    sequence: "Sequence",
    analysis,
    resolved_clips: list,
) -> str:
    """Generate overview section."""
    pacing = analysis.pacing

    # Collect shot size distribution
    shot_sizes = {}
    for _, clip, _ in resolved_clips:
        if clip and clip.cinematography and clip.cinematography.shot_size:
            size = clip.cinematography.shot_size
            shot_sizes[size] = shot_sizes.get(size, 0) + 1

    overview = ["## Overview", ""]

    # Summary stats
    overview.append(f"This sequence contains **{pacing.clip_count} clips** ")
    overview.append(f"with a total duration of **{pacing.total_duration_seconds:.1f} seconds**.")
    overview.append("")

    # Pacing summary
    overview.append(f"**Pacing:** {pacing.classification.replace('_', ' ').title()} ")
    overview.append(f"(average shot duration: {pacing.average_duration:.1f}s)")
    overview.append("")

    # Shot size distribution
    if shot_sizes:
        overview.append("**Shot Size Distribution:**")
        for size, count in sorted(shot_sizes.items(), key=lambda x: -x[1]):
            pct = (count / len(resolved_clips)) * 100
            overview.append(f"- {size}: {count} ({pct:.0f}%)")
        overview.append("")

    return "\n".join(overview)


def _generate_cinematography_section(resolved_clips: list) -> str:
    """Generate cinematography analysis section."""
    section = ["## Cinematography Analysis", ""]

    # Collect cinematography data
    camera_angles = {}
    camera_movements = {}
    lighting_styles = {}
    shot_sizes = {}

    for _, clip, _ in resolved_clips:
        if not clip or not clip.cinematography:
            continue

        cine = clip.cinematography

        if cine.shot_size and cine.shot_size != "unknown":
            shot_sizes[cine.shot_size] = shot_sizes.get(cine.shot_size, 0) + 1

        if cine.camera_angle and cine.camera_angle != "unknown":
            camera_angles[cine.camera_angle] = camera_angles.get(cine.camera_angle, 0) + 1

        if cine.camera_movement and cine.camera_movement != "unknown":
            camera_movements[cine.camera_movement] = camera_movements.get(cine.camera_movement, 0) + 1

        if cine.lighting_style and cine.lighting_style != "unknown":
            lighting_styles[cine.lighting_style] = lighting_styles.get(cine.lighting_style, 0) + 1

    # Shot sizes
    if shot_sizes:
        section.append("### Shot Sizes")
        for size, count in sorted(shot_sizes.items(), key=lambda x: -x[1])[:5]:
            section.append(f"- **{size}**: {count} clips")
        section.append("")

    # Camera angles
    if camera_angles:
        section.append("### Camera Angles")
        for angle, count in sorted(camera_angles.items(), key=lambda x: -x[1])[:5]:
            section.append(f"- **{angle.replace('_', ' ').title()}**: {count} clips")
        section.append("")

    # Camera movement
    if camera_movements:
        section.append("### Camera Movement")
        for movement, count in sorted(camera_movements.items(), key=lambda x: -x[1])[:5]:
            section.append(f"- **{movement.replace('_', ' ').title()}**: {count} clips")
        section.append("")

    # Lighting
    if lighting_styles:
        section.append("### Lighting")
        for style, count in sorted(lighting_styles.items(), key=lambda x: -x[1])[:5]:
            section.append(f"- **{style.replace('_', ' ').title()}**: {count} clips")
        section.append("")

    if len(section) == 2:  # Only header
        section.append("*No cinematography data available. Run clip analysis first.*")
        section.append("")

    return "\n".join(section)


def _generate_pacing_section(analysis) -> str:
    """Generate pacing analysis section."""
    pacing = analysis.pacing

    section = ["## Pacing & Rhythm", ""]

    section.append(f"| Metric | Value |")
    section.append(f"|--------|-------|")
    section.append(f"| Classification | {pacing.classification.replace('_', ' ').title()} |")
    section.append(f"| Average Shot Duration | {pacing.average_duration:.2f}s |")
    section.append(f"| Shortest Shot | {pacing.min_duration:.2f}s |")
    section.append(f"| Longest Shot | {pacing.max_duration:.2f}s |")
    section.append(f"| Standard Deviation | {pacing.std_deviation:.2f}s |")
    section.append("")

    # Pacing interpretation
    if pacing.classification == "very_fast":
        section.append("*Very fast pacing typical of action sequences or music videos.*")
    elif pacing.classification == "fast":
        section.append("*Fast pacing creates energy and urgency.*")
    elif pacing.classification == "medium":
        section.append("*Medium pacing balances engagement with comprehension.*")
    elif pacing.classification == "slow":
        section.append("*Slower pacing allows for contemplation and emotional depth.*")
    elif pacing.classification == "very_slow":
        section.append("*Very slow pacing emphasizes atmosphere and observation.*")
    section.append("")

    return "\n".join(section)


def _generate_visual_consistency_section(analysis) -> str:
    """Generate visual consistency section."""
    vc = analysis.visual_consistency

    section = ["## Visual Consistency", ""]

    section.append(f"| Aspect | Score/Value |")
    section.append(f"|--------|-------------|")
    section.append(f"| Color Consistency | {vc.color_consistency:.0%} |")
    section.append(f"| Lighting Consistency | {vc.lighting_consistency:.0%} |")
    section.append(f"| Shot Size Variety | {vc.shot_size_variety} types |")
    section.append(f"| Dominant Shot Size | {vc.dominant_shot_size} |")
    section.append(f"| Color Temperature Shifts | {vc.color_temperature_shifts} |")
    section.append("")

    return "\n".join(section)


def _generate_continuity_section(analysis) -> str:
    """Generate continuity notes section."""
    section = ["## Continuity Notes", ""]

    section.append("*These are advisory observations, not errors. Many are valid artistic choices.*")
    section.append("")

    for warning in analysis.continuity_warnings:
        severity_emoji = {"low": "ℹ️", "medium": "⚠️", "high": "🔴"}.get(warning.severity, "•")
        section.append(f"- {severity_emoji} **{warning.warning_type.replace('_', ' ').title()}**: {warning.explanation}")

    section.append("")
    return "\n".join(section)


def _generate_recommendations_section(analysis) -> str:
    """Generate recommendations section."""
    section = ["## Suggestions", ""]

    section.append("*These are advisory suggestions based on the analysis:*")
    section.append("")

    for suggestion in analysis.suggestions:
        section.append(f"- {suggestion}")

    section.append("")
    return "\n".join(section)


def _generate_clip_details(resolved_clips: list) -> str:
    """Generate detailed per-clip breakdown."""
    section = ["## Clip Details", ""]

    for i, (seq_clip, clip, source) in enumerate(resolved_clips, 1):
        if not clip:
            continue

        source_name = source.file_path.name if source else "Unknown"
        fps = source.fps if source else 30.0

        section.append(f"### Clip {i}")
        section.append(f"- **Source:** {source_name}")
        section.append(f"- **Duration:** {clip.duration_seconds(fps):.2f}s")

        if clip.cinematography:
            cine = clip.cinematography
            if cine.shot_size and cine.shot_size != "unknown":
                section.append(f"- **Shot Size:** {cine.shot_size}")
            if cine.camera_angle and cine.camera_angle != "unknown":
                section.append(f"- **Camera Angle:** {cine.camera_angle.replace('_', ' ').title()}")
            if cine.camera_movement and cine.camera_movement != "unknown":
                section.append(f"- **Movement:** {cine.camera_movement.replace('_', ' ').title()}")
            if cine.lighting_style and cine.lighting_style != "unknown":
                section.append(f"- **Lighting:** {cine.lighting_style.replace('_', ' ').title()}")

        if clip.description:
            desc = clip.description[:150] + "..." if len(clip.description) > 150 else clip.description
            section.append(f"- **Description:** {desc}")

        section.append("")

    return "\n".join(section)


def report_to_html(markdown_report: str) -> str:
    """Convert markdown report to HTML with basic styling.

    Args:
        markdown_report: Markdown-formatted report

    Returns:
        HTML string with embedded CSS
    """
    # Simple markdown to HTML conversion
    # For production, consider using a proper markdown library

    html_content = markdown_report

    # Convert headers
    import re
    html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)

    # Convert bold and italic
    html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)

    # Convert lists
    html_content = re.sub(r'^- (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)

    # Wrap list items
    html_content = re.sub(r'(<li>.*?</li>\n)+', lambda m: f'<ul>\n{m.group(0)}</ul>\n', html_content)

    # Convert tables (basic)
    lines = html_content.split('\n')
    in_table = False
    new_lines = []
    for line in lines:
        if '|' in line and '---' not in line:
            if not in_table:
                new_lines.append('<table>')
                in_table = True
            cells = [c.strip() for c in line.split('|')[1:-1]]
            row = '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
            new_lines.append(row)
        else:
            if in_table and '|' not in line:
                new_lines.append('</table>')
                in_table = False
            if '---' not in line:
                new_lines.append(line)
    if in_table:
        new_lines.append('</table>')

    html_content = '\n'.join(new_lines)

    # Convert line breaks
    html_content = re.sub(r'\n\n', '</p><p>', html_content)

    # Wrap in HTML document
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Film Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        td, th {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 5px 0; }}
        em {{ color: #666; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

    return html
