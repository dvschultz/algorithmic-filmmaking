"""Workflow tabs for Scene Ripper."""

from .base_tab import BaseTab
from .collect_tab import CollectTab
from .cut_tab import CutTab
from .analyze_tab import AnalyzeTab
from .frames_tab import FramesTab
from .generate_tab import GenerateTab
from .sequence_tab import SequenceTab
from .render_tab import RenderTab

__all__ = [
    "BaseTab",
    "CollectTab",
    "CutTab",
    "AnalyzeTab",
    "FramesTab",
    "GenerateTab",
    "SequenceTab",
    "RenderTab",
]
