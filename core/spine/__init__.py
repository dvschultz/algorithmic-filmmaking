"""Scene Ripper spine package — GUI-agnostic tool implementations.

The spine is a zero-dependency layer below both consumers:

- ``core/chat_tools.py`` (the GUI agent's tool catalog) wraps spine functions.
- ``scene_ripper_mcp/tools/*`` (the headless MCP server) wraps the same spine
  functions.

Spine modules MUST NOT import PySide6, mpv, av, or any other GUI/heavy-runtime
dependency at module load. Lazy imports inside function bodies are fine when
they are gated by a runtime check. The ``tests/test_spine_imports.py``
boundary test enforces this invariant.
"""
