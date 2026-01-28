"""Chat export functionality for exporting agent conversations to files."""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ChatExportConfig:
    """Configuration for chat export."""

    output_dir: Path
    format: str  # "markdown", "json", "both"
    include_user: bool = True
    include_assistant: bool = True
    include_tools: bool = True
    include_tool_args: bool = False  # Verbose mode
    project_name: str = ""


def generate_export_filename(format_type: str) -> str:
    """Generate a timestamped export filename.

    Args:
        format_type: File format ("md" or "json")

    Returns:
        Filename like "chat_2026-01-28_143022.md"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"chat_{timestamp}.{format_type}"


def _sanitize_path_in_content(content: str) -> str:
    """Replace user home directory with ~ in content.

    Args:
        content: Text content that may contain paths

    Returns:
        Content with home directory replaced by ~
    """
    if not content:
        return content

    home_dir = os.path.expanduser("~")
    return content.replace(home_dir, "~")


def _format_tool_call_markdown(tool_call: dict, include_args: bool = False) -> str:
    """Format a tool call for Markdown output.

    Args:
        tool_call: Tool call dict with id, type, function fields
        include_args: Whether to include function arguments

    Returns:
        Formatted Markdown string
    """
    func = tool_call.get("function", {})
    name = func.get("name", "unknown")

    lines = [f"> **Tool:** {name}"]

    if include_args:
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            if args:
                args_formatted = json.dumps(args, indent=2)
                lines.append(f"> **Arguments:**\n> ```json\n> {args_formatted}\n> ```")
        except json.JSONDecodeError:
            pass

    return "\n".join(lines)


def _format_tool_result_markdown(message: dict, include_args: bool = False) -> str:
    """Format a tool result message for Markdown output.

    Args:
        message: Tool result message with role=tool
        include_args: Whether to include full result content

    Returns:
        Formatted Markdown string
    """
    name = message.get("name", "unknown")
    content = message.get("content", "")

    lines = [f"> **Tool Result:** {name}"]

    if include_args and content:
        # Try to parse as JSON for pretty printing
        try:
            result = json.loads(content)
            success = result.get("success", None)
            if success is not None:
                status = "Success" if success else "Failed"
                lines.append(f"> **Status:** {status}")
        except json.JSONDecodeError:
            pass

    return "\n".join(lines)


def _filter_messages(
    messages: list[dict],
    include_user: bool,
    include_assistant: bool,
    include_tools: bool
) -> list[dict]:
    """Filter messages based on configuration.

    Args:
        messages: List of message dicts
        include_user: Include user messages
        include_assistant: Include assistant messages
        include_tools: Include tool calls and results

    Returns:
        Filtered list of messages
    """
    filtered = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "user" and include_user:
            filtered.append(msg)
        elif role == "assistant":
            if include_assistant:
                # Check if it's a tool call message
                has_tool_calls = "tool_calls" in msg
                if has_tool_calls and not include_tools:
                    # Skip tool call messages if tools not included
                    continue
                filtered.append(msg)
        elif role == "tool" and include_tools:
            filtered.append(msg)

    return filtered


def export_chat_as_markdown(
    messages: list[dict],
    config: ChatExportConfig
) -> tuple[bool, str]:
    """Export chat history as a Markdown file.

    Args:
        messages: List of message dicts from chat history
        config: Export configuration

    Returns:
        Tuple of (success, filepath_or_error)
    """
    # Filter messages
    filtered = _filter_messages(
        messages,
        config.include_user,
        config.include_assistant,
        config.include_tools
    )

    if not filtered:
        return False, "No messages to export after filtering"

    # Build Markdown content
    lines = [
        "# Agent Chat Export",
        "",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if config.project_name:
        lines.append(f"**Project:** {config.project_name}")

    lines.extend([
        f"**Messages:** {len(filtered)}",
        "",
        "---",
        ""
    ])

    for msg in filtered:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Sanitize paths in content
        content = _sanitize_path_in_content(content)

        if role == "user":
            lines.extend([
                "## User",
                "",
                content,
                "",
                "---",
                ""
            ])
        elif role == "assistant":
            lines.extend([
                "## Assistant",
                ""
            ])

            if content:
                lines.extend([content, ""])

            # Handle tool calls
            if "tool_calls" in msg and config.include_tools:
                for tool_call in msg["tool_calls"]:
                    lines.extend([
                        _format_tool_call_markdown(tool_call, config.include_tool_args),
                        ""
                    ])

            lines.extend(["---", ""])

        elif role == "tool":
            lines.extend([
                _format_tool_result_markdown(msg, config.include_tool_args),
                "",
                "---",
                ""
            ])

    # Generate filename and write
    filename = generate_export_filename("md")
    output_path = config.output_dir / filename

    try:
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return True, str(output_path)
    except (OSError, IOError) as e:
        return False, f"Failed to write file: {e}"


def export_chat_as_json(
    messages: list[dict],
    config: ChatExportConfig
) -> tuple[bool, str]:
    """Export chat history as a JSON file.

    Args:
        messages: List of message dicts from chat history
        config: Export configuration

    Returns:
        Tuple of (success, filepath_or_error)
    """
    # Filter messages
    filtered = _filter_messages(
        messages,
        config.include_user,
        config.include_assistant,
        config.include_tools
    )

    if not filtered:
        return False, "No messages to export after filtering"

    # Sanitize paths in messages
    sanitized_messages = []
    for msg in filtered:
        sanitized = msg.copy()
        if "content" in sanitized:
            sanitized["content"] = _sanitize_path_in_content(sanitized["content"])
        sanitized_messages.append(sanitized)

    # Build JSON structure
    export_data = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "project": {
            "name": config.project_name or "Unnamed Project"
        },
        "message_count": len(sanitized_messages),
        "messages": sanitized_messages
    }

    # Generate filename and write
    filename = generate_export_filename("json")
    output_path = config.output_dir / filename

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        return True, str(output_path)
    except (OSError, IOError) as e:
        return False, f"Failed to write file: {e}"


def export_chat(
    messages: list[dict],
    config: ChatExportConfig
) -> tuple[bool, list[str], str]:
    """Export chat history based on configuration.

    Args:
        messages: List of message dicts from chat history
        config: Export configuration

    Returns:
        Tuple of (success, list_of_created_files, error_message)
    """
    created_files = []
    errors = []

    if config.format in ("markdown", "both"):
        success, result = export_chat_as_markdown(messages, config)
        if success:
            created_files.append(result)
        else:
            errors.append(f"Markdown: {result}")

    if config.format in ("json", "both"):
        success, result = export_chat_as_json(messages, config)
        if success:
            created_files.append(result)
        else:
            errors.append(f"JSON: {result}")

    if errors:
        return False, created_files, "; ".join(errors)

    return True, created_files, ""
