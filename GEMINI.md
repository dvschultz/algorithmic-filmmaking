# Gemini Code Assistant Context

This document provides a comprehensive overview of the "Scene Ripper" project, its architecture, and development conventions to assist the Gemini code assistant.

## Project Overview

Scene Ripper is a desktop application for algorithmic filmmaking and video analysis. It provides both a graphical user interface (GUI) and a command-line interface (CLI) for users to import videos, automatically detect scenes, analyze clip properties, arrange clips into new sequences, and export the results.

A key feature is the integrated "Agent Chat," which allows users to control the application and perform complex workflows using natural language prompts powered by LLMs like Gemini.

### Core Technologies

| Component           | Technology                                        |
| ------------------- | ------------------------------------------------- |
| **Language**        | Python 3.11+                                      |
| **GUI Framework**   | PySide6 (Qt 6)                                    |
| **CLI Framework**   | Click                                             |
| **Scene Detection** | PySceneDetect                                     |
| **Video Processing**| FFmpeg, OpenCV                                    |
| **Video Download**  | yt-dlp                                            |
| **Transcription**   | faster-whisper                                    |
| **AI/LLM Agent**    | `litellm` for multi-provider support (Gemini, etc.)|
| **Testing**         | pytest                                            |
| **Linting**         | Ruff                                              |

### Architecture

The project is designed with a clean separation of concerns:

-   **`main.py`**: The entry point for the PySide6 GUI application.
-   **`cli/`**: Contains the complete CLI application built with `click`.
    -   `cli/main.py`: CLI entry point, which registers commands from the `cli/commands/` directory.
-   **`core/`**: Houses the core business logic, decoupled from the UI.
    -   `core/project.py`: Defines the central `Project` class, which acts as the single source of truth for all project data (sources, clips, sequences). Both the GUI and CLI interact with this class.
    -   `core/scene_detect.py`, `core/transcription.py`, etc.: Wrappers for core functionalities.
    -   `core/chat_tools.py`: Defines the functions (tools) that the LLM agent can call to interact with the application state.
-   **`ui/`**: Contains all GUI components.
    -   `ui/main_window.py`: The main application window, responsible for orchestrating UI elements, managing background workers (`QThread`), and handling signals.
    -   `ui/tabs/`: The application's workflow is divided into tabs (Collect, Cut, Analyze, etc.).
    -   `ui/chat_panel.py` & `ui/chat_worker.py`: The UI and background worker for the LLM agent functionality.
-   **`models/`**: Defines the data structures (dataclasses) for `Source`, `Clip`, and `Sequence`.
-   **`tests/`**: Contains unit and integration tests for the CLI and core logic.

A key architectural pattern is the use of `QThread` workers for long-running tasks like scene detection, thumbnail generation, analysis, and exporting. This keeps the GUI responsive. The LLM agent also runs in a separate worker thread (`ChatAgentWorker`) and communicates with the main UI thread to execute GUI-modifying "tools."

## Building and Running

### 1. Installation

First, install the required system and Python dependencies.

**System Dependencies (Example for Ubuntu):**

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

**Python Dependencies:**

```bash
# It is recommended to use a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application

**GUI:**

To run the main graphical application:

```bash
python main.py
```

**CLI:**

The CLI is defined as a script in `pyproject.toml`. After installing the package in editable mode, you can run it directly.

```bash
# Install the project in editable mode to make the CLI available
pip install -e .

# Now you can use the 'scene_ripper' command
scene_ripper --help

# You can also invoke it as a module
python -m cli.main --help
```

#### CLI Commands

The CLI provides a rich set of commands for headless operation:

-   `scene_ripper project ...`: Create and manage project files.
-   `scene_ripper detect <video_path>`: Detect scenes in a video.
-   `scene_ripper analyze <type> <project.json>`: Run analysis (colors, shots) on a project.
-   `scene_ripper transcribe <project.json>`: Transcribe clips in a project.
-   `scene_ripper youtube search "<query>"`: Search YouTube.
-   `scene_ripper youtube download <url>`: Download a video from a URL.
-   `scene_ripper export <type> <project.json>`: Export clips, sequences, or datasets.

## Development Conventions

### Code Style & Linting

The project uses **Ruff** for linting and formatting.

-   **To check for issues:** `ruff check .`
-   **To format code:** `ruff format .`

Please adhere to existing code style and conventions.

### Testing

The project uses **pytest** for testing. Tests are located in the `tests/` directory.

-   **To run all tests:** `python -m pytest tests/ -v`

When adding new features or fixing bugs, please include corresponding tests. The CI pipeline requires tests to pass.

### Commits

Follow conventional commit standards for commit messages (e.g., `feat:`, `fix:`, `refactor:`, `docs:`).
This is not strictly enforced but is the preferred style found in the repository history.

### Agent-Accessible Tools

The core of the agent functionality is in `core/chat_tools.py` and `ui/main_window.py`. When the agent needs to perform an action, it calls a "tool."

-   **Backend-only tools**: Defined in `core/chat_tools.py` and can be executed directly by the `ChatAgentWorker`.
-   **GUI-modifying tools**: These are also defined in `core/chat_tools.py` but have `modifies_gui_state=True`. They are executed on the main UI thread via the `_on_gui_tool_requested` slot in `MainWindow`. If your tool needs to change the UI (e.g., switch tabs, select a clip), it must be marked as a GUI tool. For long-running GUI tools, they can return a `_wait_for_worker` token, and the `MainWindow` will wait for the corresponding worker's `finished` signal before sending the result back to the agent.
