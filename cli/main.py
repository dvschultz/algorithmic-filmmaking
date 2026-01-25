"""Scene Ripper CLI entry point."""

import click

from cli import __version__


@click.group()
@click.version_option(version=__version__, prog_name="scene_ripper")
@click.option("--json", "output_json", is_flag=True, help="Output in JSON format")
@click.pass_context
def cli(ctx: click.Context, output_json: bool) -> None:
    """Scene Ripper - Video scene detection and analysis.

    A command-line tool for detecting scenes in videos, analyzing clips,
    and exporting results. All operations work headlessly without a GUI.

    \b
    Examples:
        scene_ripper detect movie.mp4
        scene_ripper analyze colors project.json
        scene_ripper export clips project.json -o ./clips/

    \b
    Shell Completion:
        # Bash (~/.bashrc)
        eval "$(_SCENE_RIPPER_COMPLETE=bash_source scene_ripper)"

        # Zsh (~/.zshrc)
        eval "$(_SCENE_RIPPER_COMPLETE=zsh_source scene_ripper)"

        # Fish (~/.config/fish/completions/scene_ripper.fish)
        _SCENE_RIPPER_COMPLETE=fish_source scene_ripper > ~/.config/fish/completions/scene_ripper.fish
    """
    ctx.ensure_object(dict)
    ctx.obj["json"] = output_json


@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Generate shell completion script.

    Output the completion script for your shell and add it to your
    shell's configuration file.

    \b
    Examples:
        # Bash - add to ~/.bashrc
        scene_ripper completion bash >> ~/.bashrc

        # Zsh - add to ~/.zshrc
        scene_ripper completion zsh >> ~/.zshrc

        # Fish - save to completions directory
        scene_ripper completion fish > ~/.config/fish/completions/scene_ripper.fish
    """
    import os
    import subprocess

    env_var = "_SCENE_RIPPER_COMPLETE"
    source_type = f"{shell}_source"

    # Get the completion script by invoking Click's completion mechanism
    env = os.environ.copy()
    env[env_var] = source_type

    try:
        result = subprocess.run(
            ["scene_ripper"],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            click.echo(result.stdout)
        else:
            # Fallback: output manual instructions
            if shell == "bash":
                click.echo('eval "$(_SCENE_RIPPER_COMPLETE=bash_source scene_ripper)"')
            elif shell == "zsh":
                click.echo('eval "$(_SCENE_RIPPER_COMPLETE=zsh_source scene_ripper)"')
            elif shell == "fish":
                click.echo("_SCENE_RIPPER_COMPLETE=fish_source scene_ripper | source")
    except FileNotFoundError:
        # scene_ripper not installed, provide manual instructions
        if shell == "bash":
            click.echo('eval "$(_SCENE_RIPPER_COMPLETE=bash_source scene_ripper)"')
        elif shell == "zsh":
            click.echo('eval "$(_SCENE_RIPPER_COMPLETE=zsh_source scene_ripper)"')
        elif shell == "fish":
            click.echo("_SCENE_RIPPER_COMPLETE=fish_source scene_ripper | source")


def register_commands() -> None:
    """Register all command modules."""
    # Import and register commands
    # Using lazy imports to keep CLI startup fast
    from cli.commands import detect, project, analyze, transcribe, export, youtube

    cli.add_command(detect.detect)
    cli.add_command(project.project)
    cli.add_command(analyze.analyze)
    cli.add_command(transcribe.transcribe)
    cli.add_command(export.export)
    cli.add_command(youtube.search)
    cli.add_command(youtube.download)


def main() -> None:
    """Main entry point for the CLI."""
    register_commands()
    cli()


if __name__ == "__main__":
    main()
