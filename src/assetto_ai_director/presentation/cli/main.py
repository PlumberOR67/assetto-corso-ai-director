"""The ``aidirector`` command-line entry point.

Commands are registered here as each stage of the pipeline lands. Every command
is expected to work without a graphical session, so the whole tool remains usable
over SSH and inside continuous integration.
"""

from __future__ import annotations

import typer
from rich.console import Console

from assetto_ai_director import __version__
from assetto_ai_director.presentation.cli.console import configure_stdio
from assetto_ai_director.presentation.cli.csv_cmd import csv
from assetto_ai_director.presentation.cli.diagnose_cmd import diagnose
from assetto_ai_director.presentation.cli.direct_cmd import direct
from assetto_ai_director.presentation.cli.doctor_cmd import doctor
from assetto_ai_director.presentation.cli.events_cmd import events
from assetto_ai_director.presentation.cli.export_cmd import export
from assetto_ai_director.presentation.cli.inspect_cmd import inspect
from assetto_ai_director.presentation.cli.install_cmd import install
from assetto_ai_director.presentation.cli.measure_cmd import measure
from assetto_ai_director.presentation.cli.score_cmd import score
from assetto_ai_director.presentation.cli.styles_cmd import styles

__all__ = ["app"]

# Before any Console is built. A non-UTF-8 console code page is still the default
# on a non-English Windows install, which is the primary platform for this tool,
# and an unencodable character in a report otherwise terminates the command.
configure_stdio()

app = typer.Typer(
    name="aidirector",
    help="AI cinematographer for Assetto Corsa: analyse a session, direct a cut.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.callback()
def main() -> None:
    """Group the commands under ``aidirector``.

    Present so Typer keeps sub-command dispatch even when only one command is
    registered. Without it, a single-command app collapses into a bare
    executable and ``aidirector version`` becomes an unexpected argument.
    """


@app.command()
def version() -> None:
    """Print the installed version and exit."""
    console.print(f"assetto-ai-director [bold cyan]{__version__}[/]")


app.command(name="doctor")(doctor)
app.command(name="diagnose")(diagnose)
app.command(name="inspect")(inspect)
app.command(name="measure")(measure)
app.command(name="events")(events)
app.command(name="score")(score)
app.command(name="direct")(direct)
app.command(name="export")(export)
app.command(name="install")(install)
app.command(name="csv")(csv)
app.command(name="styles")(styles)


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    app()
