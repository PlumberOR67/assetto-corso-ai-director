"""Tests for installing a camera set into a game.

This is the only thing the project does outside its own output folder, so the
tests are mostly about what it refuses: to write without being told twice, to
guess which of two installations was meant, and to replace a file without keeping
what was there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from assetto_ai_director.infrastructure import ac_install
from assetto_ai_director.presentation.cli.main import app

runner = CliRunner()

_CAMERAS = """[HEADER]
VERSION=3
CAMERA_COUNT=1
SET_NAME=AI Director initial_d

[CAMERA_0]
NAME=AID 00 drift_start
POSITION=1,2,3
FORWARD=0,0,1
UP=0,1,0
MIN_FOV=22.00
MAX_FOV=52.00
IN_POINT=0.00000
OUT_POINT=1.00000
IS_FIXED=1
"""


@pytest.fixture(autouse=True)
def _no_machine_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests from finding a real installation.

    Args:
        monkeypatch: Pytest's patcher.
    """
    monkeypatch.delenv(ac_install.ENV_VAR, raising=False)
    monkeypatch.setattr(ac_install, "_DEFAULT_STEAM_DIRS", ())
    monkeypatch.setattr(ac_install, "_steam_path_from_registry", lambda: None)


@pytest.fixture
def game(tmp_path: Path) -> Path:
    """A mock installation with one track layout.

    Args:
        tmp_path: Pytest's temporary directory.

    Returns:
        The installation root.
    """
    root = tmp_path / "ac"
    (root / "content" / "tracks" / "pk_akina" / "akina_downhill" / "data").mkdir(parents=True)
    return root


@pytest.fixture
def cameras(tmp_path: Path) -> Path:
    """A camera set to install.

    Args:
        tmp_path: Pytest's temporary directory.

    Returns:
        The file.
    """
    path = tmp_path / "cameras_9.ini"
    path.write_text(_CAMERAS, encoding="utf-8")
    return path


def _run(cameras: Path, game: Path, *extra: str) -> tuple[int, str]:
    """Invoke the command against a mock installation.

    Args:
        cameras: The set to install.
        game: The installation root.
        *extra: Further arguments.

    Returns:
        ``(exit_code, output)``.
    """
    result = runner.invoke(
        app,
        [
            "install",
            str(cameras),
            "--track",
            "pk_akina",
            "--layout",
            "akina_downhill",
            "--ac-root",
            str(game),
            *extra,
        ],
    )
    return result.exit_code, result.output


class TestDryRunIsTheDefault:
    def test_nothing_is_written_without_being_asked(self, cameras: Path, game: Path) -> None:
        code, output = _run(cameras, game)
        assert code == 0
        assert "Dry run" in output
        destination = game / "content" / "tracks" / "pk_akina" / "akina_downhill" / "data"
        assert not (destination / "cameras_9.ini").exists()

    def test_it_names_the_destination_it_would_use(self, cameras: Path, game: Path) -> None:
        _, output = _run(cameras, game)
        assert "akina_downhill" in output.replace("\n", "")

    def test_it_reads_the_set_back_before_offering_to_install_it(
        self, cameras: Path, game: Path
    ) -> None:
        _, output = _run(cameras, game)
        assert "1 cameras" in output
        assert "contiguous" in output


class TestWriting:
    def test_the_file_lands_in_the_layout_folder(self, cameras: Path, game: Path) -> None:
        code, _ = _run(cameras, game, "--write")
        assert code == 0
        installed = (
            game / "content" / "tracks" / "pk_akina" / "akina_downhill" / "data" / "cameras_9.ini"
        )
        assert installed.read_text(encoding="utf-8") == _CAMERAS

    def test_an_existing_set_is_kept(self, cameras: Path, game: Path) -> None:
        # A camera set somebody hand-authored and one this project generated look
        # identical from the command line.
        data = game / "content" / "tracks" / "pk_akina" / "akina_downhill" / "data"
        (data / "cameras_9.ini").write_text("; hand written\n", encoding="utf-8")

        _run(cameras, game, "--write")
        backup = data / "cameras_9.ini.aidirector-backup"
        assert backup.read_text(encoding="utf-8") == "; hand written\n"

    def test_the_set_index_chooses_the_name(self, cameras: Path, game: Path) -> None:
        _run(cameras, game, "--write", "--set-index", "3")
        data = game / "content" / "tracks" / "pk_akina" / "akina_downhill" / "data"
        assert (data / "cameras_3.ini").is_file()

    def test_set_zero_is_the_unnumbered_name(self, cameras: Path, game: Path) -> None:
        _run(cameras, game, "--write", "--set-index", "0")
        data = game / "content" / "tracks" / "pk_akina" / "akina_downhill" / "data"
        assert (data / "cameras.ini").is_file()


class TestRefusals:
    def test_a_wrong_layout_is_refused_rather_than_created(self, cameras: Path, game: Path) -> None:
        # Writing to the track root instead of the layout produces a file the
        # game silently never loads, which is the mistake this command exists to
        # prevent. Creating the folder would hide it.
        result = runner.invoke(
            app,
            [
                "install",
                str(cameras),
                "--track",
                "pk_akina",
                "--layout",
                "akina_uphill",
                "--ac-root",
                str(game),
                "--write",
            ],
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output
        assert not (game / "content" / "tracks" / "pk_akina" / "akina_uphill").exists()

    def test_no_installation_is_refused(self, cameras: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, ["install", str(cameras), "--track", "pk_akina"])
        assert result.exit_code == 1
        assert "No Assetto Corsa installation found" in result.output

    def test_no_track_named_is_refused(self, cameras: Path, game: Path) -> None:
        result = runner.invoke(app, ["install", str(cameras), "--ac-root", str(game)])
        assert result.exit_code == 1
        assert "No track named" in result.output
