"""Tests for finding an Assetto Corsa installation.

Two properties matter more than the rest, and both are about *not* being wrong:

* A proposed directory that is not an Assetto Corsa install must produce no
  candidate. Discovery guessing is fine; discovery returning a plausible wrong
  path is not, because everything downstream then reports an empty track folder
  instead of a missing game.
* A machine with no Assetto Corsa must produce a complete, successful snapshot
  saying so. That is what continuous integration runs, and it is also what the
  development machine looks like.

Every test that asserts absence stubs the machine-wide discovery sources.
Otherwise the suite would pass or fail depending on whether the developer
happens to own the game.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from assetto_ai_director.infrastructure import ac_install
from assetto_ai_director.infrastructure.ac_install import (
    ENV_VAR,
    discover_ac_root,
    environment_snapshot,
    find_track_data,
)


@pytest.fixture(autouse=True)
def _no_machine_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cut off Steam and the registry so tests see only what they set up.

    Args:
        monkeypatch: Pytest's patcher.
    """
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(ac_install, "_DEFAULT_STEAM_DIRS", ())
    monkeypatch.setattr(ac_install, "_steam_path_from_registry", lambda: None)


def _install(root: Path, *, executable: bool = True) -> Path:
    """Build a directory that looks like an Assetto Corsa install.

    Args:
        root: Where to build it.
        executable: Whether to place a game executable beside the content.

    Returns:
        The install root.
    """
    (root / "content" / "tracks").mkdir(parents=True)
    if executable:
        (root / "acs.exe").write_bytes(b"")
    return root


def _track(root: Path, track_id: str, layout: str = "") -> Path:
    """Build a track folder inside an install.

    Args:
        root: The install root.
        track_id: Track folder name.
        layout: Layout name, empty for a single-layout track.

    Returns:
        The layout directory.
    """
    layout_dir = root / "content" / "tracks" / track_id
    if layout:
        layout_dir = layout_dir / layout
    (layout_dir / "data").mkdir(parents=True)
    return layout_dir


class TestDiscovery:
    def test_accepts_a_real_install(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        found = discover_ac_root(root)
        assert len(found) == 1
        assert found[0].root == root.resolve()
        assert found[0].found_by == "configured by hand"
        assert found[0].has_executable

    def test_rejects_a_directory_without_content(self, tmp_path: Path) -> None:
        empty = tmp_path / "not-ac"
        empty.mkdir()
        assert discover_ac_root(empty) == ()

    def test_rejects_a_path_that_does_not_exist(self, tmp_path: Path) -> None:
        assert discover_ac_root(tmp_path / "nowhere") == ()

    def test_content_copy_is_accepted_but_flagged(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "copy", executable=False)
        found = discover_ac_root(root)
        assert len(found) == 1
        assert not found[0].has_executable

    def test_environment_variable_is_a_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _install(tmp_path / "ac")
        monkeypatch.setenv(ENV_VAR, str(root))
        found = discover_ac_root()
        assert len(found) == 1
        assert ENV_VAR in found[0].found_by

    def test_explicit_beats_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chosen = _install(tmp_path / "chosen")
        other = _install(tmp_path / "other")
        monkeypatch.setenv(ENV_VAR, str(other))
        found = discover_ac_root(chosen)
        assert found[0].root == chosen.resolve()
        assert len(found) == 2

    def test_the_same_root_is_not_reported_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _install(tmp_path / "ac")
        monkeypatch.setenv(ENV_VAR, str(root))
        assert len(discover_ac_root(root)) == 1


class TestSteamLibraries:
    def test_reads_the_modern_library_index(self, tmp_path: Path) -> None:
        steam = tmp_path / "Steam"
        library = tmp_path / "Games"
        _install(library / "steamapps" / "common" / "assettocorsa")
        index = steam / "steamapps" / "libraryfolders.vdf"
        index.parent.mkdir(parents=True)
        index.write_text(
            '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"'
            + str(library).replace("\\", "\\\\")
            + '"\n\t}\n}\n',
            encoding="utf-8",
        )

        libraries = ac_install._libraries_from_index(index)
        assert library in [Path(str(item)) for item in libraries]

    def test_reads_the_older_numbered_form(self, tmp_path: Path) -> None:
        index = tmp_path / "libraryfolders.vdf"
        index.write_text('"1"\t\t"D:\\\\SteamLibrary"\n', encoding="utf-8")
        assert ac_install._libraries_from_index(index) == [Path("D:\\SteamLibrary")]

    def test_a_missing_index_costs_one_source_not_an_exception(self, tmp_path: Path) -> None:
        assert ac_install._libraries_from_index(tmp_path / "absent.vdf") == []


class TestTrackData:
    def test_finds_a_layout_scoped_data_folder(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        layout_dir = _track(root, "pk_akina", "akina_downhill")
        (layout_dir / "data" / "fast_lane.ai").write_bytes(b"\x00" * 32)
        (layout_dir / "data" / "cameras_1.ini").write_text("[HEADER]\n", encoding="utf-8")
        (layout_dir / "map.ini").write_text("[PARAMETERS]\n", encoding="utf-8")

        files = find_track_data(root, "pk_akina", "akina_downhill")
        assert files is not None
        assert files.layout_dir == layout_dir
        assert files.fast_lane is not None
        assert files.camera_sets == (layout_dir / "data" / "cameras_1.ini",)
        assert files.sizes()["fast_lane.ai"] == 32

    def test_map_falls_back_to_the_track_root(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        layout_dir = _track(root, "pk_akina", "akina_downhill")
        track_root = layout_dir.parent
        (track_root / "map.png").write_bytes(b"\x89PNG")

        files = find_track_data(root, "pk_akina", "akina_downhill")
        assert files is not None
        assert files.map_image == track_root / "map.png"

    def test_a_layout_the_install_lacks_is_not_silently_the_track_root(
        self, tmp_path: Path
    ) -> None:
        root = _install(tmp_path / "ac")
        _track(root, "pk_akina", "akina_downhill")
        assert find_track_data(root, "pk_akina", "akina_uphill") is None

    def test_an_uninstalled_track_is_none(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        assert find_track_data(root, "not_installed") is None

    def test_missing_names_what_was_looked_for(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        _track(root, "pk_akina")
        files = find_track_data(root, "pk_akina")
        assert files is not None
        assert "data/fast_lane.ai" in files.missing
        assert "map.ini" in files.missing
        assert files.sizes() == {}

    def test_an_empty_file_is_present_with_zero_size(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        layout_dir = _track(root, "pk_akina")
        (layout_dir / "data" / "fast_lane.ai").write_bytes(b"")
        files = find_track_data(root, "pk_akina")
        assert files is not None
        assert "data/fast_lane.ai" not in files.missing
        assert files.sizes()["fast_lane.ai"] == 0


class TestCspManifest:
    @staticmethod
    def _write(root: Path, body: bytes) -> Path:
        """Place a manifest inside an install.

        Args:
            root: The install root.
            body: Exact bytes to write, so encoding can be varied per test.

        Returns:
            The manifest path.
        """
        manifest = root / "extension" / "config" / "data_manifest.ini"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(body)
        return manifest

    def test_reports_the_version_section_verbatim(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        self._write(root, b"[VERSION]\nSHADERS_PATCH=0.2.7\nOTHER=1\n")

        csp = environment_snapshot(explicit_root=root).csp
        assert csp is not None
        assert csp.values == {"shaders_patch": "0.2.7", "other": "1"}

    def test_a_byte_order_mark_does_not_hide_the_install(self, tmp_path: Path) -> None:
        # The first manual run of `aidirector doctor` reported Custom Shaders
        # Patch as absent because the manifest began with a UTF-8 BOM, which is
        # what Windows tooling writes by default. A confident false negative is
        # worse than no answer, so this case is pinned.
        root = _install(tmp_path / "ac")
        self._write(root, b"\xef\xbb\xbf[VERSION]\nSHADERS_PATCH=0.2.7\n")

        csp = environment_snapshot(explicit_root=root).csp
        assert csp is not None
        assert csp.values == {"shaders_patch": "0.2.7"}

    def test_an_unreadable_manifest_is_not_reported_as_absent(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        manifest = self._write(root, b"this is not an ini file at all\n")

        snapshot = environment_snapshot(explicit_root=root)
        assert snapshot.csp is not None
        assert snapshot.csp.values is None
        assert not snapshot.csp.readable
        assert any("could not be parsed" in gap for gap in snapshot.blind_spots)
        assert any(str(manifest) in gap for gap in snapshot.blind_spots)

    def test_a_manifest_without_a_version_section_is_readable_and_empty(
        self, tmp_path: Path
    ) -> None:
        root = _install(tmp_path / "ac")
        self._write(root, b"[OTHER]\nkey=value\n")

        csp = environment_snapshot(explicit_root=root).csp
        assert csp is not None
        assert csp.values == {}
        assert csp.readable

    def test_absent_csp_is_none_not_an_error(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        assert environment_snapshot(explicit_root=root).csp is None


class TestSnapshot:
    def test_a_machine_without_the_game_reports_that_and_succeeds(self) -> None:
        snapshot = environment_snapshot("pk_akina", "akina_downhill")
        assert snapshot.install is None
        assert snapshot.candidates == ()
        assert snapshot.track is None
        assert any(ENV_VAR in gap for gap in snapshot.blind_spots)

    def test_an_install_without_a_named_track_says_so(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        snapshot = environment_snapshot(explicit_root=root)
        assert snapshot.install is not None
        assert snapshot.track is None
        assert snapshot.blind_spots == (
            "No track was named for this session, so its data was not looked up.",
        )

    def test_blind_spots_name_every_missing_file_once(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        _track(root, "pk_akina", "akina_downhill")
        snapshot = environment_snapshot("pk_akina", "akina_downhill", explicit_root=root)
        assert snapshot.track is not None
        # One sentence naming the directory once, rather than one sentence per
        # file each repeating an absolute path.
        assert len(snapshot.blind_spots) == 1
        gap = snapshot.blind_spots[0]
        assert "akina_downhill" in gap
        assert all(name in gap for name in snapshot.track.missing)

    def test_nothing_missing_is_an_empty_tuple(self, tmp_path: Path) -> None:
        root = _install(tmp_path / "ac")
        layout_dir = _track(root, "pk_akina", "akina_downhill")
        for name in ("fast_lane.ai", "surfaces.ini", "cameras_0.ini"):
            (layout_dir / "data" / name).write_bytes(b"x")
        (layout_dir / "map.png").write_bytes(b"x")
        (layout_dir / "map.ini").write_bytes(b"x")

        snapshot = environment_snapshot("pk_akina", "akina_downhill", explicit_root=root)
        assert snapshot.blind_spots == ()
