"""Tests for the ``.acreplay`` structural reader.

Split in two. The synthetic tests build files byte by byte and always run: they
prove the reader's failure modes are correct, which is the part that matters most
because the alternative to a loud failure is plausible garbage.

The fixture tests run only when real replays are present in ``replays/``. Real
files are large and user-owned, so they are not committed; the suite degrades to
the synthetic tests rather than failing when they are absent.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from assetto_ai_director.domain.errors import (
    ReplayError,
    StrideMismatchError,
    TruncatedReplayError,
    UnsupportedReplayVersionError,
)
from assetto_ai_director.infrastructure.acreplay.layout import (
    CSP_MARKER,
    LAP_RECORD_BYTES,
    PER_CAR_EXTRA_RECORD_BYTES,
    car_frame_stride,
)
from assetto_ai_director.infrastructure.acreplay.structure import read_structure

REPLAY_DIR = Path(__file__).resolve().parents[2] / "replays"
FIXTURES = sorted(REPLAY_DIR.glob("*.acreplay")) if REPLAY_DIR.is_dir() else []


# --------------------------------------------------------------------------
# Synthetic replay construction
# --------------------------------------------------------------------------


def _pascal(text: str) -> bytes:
    return struct.pack("<i", len(text)) + text.encode("latin-1")


def _build_replay(
    *,
    version: int = 16,
    interval_ms: float = 30.0,
    cars: list[tuple[str, str, int]] | None = None,
    frame_count: int = 4,
    track_object_count: int = 0,
    lap_record_count: int = 2,
    player_extras: int = 3,
    with_csp: bool = True,
    corrupt_stride_by: int = 0,
) -> bytes:
    """Build a structurally valid (or deliberately broken) replay.

    Args:
        version: Header version field.
        interval_ms: Recording interval.
        cars: ``(model, driver, wing_count)`` per car.
        frame_count: Session frame count.
        track_object_count: Animated track object count.
        lap_record_count: Session trailer record count.
        player_extras: Trailing records on car zero, which marks it as the player.
        with_csp: Whether to append the CSP footer.
        corrupt_stride_by: Bytes to add to car zero's frame region, simulating a
            wrong stride assumption.

    Returns:
        The encoded file.
    """
    cars = cars or [("ks_toyota_ae86_tuned", "ducheck", 3), ("ks_mazda_rx7", "", 5)]
    car_frame_count = max(frame_count - 1, 0)

    body = bytearray()
    body += struct.pack("<I", version)
    body += struct.pack("<d", interval_ms)
    body += _pascal("7_heavy_clouds")
    body += _pascal("pk_akina")
    body += _pascal("akina_downhill")
    body += struct.pack("<IIII", len(cars), frame_count, frame_count, track_object_count)
    body += bytes(frame_count * (4 + 12 * track_object_count))

    for index, (model, driver, wings) in enumerate(cars):
        body += _pascal(model)
        body += _pascal(driver)
        body += _pascal("RUS" if driver else "")
        body += _pascal("")
        body += _pascal(f"{index:02d}_skin")
        body += struct.pack("<II", car_frame_count, wings)
        extra = corrupt_stride_by if index == 0 else 0
        body += bytes(car_frame_count * car_frame_stride(wings) + extra)
        extras = player_extras if index == 0 else 0
        body += struct.pack("<I", extras)
        body += bytes(extras * PER_CAR_EXTRA_RECORD_BYTES)

    body += struct.pack("<I", lap_record_count)
    body += bytes(len(cars) * lap_record_count * LAP_RECORD_BYTES)

    if with_csp:
        block_offset = len(body)
        body += b"csp-extension-payload"
        body += CSP_MARKER + struct.pack("<II", block_offset, 1)
    return bytes(body)


@pytest.fixture
def replay_file(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.acreplay"
    path.write_bytes(_build_replay())
    return path


# --------------------------------------------------------------------------
# Synthetic tests — always run
# --------------------------------------------------------------------------


class TestStructuralParse:
    def test_reads_header_fields(self, replay_file: Path) -> None:
        structure = read_structure(replay_file)
        assert structure.version == 16
        assert structure.track_id == "pk_akina"
        assert structure.track_config == "akina_downhill"
        assert structure.weather_id == "7_heavy_clouds"
        assert structure.sample_rate == pytest.approx(1000.0 / 30.0)

    def test_verifies_against_the_files_own_footer(self, replay_file: Path) -> None:
        assert read_structure(replay_file).verified

    def test_variable_stride_is_honoured(self, replay_file: Path) -> None:
        # The finding published parsers get wrong: the record is not fixed-size.
        structure = read_structure(replay_file)
        assert [block.stride for block in structure.cars] == [288, 296]

    def test_player_is_identified_structurally(self, replay_file: Path) -> None:
        # By the presence of trailing extra records, not by assuming car zero.
        structure = read_structure(replay_file)
        assert structure.cars[0].identity.is_player
        assert not structure.cars[1].identity.is_player

    def test_empty_strings_become_none(self, replay_file: Path) -> None:
        assert read_structure(replay_file).cars[1].identity.driver_name is None

    def test_frames_offset_points_past_the_car_header(self, replay_file: Path) -> None:
        structure = read_structure(replay_file)
        first, second = structure.cars
        assert second.frames_offset > first.frames_offset + first.frames_length

    def test_works_without_a_csp_block(self, tmp_path: Path) -> None:
        path = tmp_path / "no_csp.acreplay"
        path.write_bytes(_build_replay(with_csp=False))
        structure = read_structure(path)
        assert structure.csp_block_offset is None
        assert structure.verified

    def test_ring_buffer_wrap_is_reported(self, tmp_path: Path) -> None:
        raw = bytearray(_build_replay())
        # current_recording_index sits immediately after the car count.
        marker = raw.index(_pascal("akina_downhill")) + len(_pascal("akina_downhill"))
        struct.pack_into("<I", raw, marker + 4, 1)
        path = tmp_path / "wrapped.acreplay"
        path.write_bytes(bytes(raw))
        assert read_structure(path).wrapped


class TestFailureModes:
    """The reader must fail loudly rather than produce plausible garbage."""

    def test_unsupported_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "v15.acreplay"
        path.write_bytes(_build_replay(version=15))
        with pytest.raises(UnsupportedReplayVersionError) as excinfo:
            read_structure(path)
        assert excinfo.value.version == 15

    @pytest.mark.parametrize("drift", [-8, -4, 4, 8, 288])
    def test_wrong_stride_always_fails_loudly(self, tmp_path: Path, drift: int) -> None:
        """The single most important guarantee in the project.

        A stride wrong by even four bytes must raise, because the alternative is
        smooth, believable, entirely fictitious telemetry.

        The *class* of error is not the guarantee — a bad stride may surface
        either as a nonsense read part-way through the walk or as an end-offset
        mismatch, depending on where the misalignment lands. What matters is that
        no misaligned file ever yields a structure.
        """
        path = tmp_path / f"bad_stride_{drift}.acreplay"
        path.write_bytes(_build_replay(corrupt_stride_by=drift))
        with pytest.raises(ReplayError):
            read_structure(path)

    def test_end_offset_mismatch_is_caught(self, tmp_path: Path) -> None:
        """A walk that completes but ends in the wrong place is still a failure.

        Built by pointing the file's own footer somewhere else, which is exactly
        what a subtly wrong layout assumption would look like from the reader's
        point of view.
        """
        raw = bytearray(_build_replay())
        footer_offset = len(raw) - 8
        real_offset = struct.unpack_from("<I", raw, footer_offset)[0]
        struct.pack_into("<I", raw, footer_offset, real_offset - 16)
        path = tmp_path / "offset_mismatch.acreplay"
        path.write_bytes(bytes(raw))
        with pytest.raises(StrideMismatchError) as excinfo:
            read_structure(path)
        assert excinfo.value.actual_end != excinfo.value.expected_end

    def test_truncated_file_raises_before_allocating(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.acreplay"
        path.write_bytes(_build_replay()[:120])
        with pytest.raises(TruncatedReplayError):
            read_structure(path)

    def test_absurd_frame_count_is_rejected(self, tmp_path: Path) -> None:
        raw = bytearray(_build_replay())
        offset = raw.index(_pascal("akina_downhill")) + len(_pascal("akina_downhill"))
        struct.pack_into("<I", raw, offset + 8, 2**30)
        path = tmp_path / "absurd.acreplay"
        path.write_bytes(bytes(raw))
        with pytest.raises(TruncatedReplayError):
            read_structure(path)

    def test_implausible_string_length_is_rejected(self, tmp_path: Path) -> None:
        raw = bytearray(_build_replay())
        struct.pack_into("<i", raw, 12, 2**20)  # weather string length
        path = tmp_path / "bad_string.acreplay"
        path.write_bytes(bytes(raw))
        with pytest.raises(TruncatedReplayError):
            read_structure(path)


# --------------------------------------------------------------------------
# Fixture tests — the real acceptance criterion
# --------------------------------------------------------------------------


@pytest.mark.fixtures
@pytest.mark.skipif(not FIXTURES, reason="no .acreplay fixtures in replays/")
class TestRealReplays:
    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name[-24:])
    def test_layout_verifies_exactly(self, path: Path) -> None:
        """The acceptance criterion: land precisely on the file's own offset.

        This is what licenses the reader to exist under the project's
        never-invent-a-format rule. The file carries its own oracle.
        """
        structure = read_structure(path)
        assert structure.verified, (
            f"structural walk ended at {structure.structural_end}, "
            f"file says data ends at {structure.csp_block_offset}"
        )

    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name[-24:])
    def test_metadata_is_sane(self, path: Path) -> None:
        structure = read_structure(path)
        assert structure.version == 16
        assert structure.track_id
        assert structure.cars
        assert structure.sample_rate > 0
        assert all(block.frame_count > 0 for block in structure.cars)

    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name[-24:])
    def test_exactly_one_player(self, path: Path) -> None:
        structure = read_structure(path)
        players = [block for block in structure.cars if block.identity.is_player]
        assert len(players) == 1

    def test_variable_stride_occurs_in_real_data(self) -> None:
        """Guards the finding that motivated the variable-stride formula.

        If every car in every fixture had the same wing count, the formula would
        be untested against reality and a fixed-stride reader would pass.
        """
        strides = {block.stride for path in FIXTURES for block in read_structure(path).cars}
        assert len(strides) > 1, f"expected mixed strides in the fixtures, saw {strides}"
