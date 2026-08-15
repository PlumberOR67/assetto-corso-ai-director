"""Tests for reading what a track knows about itself.

The AI-line tests are the load-bearing ones, and what they pin is not that the
layout is right — this project has never seen a real ``fast_lane.ai`` — but that
a **wrong** layout is refused rather than decoded into a convincing centreline in
the wrong place. Every corruption below is a way the file could disagree with the
assumption, and every one of them must raise.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from assetto_ai_director.domain.errors import (
    StrideMismatchError,
    TruncatedReplayError,
    UnsupportedReplayVersionError,
)
from assetto_ai_director.domain.track import SplineSource
from assetto_ai_director.infrastructure.track import (
    read_cameras_ini,
    read_fast_lane,
    read_map_ini,
)
from assetto_ai_director.infrastructure.track.fast_lane import EXTRA_STRIDE

_SIDE_LEFT_INDEX = 5
_SIDE_RIGHT_INDEX = 6


def _fast_lane(
    points: np.ndarray,
    *,
    version: int = 7,
    left: float = 5.0,
    right: float = 5.0,
    extra_count: int | None = None,
    trailing: bytes = b"",
) -> bytes:
    """Build a fast_lane.ai file to the layout the reader assumes.

    Args:
        points: World positions, shape ``(n, 3)``.
        version: Header version to declare.
        left: Left half width to write into every extra record.
        right: Right half width.
        extra_count: Override the declared extra count, to make it disagree.
        trailing: Bytes to append past the end, to break the size check.

    Returns:
        The file's bytes.
    """
    count = points.shape[0]
    body = struct.pack("<4i", version, count, 0, count)
    for index, point in enumerate(points):
        body += struct.pack("<3f", *point) + struct.pack("<f", index / count)
        body += struct.pack("<i", index)

    declared = count if extra_count is None else extra_count
    body += struct.pack("<i", declared)
    for _ in range(declared):
        floats = [0.0] * (EXTRA_STRIDE // 4)
        floats[_SIDE_LEFT_INDEX] = left
        floats[_SIDE_RIGHT_INDEX] = right
        body += struct.pack(f"<{len(floats)}f", *floats)
    return body + trailing


def _circle(count: int = 64, radius: float = 200.0) -> np.ndarray:
    """A closed circuit, as world positions.

    Args:
        count: How many points.
        radius: How big.

    Returns:
        Positions, shape ``(count, 3)``.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack(
        [radius * np.cos(angle), np.zeros(count), radius * np.sin(angle)], axis=1
    ).astype(np.float32)


class TestFastLaneAcceptsWhatItUnderstands:
    def test_a_well_formed_file_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "fast_lane.ai"
        points = _circle()
        path.write_bytes(_fast_lane(points))

        spline = read_fast_lane(path)
        assert spline.source is SplineSource.AI_LINE
        assert spline.points.shape == points.shape
        assert spline.knows_width
        assert spline.closed

    def test_the_widths_come_back(self, tmp_path: Path) -> None:
        path = tmp_path / "fast_lane.ai"
        path.write_bytes(_fast_lane(_circle(), left=6.5, right=4.5))
        spline = read_fast_lane(path)
        assert spline.left_width is not None
        assert float(spline.left_width[0]) == pytest.approx(6.5)
        assert spline.right_width is not None
        assert float(spline.right_width[0]) == pytest.approx(4.5)

    def test_a_point_to_point_course_is_not_reported_as_a_circuit(self, tmp_path: Path) -> None:
        path = tmp_path / "fast_lane.ai"
        straight = np.stack(
            [np.zeros(50), np.zeros(50), np.linspace(0.0, 2000.0, 50)], axis=1
        ).astype(np.float32)
        path.write_bytes(_fast_lane(straight))
        assert not read_fast_lane(path).closed


class TestFastLaneRefusesWhatItDoesNot:
    """A wrong layout must raise, not decode into a plausible wrong track."""

    def test_an_unknown_version_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "fast_lane.ai"
        path.write_bytes(_fast_lane(_circle(), version=6))
        with pytest.raises(UnsupportedReplayVersionError):
            read_fast_lane(path)

    def test_trailing_bytes_break_the_size_check(self, tmp_path: Path) -> None:
        # The whole safety argument: if the assumed record sizes were wrong, the
        # parse would land somewhere other than the last byte, exactly like this.
        path = tmp_path / "fast_lane.ai"
        path.write_bytes(_fast_lane(_circle(), trailing=b"\x00" * 9))
        with pytest.raises(StrideMismatchError):
            read_fast_lane(path)

    def test_a_declared_count_the_file_cannot_hold_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "fast_lane.ai"
        # Claim a thousand points in a file sized for sixty-four, before anything
        # is allocated on the strength of the claim.
        data = bytearray(_fast_lane(_circle()))
        data[4:8] = struct.pack("<i", 1000)
        path.write_bytes(bytes(data))
        with pytest.raises(TruncatedReplayError):
            read_fast_lane(path)

    def test_an_empty_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "fast_lane.ai"
        path.write_bytes(b"")
        with pytest.raises(TruncatedReplayError):
            read_fast_lane(path)

    def test_implausible_widths_are_dropped_rather_than_reported(self, tmp_path: Path) -> None:
        # A misread float lands anywhere. Reporting a 4000 m half width would
        # make every camera pass the road-edge check.
        path = tmp_path / "fast_lane.ai"
        path.write_bytes(_fast_lane(_circle(), left=4000.0))
        spline = read_fast_lane(path)
        assert not spline.knows_width

    def test_an_extra_block_that_does_not_match_the_points_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "fast_lane.ai"
        points = _circle(64)
        path.write_bytes(_fast_lane(points, extra_count=32))
        # Half as many extra records: the file still adds up, so it parses, but
        # the widths cannot be matched to points and must not be guessed at.
        spline = read_fast_lane(path)
        assert not spline.knows_width
        assert spline.points.shape[0] == 64


class TestMapTransform:
    @staticmethod
    def _write(path: Path, body: str) -> Path:
        path.write_text(body, encoding="utf-8")
        return path

    def test_the_transform_round_trips(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path / "map.ini",
            "[PARAMETERS]\nWIDTH=1024\nHEIGHT=768\nX_OFFSET=500\nZ_OFFSET=300\n"
            "SCALE_FACTOR=2.0\nMARGIN=20\n",
        )
        transform = read_map_ini(path)
        world = np.array([[10.0, 3.0, -40.0], [0.0, 0.0, 0.0]])
        back = transform.to_world(transform.to_pixels(world))
        assert back[:, 0] == pytest.approx(world[:, 0])
        assert back[:, 2] == pytest.approx(world[:, 2])

    def test_elevation_takes_no_part(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path / "map.ini",
            "[PARAMETERS]\nWIDTH=100\nHEIGHT=100\nX_OFFSET=0\nZ_OFFSET=0\nSCALE_FACTOR=1\n",
        )
        transform = read_map_ini(path)
        low = transform.to_pixels(np.array([[5.0, 0.0, 5.0]]))
        high = transform.to_pixels(np.array([[5.0, 900.0, 5.0]]))
        assert low == pytest.approx(high)

    def test_a_line_outside_the_image_is_reported(self, tmp_path: Path) -> None:
        # The check that answers "is the map positioned incorrectly".
        path = self._write(
            tmp_path / "map.ini",
            "[PARAMETERS]\nWIDTH=100\nHEIGHT=100\nX_OFFSET=0\nZ_OFFSET=0\nSCALE_FACTOR=1\n",
        )
        transform = read_map_ini(path)
        assert transform.covers(np.array([[10.0, 0.0, 10.0]]))
        assert not transform.covers(np.array([[10.0, 0.0, 10.0], [900.0, 0.0, 10.0]]))

    def test_a_missing_scale_is_refused_rather_than_defaulted(self, tmp_path: Path) -> None:
        # Defaulting it to 1.0 produces a map wrong by a factor of a hundred that
        # looks exactly like a correct map of a very short track.
        path = self._write(
            tmp_path / "map.ini", "[PARAMETERS]\nWIDTH=100\nHEIGHT=100\nX_OFFSET=0\nZ_OFFSET=0\n"
        )
        with pytest.raises(ValueError, match="SCALE_FACTOR"):
            read_map_ini(path)

    def test_a_zero_scale_is_refused(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path / "map.ini",
            "[PARAMETERS]\nWIDTH=100\nHEIGHT=100\nX_OFFSET=0\nZ_OFFSET=0\nSCALE_FACTOR=0\n",
        )
        with pytest.raises(ValueError, match="scale factor"):
            read_map_ini(path)

    def test_a_file_with_no_parameters_section_is_refused(self, tmp_path: Path) -> None:
        path = self._write(tmp_path / "map.ini", "[SOMETHING]\nKEY=1\n")
        with pytest.raises(ValueError, match="PARAMETERS"):
            read_map_ini(path)

    def test_a_byte_order_mark_does_not_break_it(self, tmp_path: Path) -> None:
        path = tmp_path / "map.ini"
        path.write_bytes(
            b"\xef\xbb\xbf[PARAMETERS]\nWIDTH=100\nHEIGHT=100\nX_OFFSET=0\n"
            b"Z_OFFSET=0\nSCALE_FACTOR=1\n"
        )
        assert read_map_ini(path).width == 100.0


_CAMERAS_INI = """[HEADER]
VERSION=3
CAMERA_COUNT=2
SET_NAME=Kunos

[CAMERA_0]
NAME=Camera 1
POSITION=10.5,2.0,-30.25
FORWARD=0.0,0.0,1.0
UP=0.0,1.0,0.0
MIN_FOV=22.0
MAX_FOV=52.0
IN_POINT=0.0
OUT_POINT=0.5
SPLINE=
IS_FIXED=1

[CAMERA_1]
NAME=Camera 2
POSITION=-4,1,8
FORWARD=1,0,0
MIN_FOV=30
MAX_FOV=60
IN_POINT=0.5
OUT_POINT=1.0
NEAR_PLANE=0.5
FAR_PLANE=8000
"""


class TestCameraSetReader:
    @staticmethod
    def _write(tmp_path: Path, body: str = _CAMERAS_INI) -> Path:
        path = tmp_path / "cameras_1.ini"
        path.write_text(body, encoding="utf-8")
        return path

    def test_a_set_is_read(self, tmp_path: Path) -> None:
        result = read_cameras_ini(self._write(tmp_path))
        assert result.version == 3
        assert result.set_name == "Kunos"
        assert len(result.cameras) == 2
        assert result.count_agrees

    def test_geometry_survives(self, tmp_path: Path) -> None:
        camera = read_cameras_ini(self._write(tmp_path)).cameras[0]
        assert camera.position == pytest.approx((10.5, 2.0, -30.25))
        assert camera.forward == pytest.approx((0.0, 0.0, 1.0))
        assert camera.name == "Camera 1"

    def test_min_fov_is_kept_as_min_fov(self, tmp_path: Path) -> None:
        # MIN_FOV is the field of view when the car is furthest, so it is
        # routinely the larger number. Sorting them into near and far on the way
        # in would make this reader useless for checking the writer.
        camera = read_cameras_ini(self._write(tmp_path)).cameras[0]
        assert camera.min_fov == pytest.approx(22.0)
        assert camera.max_fov == pytest.approx(52.0)

    def test_an_absent_up_vector_defaults_but_absent_geometry_does_not(
        self, tmp_path: Path
    ) -> None:
        cameras = read_cameras_ini(self._write(tmp_path)).cameras
        assert cameras[1].up == pytest.approx((0.0, 1.0, 0.0))

        missing = _CAMERAS_INI.replace("POSITION=-4,1,8\n", "")
        with pytest.raises(ValueError, match="POSITION"):
            read_cameras_ini(self._write(tmp_path, missing))

    def test_unknown_keys_are_kept_rather_than_dropped(self, tmp_path: Path) -> None:
        # A round trip that silently discards fields cannot be used to check a
        # writer against a real file, which is half the reason this reader exists.
        camera = read_cameras_ini(self._write(tmp_path)).cameras[1]
        assert dict(camera.extra)["NEAR_PLANE"] == "0.5"
        assert dict(camera.extra)["FAR_PLANE"] == "8000"

    def test_contiguous_spans_are_recognised(self, tmp_path: Path) -> None:
        assert read_cameras_ini(self._write(tmp_path)).spans_are_contiguous

    def test_a_gap_between_spans_is_recognised(self, tmp_path: Path) -> None:
        gapped = _CAMERAS_INI.replace("IN_POINT=0.5", "IN_POINT=0.7")
        assert not read_cameras_ini(self._write(tmp_path, gapped)).spans_are_contiguous

    def test_a_header_that_disagrees_with_the_body_is_reported_not_corrected(
        self, tmp_path: Path
    ) -> None:
        wrong = _CAMERAS_INI.replace("CAMERA_COUNT=2", "CAMERA_COUNT=9")
        result = read_cameras_ini(self._write(tmp_path, wrong))
        assert result.declared_count == 9
        assert len(result.cameras) == 2
        assert not result.count_agrees
