"""Tests for the decoded car-frame channels.

These assert *physics*, not byte offsets. An offset test only proves the code
reads the byte it was told to; a physics test proves the byte means what the
module claims. Since the record layout is undocumented, the physics is the only
real evidence available — so the suite checks that trajectories descend on a
downhill, that wheels behave like a rigid body, and that sideslip lands in the
range real driving produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from assetto_ai_director.infrastructure.acreplay.frames import CarFrames, read_car_frames
from assetto_ai_director.infrastructure.acreplay.structure import read_structure

REPLAY_DIR = Path(__file__).resolve().parents[2] / "replays"
FIXTURES = sorted(REPLAY_DIR.glob("*.acreplay")) if REPLAY_DIR.is_dir() else []

pytestmark = [
    pytest.mark.fixtures,
    pytest.mark.skipif(not FIXTURES, reason="no .acreplay fixtures in replays/"),
]

MAX_FRAMES = 12000
"""Frames decoded per test. Enough for stable statistics, quick enough for CI."""


@pytest.fixture(scope="module")
def frames() -> CarFrames:
    structure = read_structure(FIXTURES[0])
    block = structure.cars[0]
    trimmed = type(block)(
        identity=block.identity,
        frame_count=min(block.frame_count, MAX_FRAMES),
        wing_count=block.wing_count,
        frames_offset=block.frames_offset,
        extra_count=block.extra_count,
    )
    return read_car_frames(FIXTURES[0], trimmed, float(structure.sample_rate))


class TestPosition:
    def test_positions_are_finite(self, frames: CarFrames) -> None:
        assert np.isfinite(frames.position).all()

    def test_trajectory_is_continuous(self, frames: CarFrames) -> None:
        # A car cannot teleport. A wrong offset produces wild jumps everywhere;
        # a session reset produces at most a handful.
        step = np.linalg.norm(np.diff(frames.position, axis=0), axis=1)
        assert (step > 10.0).sum() <= 2

    def test_track_descends(self, frames: CarFrames) -> None:
        # The fixtures are recorded on akina_downhill. If the y component is
        # really elevation, the car must lose height overall.
        elevation = frames.position[:, 1]
        assert elevation[-1] < elevation[0]
        assert elevation[0] - elevation[-1] > 20.0

    def test_speed_is_plausible_for_a_touge_run(self, frames: CarFrames) -> None:
        moving = frames.speed[frames.speed > 2.0]
        median_kmh = float(np.median(moving)) * 3.6
        assert 30.0 < median_kmh < 180.0
        assert float(np.percentile(moving, 95)) * 3.6 < 300.0


class TestWheelGeometry:
    def test_wheels_form_a_rigid_body(self, frames: CarFrames) -> None:
        # Wheel-to-body distances must be constant: the car is not made of rubber.
        for wheel in range(4):
            distance = np.linalg.norm(frames.wheel_positions[:, wheel, :] - frames.position, axis=1)
            assert distance.std() < 0.05, f"wheel {wheel} distance varies by {distance.std():.3f} m"

    def test_wheelbase_matches_the_car(self, frames: CarFrames) -> None:
        # The fixture cars are AE86s: 2400 mm wheelbase. Random bytes do not
        # reproduce a specific car's dimensions.
        wheels = frames.wheel_positions
        sample = wheels[0]
        separations = sorted(
            float(np.linalg.norm(sample[a] - sample[b]))
            for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        )
        # Two track widths, two wheelbase-ish, two diagonals.
        assert 1.0 < separations[0] < 2.0, f"track width {separations[0]:.2f} m"
        assert 2.0 < separations[2] < 3.2, f"wheelbase {separations[2]:.2f} m"

    def test_wheels_sit_below_the_body_reference(self, frames: CarFrames) -> None:
        offsets = frames.wheel_positions[:, :, 1] - frames.position[:, None, 1]
        assert (offsets.mean(axis=0) < 0.0).all()
        assert (np.abs(offsets.mean(axis=0)) < 1.5).all()


class TestHeadingAndSideslip:
    def test_heading_is_bounded(self, frames: CarFrames) -> None:
        assert np.isfinite(frames.heading).all()
        assert np.abs(frames.heading).max() <= np.pi + 1e-9

    def test_heading_agrees_with_travel_most_of_the_time(self, frames: CarFrames) -> None:
        # The decisive check that the front/rear axle assignment is right. A
        # reversed assignment yields a median near 180 degrees, not near zero.
        beta = np.degrees(np.abs(frames.sideslip[frames.speed > 5.0]))
        assert float(np.median(beta)) < 10.0

    def test_sideslip_shows_real_drifting(self, frames: CarFrames) -> None:
        # An AE86 on Akina downhill drifts. If sideslip were noise or zero, this
        # would fail — and so would the entire premise of the drift analysis.
        beta = np.degrees(np.abs(frames.sideslip[frames.speed > 5.0]))
        drifting = float((beta > 10.0).mean())
        assert 0.02 < drifting < 0.60, f"{drifting:.1%} of frames above 10 deg"

    def test_sideslip_is_wrapped(self, frames: CarFrames) -> None:
        assert np.abs(frames.sideslip).max() <= np.pi + 1e-9

    def test_stationary_frames_have_no_sideslip(self, frames: CarFrames) -> None:
        # atan2 of millimetre jitter is uniformly random; it must not be
        # reported as a drift.
        stationary = frames.speed < 1.0
        if stationary.any():
            assert np.abs(frames.sideslip[stationary]).max() == 0.0


class TestAllFixtures:
    @pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name[-24:])
    def test_every_fixture_decodes_plausibly(self, path: Path) -> None:
        structure = read_structure(path)
        block = structure.cars[0]
        trimmed = type(block)(
            identity=block.identity,
            frame_count=min(block.frame_count, 4000),
            wing_count=block.wing_count,
            frames_offset=block.frames_offset,
            extra_count=block.extra_count,
        )
        decoded = read_car_frames(path, trimmed, float(structure.sample_rate))
        moving = decoded.speed[decoded.speed > 2.0]
        assert moving.size > 0
        assert 20.0 < float(np.median(moving)) * 3.6 < 200.0
        assert np.isfinite(decoded.heading).all()
