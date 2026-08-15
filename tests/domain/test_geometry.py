"""Tests for frame-tagged vectors and the camera basis."""

from __future__ import annotations

import math

import pytest

from assetto_ai_director.domain.frames import ReferenceFrame
from assetto_ai_director.domain.geometry import CameraBasis, Vec3


class TestFrameSafety:
    """Mixing frames is the bug the type is designed to prevent."""

    def test_same_frame_arithmetic_is_allowed(self) -> None:
        a = Vec3(1.0, 2.0, 3.0, ReferenceFrame.WORLD)
        b = Vec3(4.0, 5.0, 6.0, ReferenceFrame.WORLD)
        assert (a + b) == Vec3(5.0, 7.0, 9.0, ReferenceFrame.WORLD)

    def test_cross_frame_addition_is_refused(self) -> None:
        world = Vec3(1.0, 0.0, 0.0, ReferenceFrame.WORLD)
        local = Vec3(1.0, 0.0, 0.0, ReferenceFrame.CAR_LOCAL)
        with pytest.raises(ValueError, match="different frames"):
            _ = world + local

    def test_cross_frame_dot_is_refused(self) -> None:
        world = Vec3(1.0, 0.0, 0.0, ReferenceFrame.WORLD)
        unknown = Vec3(1.0, 0.0, 0.0, ReferenceFrame.UNKNOWN)
        with pytest.raises(ValueError, match="different frames"):
            world.dot(unknown)

    def test_retagging_does_not_rotate(self) -> None:
        # with_frame exists only for post-calibration relabelling; it must never
        # silently transform components.
        original = Vec3(1.0, 2.0, 3.0, ReferenceFrame.UNKNOWN)
        retagged = original.with_frame(ReferenceFrame.CAR_LOCAL)
        assert (retagged.x, retagged.y, retagged.z) == (1.0, 2.0, 3.0)
        assert retagged.frame is ReferenceFrame.CAR_LOCAL


class TestVectorMath:
    def test_length(self) -> None:
        assert Vec3(3.0, 0.0, 4.0).length == pytest.approx(5.0)

    def test_ground_length_ignores_elevation(self) -> None:
        # Speed on a downhill course is the ground-plane component; including
        # the vertical term inflates it on steep sections.
        assert Vec3(3.0, 100.0, 4.0).length_ground == pytest.approx(5.0)

    def test_normalised_is_unit_length(self) -> None:
        assert Vec3(1.0, 2.0, 3.0).normalised().length == pytest.approx(1.0)

    def test_normalising_zero_is_safe(self) -> None:
        assert Vec3(0.0, 0.0, 0.0).normalised().length == pytest.approx(0.0)

    def test_cross_product_is_perpendicular(self) -> None:
        a = Vec3(1.0, 0.0, 0.0)
        b = Vec3(0.0, 0.0, 1.0)
        assert a.cross(b).dot(a) == pytest.approx(0.0)

    def test_heading_uses_atan2_x_z(self) -> None:
        # +z is the zero bearing, rotating toward +x, matching the game's y-up
        # convention. Path tangent and car heading must share this or their
        # difference is not sideslip.
        assert Vec3(0.0, 0.0, 1.0).heading == pytest.approx(0.0)
        assert Vec3(1.0, 0.0, 0.0).heading == pytest.approx(math.pi / 2)
        assert Vec3(0.0, 0.0, -1.0).heading == pytest.approx(math.pi)

    def test_distance(self) -> None:
        assert Vec3(0.0, 0.0, 0.0).distance_to(Vec3(0.0, 3.0, 4.0)) == pytest.approx(5.0)


class TestCameraBasis:
    def test_looking_at_produces_direction_not_target(self) -> None:
        basis = CameraBasis.looking_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        assert basis.look.length == pytest.approx(1.0)
        assert basis.look.z == pytest.approx(1.0)

    def test_orthonormalised_basis_is_orthogonal(self) -> None:
        basis = CameraBasis(
            Vec3(0.0, 5.0, 0.0), Vec3(1.0, -1.0, 1.0), Vec3(0.2, 1.0, 0.0)
        ).orthonormalised()
        assert basis.look.dot(basis.up) == pytest.approx(0.0, abs=1e-9)
        assert basis.look.length == pytest.approx(1.0)
        assert basis.up.length == pytest.approx(1.0)

    def test_straight_down_does_not_degenerate(self) -> None:
        # A helicopter shot looking directly down makes the default up vector
        # parallel to look; the fallback must still yield a usable basis.
        basis = CameraBasis(
            Vec3(0.0, 50.0, 0.0), Vec3(0.0, -1.0, 0.0), Vec3(0.0, 1.0, 0.0)
        ).orthonormalised()
        assert basis.up.length == pytest.approx(1.0)
        assert basis.look.dot(basis.up) == pytest.approx(0.0, abs=1e-9)

    def test_distance_to_subject(self) -> None:
        basis = CameraBasis.looking_at(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0))
        assert basis.distance_to(Vec3(0.0, 0.0, 12.0)) == pytest.approx(12.0)
