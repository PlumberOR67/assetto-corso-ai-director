"""Tests for projecting a world point into a camera's frame.

**Analytic, not golden.** Every expectation here is a number that geometry
requires: a point the camera looks straight at must land at the centre; a point
at exactly half the vertical field of view must land on the edge; a point behind
the camera must report negative depth. Comparing against a recorded image would
pin whatever the code did on the day it was written, including its mistakes.

The handedness tests matter most. The rest of the codebase asserts, without ever
having measured it, that Assetto Corsa's world is left-handed with screen-right
along ``+x``. These tests do not prove that assertion true — nothing here can,
short of the game — but they do prove that projection and
:meth:`CameraBasis.orthonormalised` agree with each other, so a future correction
moves both together instead of leaving a diagnostic that mirrors the thing it
diagnoses.
"""

from __future__ import annotations

import math

import pytest

from assetto_ai_director.domain.geometry import CameraBasis, Vec3
from assetto_ai_director.domain.projection import (
    DEFAULT_ASPECT,
    angle_off_axis,
    frame_fraction,
    project_point,
)
from assetto_ai_director.domain.units import FovDegrees, Metres

_FOV = FovDegrees(40.0)


def _camera(position: Vec3, target: Vec3) -> CameraBasis:
    """Build a camera aimed at a point.

    Args:
        position: Where the camera sits.
        target: What it looks at.

    Returns:
        An orthonormal basis.
    """
    return CameraBasis.looking_at(position, target)


class TestCentring:
    def test_the_aim_point_lands_dead_centre(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        point = project_point(basis, Vec3(0.0, 0.0, 10.0), _FOV)
        assert point.u == pytest.approx(0.0, abs=1e-9)
        assert point.v == pytest.approx(0.0, abs=1e-9)
        assert float(point.depth) == pytest.approx(10.0)
        assert point.in_frame

    def test_centring_holds_for_an_arbitrary_aim(self) -> None:
        basis = _camera(Vec3(12.0, 3.5, -7.0), Vec3(-4.0, 1.0, 22.0))
        point = project_point(basis, Vec3(-4.0, 1.0, 22.0), _FOV)
        assert point.u == pytest.approx(0.0, abs=1e-9)
        assert point.v == pytest.approx(0.0, abs=1e-9)


class TestFrameEdges:
    def test_half_the_vertical_field_of_view_lands_on_the_top_edge(self) -> None:
        depth = 10.0
        offset = depth * math.tan(math.radians(float(_FOV)) / 2.0)
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, depth))
        point = project_point(basis, Vec3(0.0, offset, depth), _FOV)
        assert point.v == pytest.approx(1.0)
        assert point.in_frame

    def test_the_aspect_ratio_widens_the_horizontal_edge(self) -> None:
        depth = 10.0
        vertical = depth * math.tan(math.radians(float(_FOV)) / 2.0)
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, depth))
        point = project_point(basis, Vec3(vertical * DEFAULT_ASPECT, 0.0, depth), _FOV)
        assert point.u == pytest.approx(1.0)

    def test_just_outside_the_edge_reports_how_far(self) -> None:
        depth = 10.0
        vertical = depth * math.tan(math.radians(float(_FOV)) / 2.0)
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, depth))
        point = project_point(basis, Vec3(0.0, vertical * 1.5, depth), _FOV)
        assert not point.in_frame
        assert point.off_frame_by == pytest.approx(0.5)

    def test_a_point_inside_the_frame_is_not_off_it(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        assert project_point(basis, Vec3(0.0, 0.0, 10.0), _FOV).off_frame_by == 0.0


class TestBehindTheCamera:
    def test_depth_goes_negative(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        point = project_point(basis, Vec3(0.0, 0.0, -10.0), _FOV)
        assert float(point.depth) == pytest.approx(-10.0)
        assert not point.in_front
        assert not point.in_frame

    def test_a_point_behind_is_never_reported_as_framed(self) -> None:
        # The failure this guards: a naive projection divides by a negative
        # depth, lands the point at plausible screen coordinates, and reports a
        # shot of empty track as well framed.
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        point = project_point(basis, Vec3(0.5, 0.5, -10.0), _FOV)
        assert not point.in_frame
        assert point.off_frame_by == math.inf


class TestHandedness:
    def test_screen_right_is_world_plus_x_when_looking_along_plus_z(self) -> None:
        # Inherited from CameraBasis.orthonormalised(), which builds right as
        # up x look. Pinned here so projection and that method cannot disagree.
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        assert project_point(basis, Vec3(1.0, 0.0, 10.0), _FOV).u > 0.0

    def test_screen_up_is_world_plus_y(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        assert project_point(basis, Vec3(0.0, 1.0, 10.0), _FOV).v > 0.0

    def test_turning_the_camera_turns_the_frame_with_it(self) -> None:
        # Looking along +x, a point further along +z must now appear on the
        # opposite side from where it sat when looking along +z.
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(10.0, 0.0, 0.0))
        assert project_point(basis, Vec3(10.0, 0.0, 1.0), _FOV).u < 0.0


class TestValidation:
    @pytest.mark.parametrize("fov", [0.0, -5.0, 180.0, 200.0])
    def test_an_impossible_lens_is_refused(self, fov: float) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        with pytest.raises(ValueError, match="Field of view"):
            project_point(basis, Vec3(0.0, 0.0, 1.0), FovDegrees(fov))

    def test_a_non_positive_aspect_is_refused(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        with pytest.raises(ValueError, match="Aspect ratio"):
            project_point(basis, Vec3(0.0, 0.0, 1.0), _FOV, aspect=0.0)


class TestFrameFraction:
    def test_an_object_spanning_the_frame_height_scores_one(self) -> None:
        depth = Metres(10.0)
        height = Metres(2.0 * float(depth) * math.tan(math.radians(float(_FOV)) / 2.0))
        assert frame_fraction(height, depth, _FOV) == pytest.approx(1.0)

    def test_doubling_the_distance_halves_the_apparent_size(self) -> None:
        near = frame_fraction(Metres(1.3), Metres(10.0), _FOV)
        far = frame_fraction(Metres(1.3), Metres(20.0), _FOV)
        assert far == pytest.approx(near / 2.0)

    def test_an_object_at_the_camera_has_no_apparent_size(self) -> None:
        assert frame_fraction(Metres(1.3), Metres(0.0), _FOV) == 0.0
        assert frame_fraction(Metres(1.3), Metres(-5.0), _FOV) == 0.0


class TestAngleOffAxis:
    def test_the_aim_point_is_on_axis(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        assert float(angle_off_axis(basis, Vec3(0.0, 0.0, 10.0))) == pytest.approx(0.0, abs=1e-9)

    def test_a_right_angle_reads_as_ninety_degrees(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        angle = angle_off_axis(basis, Vec3(10.0, 0.0, 0.0))
        assert math.degrees(float(angle)) == pytest.approx(90.0)

    def test_directly_behind_reads_as_a_hundred_and_eighty(self) -> None:
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        angle = angle_off_axis(basis, Vec3(0.0, 0.0, -10.0))
        assert math.degrees(float(angle)) == pytest.approx(180.0)

    def test_a_point_on_the_camera_has_no_direction(self) -> None:
        basis = _camera(Vec3(1.0, 2.0, 3.0), Vec3(0.0, 0.0, 10.0))
        assert float(angle_off_axis(basis, Vec3(1.0, 2.0, 3.0))) == 0.0

    def test_it_ignores_the_lens(self) -> None:
        # The point of this measure: a badly aimed camera stays badly aimed no
        # matter how wide the lens that rescues it into frame.
        basis = _camera(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 10.0))
        target = Vec3(5.0, 0.0, 10.0)
        narrow = project_point(basis, target, FovDegrees(20.0))
        wide = project_point(basis, target, FovDegrees(120.0))
        assert not narrow.in_frame
        assert wide.in_frame
        assert float(angle_off_axis(basis, target)) > 0.0
