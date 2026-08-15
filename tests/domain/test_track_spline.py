"""Tests for projecting a world point onto a track.

Projection onto the nearest *segment* rather than the nearest vertex is the part
worth pinning. A spline is sampled every few metres, so snapping to a vertex
quantises lateral offset into steps of the same size as the offsets being
measured — a camera 3 m off the road and one 8 m off would read the same, which
makes the road-edge check useless in exactly the situations it exists for.
"""

from __future__ import annotations

import numpy as np
import pytest

from assetto_ai_director.domain.track import SplineSource, TrackSpline


def _straight(length: float = 100.0, count: int = 11) -> TrackSpline:
    """A level straight along ``+z``, five metres of road either side.

    Args:
        length: How long.
        count: How many vertices.

    Returns:
        The spline.
    """
    points = np.stack([np.zeros(count), np.zeros(count), np.linspace(0.0, length, count)], axis=1)
    return TrackSpline(
        points=points,
        source=SplineSource.AI_LINE,
        left_width=np.full(count, 5.0),
        right_width=np.full(count, 5.0),
    )


class TestValidation:
    def test_a_line_needs_three_components(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 3\)"):
            TrackSpline(points=np.zeros((5, 2)), source=SplineSource.DRIVEN)

    def test_a_line_needs_more_than_one_point(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            TrackSpline(points=np.zeros((1, 3)), source=SplineSource.DRIVEN)

    def test_widths_must_correspond_to_points(self) -> None:
        with pytest.raises(ValueError, match="correspond"):
            TrackSpline(
                points=np.zeros((5, 3)),
                source=SplineSource.AI_LINE,
                left_width=np.zeros(4),
            )


class TestProjection:
    def test_a_point_on_the_line_has_no_lateral_offset(self) -> None:
        point = _straight().project(np.array([0.0, 0.0, 50.0]))
        assert float(point.lateral) == pytest.approx(0.0, abs=1e-9)
        assert float(point.position) == pytest.approx(0.5)

    def test_lateral_offset_is_continuous_between_samples(self) -> None:
        # Snapping to the nearest vertex would step; this must not.
        track = _straight(count=11)
        offsets = [
            abs(float(track.project(np.array([3.0, 0.0, z])).lateral))
            for z in np.linspace(10.0, 90.0, 17)
        ]
        assert all(value == pytest.approx(3.0) for value in offsets)

    def test_the_two_sides_get_opposite_signs(self) -> None:
        track = _straight()
        left = float(track.project(np.array([4.0, 0.0, 50.0])).lateral)
        right = float(track.project(np.array([-4.0, 0.0, 50.0])).lateral)
        assert left * right < 0.0
        assert abs(left) == pytest.approx(abs(right))

    def test_elevation_follows_the_line(self) -> None:
        count = 5
        points = np.stack(
            [np.zeros(count), np.linspace(0.0, 40.0, count), np.linspace(0.0, 100.0, count)],
            axis=1,
        )
        track = TrackSpline(points=points, source=SplineSource.AI_LINE)
        halfway = track.project(np.array([0.0, 999.0, 50.0]))
        assert float(halfway.elevation) == pytest.approx(20.0)

    def test_elevation_does_not_move_the_projection_along_the_track(self) -> None:
        # A camera nine metres up a bank is beside the same piece of road as one
        # at road level, not further along it.
        track = _straight()
        low = track.project(np.array([6.0, 0.0, 50.0]))
        high = track.project(np.array([6.0, 9.0, 50.0]))
        assert float(low.position) == pytest.approx(float(high.position))

    def test_a_point_past_the_end_clamps_to_it(self) -> None:
        point = _straight().project(np.array([0.0, 0.0, 400.0]))
        assert float(point.position) == pytest.approx(1.0)

    def test_distance_and_position_agree(self) -> None:
        track = _straight(length=200.0)
        point = track.project(np.array([0.0, 0.0, 50.0]))
        assert float(point.distance) == pytest.approx(float(point.position) * float(track.length))


class TestRoadWidth:
    def test_a_point_between_the_edges_is_on_the_road(self) -> None:
        assert _straight().project(np.array([3.0, 0.0, 50.0])).is_on_road is True

    def test_a_point_past_the_edge_is_not(self) -> None:
        assert _straight().project(np.array([9.0, 0.0, 50.0])).is_on_road is False

    def test_an_unknown_width_is_none_rather_than_false(self) -> None:
        # None means "not measured". Returning False would report every camera
        # on a track without an AI line as being off the road.
        track = TrackSpline(
            points=np.stack([np.zeros(5), np.zeros(5), np.linspace(0, 100, 5)], axis=1),
            source=SplineSource.DRIVEN,
        )
        point = track.project(np.array([3.0, 0.0, 50.0]))
        assert point.half_width is None
        assert point.is_on_road is None

    def test_the_two_edges_are_read_independently(self) -> None:
        count = 5
        track = TrackSpline(
            points=np.stack([np.zeros(count), np.zeros(count), np.linspace(0, 100, count)], axis=1),
            source=SplineSource.AI_LINE,
            left_width=np.full(count, 2.0),
            right_width=np.full(count, 12.0),
        )
        narrow = track.project(np.array([5.0, 0.0, 50.0]))
        wide = track.project(np.array([-5.0, 0.0, 50.0]))
        assert {narrow.is_on_road, wide.is_on_road} == {True, False}


class TestSource:
    def test_only_the_ai_line_counts_as_measured(self) -> None:
        assert SplineSource.AI_LINE.is_measured
        assert not SplineSource.DRIVEN.is_measured

    def test_a_driven_line_never_claims_to_know_width(self) -> None:
        track = TrackSpline(
            points=np.stack([np.zeros(5), np.zeros(5), np.linspace(0, 100, 5)], axis=1),
            source=SplineSource.DRIVEN,
        )
        assert not track.knows_width
