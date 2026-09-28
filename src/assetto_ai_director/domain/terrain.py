"""Terrain grid from a CSP scan: ground heights, normals, and sight lines.

This is what the Lua scanner produces and what the camera placement uses to
decide whether a candidate position is underground or has no view of the car.
The grid is deliberately simple: a list of cells in row-major order (stations
outer loop, profile points inner loop), with enough metadata to verify that it
matches the track it claims to describe.

**Provenance travels with the data.** Each cell carries a quality indicator so
a consumer can refuse to act on interpolated or missing data rather than treat
it as measured. This is the same rule the rest of the project applies to track
geometry: a check that cannot measure must report UNKNOWN, not guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from assetto_ai_director.domain.geometry import Vec3
from assetto_ai_director.domain.provenance import Provenance

if TYPE_CHECKING:
    pass

__all__ = ["TerrainCell", "TerrainGrid"]


@dataclass(frozen=True, slots=True)
class TerrainCell:
    """One measurement point in the terrain grid.

    Attributes:
        station_index: Which station along the track this cell belongs to.
        profile_index: Which lateral point within the station.
        position: World position of this measurement (x, y, z).
        ground_height: Ground elevation at this point, or None if the ray missed.
        surface_normal: Surface normal from ray:physics, or None.
        sight_clear: Whether sight line to the track centreline is unobstructed,
            or None if not measured.
        sight_distance: Distance to where the sight ray hit, or None.
        quality: How this measurement was obtained.
    """

    station_index: int
    profile_index: int
    position: Vec3
    ground_height: float | None
    surface_normal: Vec3 | None
    sight_clear: bool | None
    sight_distance: float | None
    quality: Provenance


@dataclass(frozen=True, slots=True)
class TerrainGrid:
    """A complete terrain scan for one track layout.

    Attributes:
        track_id: Track folder name, e.g. "pk_akina".
        layout: Layout name, empty for single-layout tracks.
        station_count: Number of stations along the track.
        profile_points_per_station: Lateral points per station.
        station_spacing_m: Distance between stations in metres.
        profile_spacing_m: Lateral spacing between profile points in metres.
        cells: Grid cells in row-major order. Length must equal
            station_count * profile_points_per_station.
        csp_version: Custom Shaders Patch version that performed the scan.
    """

    track_id: str
    layout: str
    station_count: int
    profile_points_per_station: int
    station_spacing_m: float
    profile_spacing_m: float
    cells: tuple[TerrainCell, ...]
    csp_version: str

    def __post_init__(self) -> None:
        """Validate the grid dimensions.

        Raises:
            ValueError: If the cell count does not match the declared grid size.
        """
        expected = self.station_count * self.profile_points_per_station
        if len(self.cells) != expected:
            message = (
                f"Terrain grid has {len(self.cells)} cells but declares "
                f"{self.station_count} x {self.profile_points_per_station} = {expected}."
            )
            raise ValueError(message)

    @property
    def total_cells(self) -> int:
        """Total number of measurement points."""
        return len(self.cells)

    def _cell_index(self, station: int, profile: int) -> int:
        """Convert (station, profile) to flat index.

        Args:
            station: Station index (0-based).
            profile: Profile point index (0-based).

        Returns:
            Flat index into the cells array.

        Raises:
            IndexError: If indices are out of bounds.
        """
        if station < 0 or station >= self.station_count:
            message = f"Station {station} out of range [0, {self.station_count})"
            raise IndexError(message)
        if profile < 0 or profile >= self.profile_points_per_station:
            message = f"Profile {profile} out of range [0, {self.profile_points_per_station})"
            raise IndexError(message)
        return station * self.profile_points_per_station + profile

    def get_cell(self, station: int, profile: int) -> TerrainCell:
        """Retrieve one cell from the grid.

        Args:
            station: Station index (0-based).
            profile: Profile point index (0-based).

        Returns:
            The cell at that position.
        """
        return self.cells[self._cell_index(station, profile)]

    def sample_ground_height(self, x: float, z: float) -> float | None:
        """Find ground height at a world position by nearest-neighbour lookup.

        **Nearest-neighbour, not bilinear.** The grid is sparse (every 2 m laterally,
        every 5 m longitudinally) and the terrain between samples is not guaranteed
        to be smooth. A reader who wants sub-metre precision should interpolate
        themselves; this method answers "what is the nearest measured height" and
        leaves the interpretation to the caller.

        Args:
            x: World x coordinate in metres.
            z: World z coordinate in metres.

        Returns:
            Ground height in metres, or None if no cell has a measurement.
        """
        best_dist = float("inf")
        best_height = None

        for cell in self.cells:
            if cell.ground_height is None:
                continue
            dx = cell.position.x - x
            dz = cell.position.z - z
            dist = dx * dx + dz * dz  # squared, no sqrt needed for comparison
            if dist < best_dist:
                best_dist = dist
                best_height = cell.ground_height

        return best_height

    def sample_normal(self, x: float, z: float) -> Vec3 | None:
        """Find surface normal at a world position by nearest-neighbour lookup.

        Args:
            x: World x coordinate in metres.
            z: World z coordinate in metres.

        Returns:
            Surface normal vector, or None if no cell has a measurement.
        """
        best_dist = float("inf")
        best_normal = None

        for cell in self.cells:
            if cell.surface_normal is None:
                continue
            dx = cell.position.x - x
            dz = cell.position.z - z
            dist = dx * dx + dz * dz
            if dist < best_dist:
                best_dist = dist
                best_normal = cell.surface_normal

        return best_normal

    def is_underground(self, x: float, y: float, z: float, tolerance: float = 0.0) -> bool:
        """Test whether a point lies below the ground surface.

        Args:
            x: World x coordinate in metres.
            y: World y coordinate in metres.
            z: World z coordinate in metres.
            tolerance: Allowable depth below ground in metres. Positive means
                "allow this much below ground before calling it underground".

        Returns:
            True if the point is below ground (beyond tolerance), False if above
            or if no measurement exists.
        """
        ground = self.sample_ground_height(x, z)
        if ground is None:
            return False  # unknown, not underground
        return y < ground - tolerance

    def sight_clear(self, x: float, z: float) -> bool | None:
        """Report the sight line the scanner recorded nearest to a point.

        Uses the pre-measured sight data from the nearest grid cell. This does
        NOT cast a new ray — it reads what the Lua scanner recorded, and the
        scanner measured sight from each cell to the centreline. It says nothing
        about any other target, so it takes no target: accepting one it could
        not use would let a caller believe it had checked something it had not.

        Args:
            x: World x coordinate of the viewpoint in metres.
            z: World z coordinate of the viewpoint in metres.

        Returns:
            True if sight is clear, False if obstructed, None if not measured.
        """
        # Find nearest cell
        best_dist = float("inf")
        best_cell = None

        for cell in self.cells:
            dx = cell.position.x - x
            dz = cell.position.z - z
            dist = dx * dx + dz * dz
            if dist < best_dist:
                best_dist = dist
                best_cell = cell

        if best_cell is None:
            return None

        # The sight data was measured from this cell's position to the centreline.
        # If the caller's target differs, we report what we have — which may not
        # be exactly right, but it is the best available data.
        return best_cell.sight_clear
