"""Reading a terrain scan produced by the CSP Lua scanner.

This is the companion to :mod:`.fast_lane`: both read a file that claims a
certain structure, and both verify that the claim matches the data. The terrain
scan is JSON rather than binary, but the principle is the same — a wrong file
must produce an error, not plausible garbage.

**Self-verification checklist:**
- schema_version must be 1 (the only version defined)
- total_cells must match len(grid)
- station_count * profile_points_per_station must match len(grid)
- all cell indices must be in bounds
- all positions must be 3-element arrays
- provenance values must be from the known set
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from assetto_ai_director.domain.errors import StrideMismatchError
from assetto_ai_director.domain.geometry import Vec3
from assetto_ai_director.domain.provenance import Provenance
from assetto_ai_director.domain.terrain import TerrainCell, TerrainGrid

__all__ = ["read_terrain_scan"]


def read_terrain_scan(path: Path) -> TerrainGrid:
    """Read a terrain scan file and verify its structure.

    Args:
        path: Path to the JSON file produced by the CSP Lua scanner.

    Returns:
        A validated TerrainGrid object.

    Raises:
        ValueError: If required fields are missing or inconsistent.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If the file cannot be read.
    """
    text = path.read_text(encoding='utf-8')
    data = json.loads(text)

    _validate_top_level(data)
    cells = _parse_cells(data['grid'], data['scan_metadata'])

    return TerrainGrid(
        track_id=data['track_id'],
        layout=data.get('layout', ''),
        station_count=data['scan_metadata']['station_count'],
        profile_points_per_station=data['scan_metadata']['profile_points_per_station'],
        station_spacing_m=data['scan_metadata']['station_spacing_m'],
        profile_spacing_m=data['scan_metadata']['profile_spacing_m'],
        cells=tuple(cells),
        csp_version=data['scan_metadata']['csp_version'],
    )


def _validate_top_level(data: dict[str, Any]) -> None:
    """Check the top-level structure.

    Args:
        data: Parsed JSON object.

    Raises:
        ValueError: If required keys are missing or have wrong types.
    """
    # Required top-level keys
    for key in ('schema_version', 'track_id', 'scan_metadata', 'grid'):
        if key not in data:
            raise ValueError(f"Missing required field: {key}")

    # Schema version
    if data['schema_version'] != 1:
        raise ValueError(
            f"Unsupported schema version: {data['schema_version']} "
            f"(this reader only supports version 1)"
        )

    # Track ID
    if not isinstance(data['track_id'], str) or not data['track_id']:
        raise ValueError("track_id must be a non-empty string")

    # Metadata
    meta = data['scan_metadata']
    for key in (
        'station_count',
        'profile_points_per_station',
        'station_spacing_m',
        'profile_spacing_m',
        'profile_range_m',
        'total_cells',
        'csp_version',
        'scanned_at',
    ):
        if key not in meta:
            raise ValueError(f"Missing metadata field: {key}")

    # Grid array
    grid = data['grid']
    if not isinstance(grid, list):
        raise ValueError("grid must be an array")

    # Consistency: total_cells == len(grid)
    if meta['total_cells'] != len(grid):
        raise StrideMismatchError(
            meta['total_cells'],
            len(grid),
            "Declared total_cells does not match actual grid length",
        )

    # Consistency: station_count * profile_points == len(grid)
    expected = meta['station_count'] * meta['profile_points_per_station']
    if expected != len(grid):
        raise StrideMismatchError(
            expected,
            len(grid),
            f"station_count ({meta['station_count']}) x "
            f"profile_points_per_station ({meta['profile_points_per_station']}) "
            f"does not match grid length",
        )


def _parse_cells(
    grid: list[dict[str, Any]], metadata: dict[str, Any]
) -> list[TerrainCell]:
    """Parse and validate all grid cells.

    Args:
        grid: Array of cell objects from the JSON.
        metadata: Scan metadata for validation.

    Returns:
        List of validated TerrainCell objects.

    Raises:
        ValueError: If any cell is malformed.
    """
    station_count = metadata['station_count']
    profile_count = metadata['profile_points_per_station']
    cells = []

    for i, raw in enumerate(grid):
        cell = _parse_one_cell(raw, i, station_count, profile_count)
        cells.append(cell)

    return cells


def _parse_one_cell(
    raw: dict[str, Any],
    index: int,
    max_station: int,
    max_profile: int,
) -> TerrainCell:
    """Parse and validate a single cell.

    Args:
        raw: Cell object from JSON.
        index: Position in the grid array (for error messages).
        max_station: Maximum valid station index.
        max_profile: Maximum valid profile index.

    Returns:
        A validated TerrainCell.

    Raises:
        ValueError: If the cell is malformed.
    """
    # Required keys
    for key in (
        'station_index',
        'profile_index',
        'position_world',
        'ground_height_m',
        'surface_normal',
        'sight_clear',
        'sight_distance_m',
        'measurement_quality',
    ):
        if key not in raw:
            raise ValueError(f"Cell {index}: missing field '{key}'")

    # Indices
    station = raw['station_index']
    profile = raw['profile_index']
    if not isinstance(station, int) or station < 0 or station >= max_station:
        raise ValueError(
            f"Cell {index}: station_index {station} out of range [0, {max_station})"
        )
    if not isinstance(profile, int) or profile < 0 or profile >= max_profile:
        raise ValueError(
            f"Cell {index}: profile_index {profile} out of range [0, {max_profile})"
        )

    # Position
    pos = raw['position_world']
    if not isinstance(pos, list) or len(pos) != 3:
        raise ValueError(f"Cell {index}: position_world must be a 3-element array")
    position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))

    # Ground height
    ground_height = raw['ground_height_m']
    if ground_height is not None:
        ground_height = float(ground_height)

    # Surface normal
    normal = raw['surface_normal']
    if normal is not None:
        if not isinstance(normal, list) or len(normal) != 3:
            raise ValueError(f"Cell {index}: surface_normal must be a 3-element array")
        normal = Vec3(float(normal[0]), float(normal[1]), float(normal[2]))

    # Sight data
    sight_clear = raw['sight_clear']
    if sight_clear is not None and not isinstance(sight_clear, bool):
        raise ValueError(f"Cell {index}: sight_clear must be boolean or null")

    sight_distance = raw['sight_distance_m']
    if sight_distance is not None:
        sight_distance = float(sight_distance)

    # Quality / provenance
    quality_str = raw['measurement_quality']
    try:
        quality = Provenance(quality_str)
    except ValueError:
        raise ValueError(
            f"Cell {index}: unknown quality '{quality_str}', "
            f"must be one of {[p.value for p in Provenance]}"
        ) from None

    return TerrainCell(
        station_index=station,
        profile_index=profile,
        position=position,
        ground_height=ground_height,
        surface_normal=normal,
        sight_clear=sight_clear,
        sight_distance=sight_distance,
        quality=quality,
    )
