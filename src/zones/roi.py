"""Kiosk ROI gating. A track is "in the zone" while its bottom-center anchor
(where the person meets the floor) lies inside the ROI polygon.
"""

from __future__ import annotations

from src.track.base import Track

Point = tuple[float, float]


def anchor(track: Track) -> Point:
    return (track.x1 + track.x2) / 2, track.y2


def point_in_polygon(x: float, y: float, polygon: list[Point]) -> bool:
    # Ray casting: a point is inside when a ray to the right crosses an odd number
    # of polygon edges.
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def in_zone(track: Track, polygon: list[Point], box_depth_frac: float) -> bool:
    # Require the lower box_depth_frac of the box's central axis to lie inside the
    # ROI, not just the feet point: check both the bottom-center and the point that
    # far up from it. A feet-only test let walkers whose feet clip the polygon edge
    # (body outside) and phantom boxes grazing the boundary count as in-zone.
    cx = (track.x1 + track.x2) / 2
    depth = box_depth_frac * (track.y2 - track.y1)
    return point_in_polygon(cx, track.y2, polygon) and point_in_polygon(
        cx, track.y2 - depth, polygon
    )
