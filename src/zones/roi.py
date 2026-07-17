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


def in_zone(track: Track, polygon: list[Point]) -> bool:
    x, y = anchor(track)
    return point_in_polygon(x, y, polygon)
