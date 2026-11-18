"""Shared geospatial utilities: local planar projection and distance helpers.

Angular (lon, lat) coordinates are not metric. Instead of calling a geodesic
distance function per point/arc pair (too slow at the scale of this dataset -
some trips have 80k+ road-network nodes), we project every trip's own
(lon, lat) pairs onto a local equirectangular planar approximation centered on
that trip's own coordinates. Over the few-kilometre extent of a single trip
this introduces negligible error (<1 m) while making every downstream
distance/angle computation a cheap vectorised numpy operation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6378137.0
METERS_PER_DEG_LAT = 110574.0  # at mid-latitudes, refined per-trip via cos(lat0)


@dataclass
class LocalProjector:
    """Projects (lon, lat) <-> local planar (x, y) meters around an origin."""

    lon0: float
    lat0: float

    @classmethod
    def from_coords(cls, lons: np.ndarray, lats: np.ndarray) -> "LocalProjector":
        return cls(lon0=float(np.mean(lons)), lat0=float(np.mean(lats)))

    def to_xy(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """Vectorised projection to local meters. Returns an (N, 2) array [x, y]."""
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(self.lat0))
        x = (lons - self.lon0) * meters_per_deg_lon
        y = (lats - self.lat0) * METERS_PER_DEG_LAT
        return np.stack([x, y], axis=-1)


def haversine_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Vectorised great-circle distance in meters between paired points."""
    lon1, lat1, lon2, lat2 = map(lambda a: np.radians(np.asarray(a, dtype=np.float64)), (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_segment_projection(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised perpendicular distance + projection fraction of points p onto segments a->b.

    p, a, b: (N, 2) arrays of planar meter coordinates.
    Returns (distance_m, t) where t in [0, 1] clipped is the fraction along a->b
    of the closest point on the segment.
    """
    ab = b - a
    ap = p - a
    ab_len2 = np.sum(ab * ab, axis=-1)
    ab_len2_safe = np.where(ab_len2 < 1e-9, 1.0, ab_len2)
    t_raw = np.sum(ap * ab, axis=-1) / ab_len2_safe
    t = np.clip(t_raw, 0.0, 1.0)
    closest = a + t[..., None] * ab
    dist = np.linalg.norm(p - closest, axis=-1)
    dist = np.where(ab_len2 < 1e-9, np.linalg.norm(ap, axis=-1), dist)
    return dist, t


def bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Bearing in degrees [0, 360) of a planar vector (dx, dy), 0 = +y (north-like)."""
    ang = np.degrees(np.arctan2(dx, dy))
    return np.mod(ang, 360.0)


def angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two angles in degrees, result in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)
