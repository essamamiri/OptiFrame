from __future__ import annotations

from math import pi, sqrt

import numpy as np


RIM_WIDTH_MM = 4.0
FRAME_THICKNESS_MM = 3.0
POINT_COUNT = 384


def _area(points: np.ndarray) -> float:
    return float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2


def _convex_hull(points: np.ndarray) -> np.ndarray:
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) <= 3:
        return np.asarray(ordered, dtype=np.float32)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)


def _resample(points: list[tuple[float, float]], count: int = POINT_COUNT) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32)
    if len(polygon) < 3:
        raise ValueError("Contour needs at least three points.")
    # Use a hull only for obviously noisy/highly concave incoming contours.
    # Normal measurements already arrive as a smooth outside lens envelope.
    hull = _convex_hull(polygon)
    if len(hull) >= 3:
        area = abs(_area(polygon))
        hull_area = abs(_area(hull))
        if hull_area > 0 and area / hull_area < 0.88:
            polygon = hull
    if _area(polygon) < 0:
        polygon = polygon[::-1]
    closed = np.vstack([polygon, polygon[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 0:
        raise ValueError("Contour perimeter must be positive.")
    targets = np.linspace(0, perimeter, count, endpoint=False)
    cumulative = np.concatenate([[0], np.cumsum(lengths)])
    sampled = []
    edge = 0
    for target in targets:
        while edge < len(lengths) - 1 and cumulative[edge + 1] < target:
            edge += 1
        span = max(lengths[edge], 1e-6)
        t = (target - cumulative[edge]) / span
        sampled.append(closed[edge] * (1 - t) + closed[edge + 1] * t)
    return np.asarray(sampled, dtype=np.float32)


def _normalize(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return points - (minimum + maximum) / 2


def _radial_offset(points: np.ndarray, offset: float) -> np.ndarray:
    center = points.mean(axis=0)
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1)
    safe = np.where(lengths[:, None] > 1e-6, vectors / lengths[:, None], np.array([1.0, 0.0]))
    return points + safe * offset


def _smooth_lens(points: list[tuple[float, float]], clearance_mm: float, count: int = POINT_COUNT) -> np.ndarray:
    """Preserve measured lens shape, but remove pixel-scale contour noise."""
    sampled = _normalize(_resample(points, count))
    original_span = sampled.max(axis=0) - sampled.min(axis=0)
    smoothed = sampled.copy()
    for _ in range(10):
        smoothed = (
            np.roll(smoothed, 2, axis=0)
            + np.roll(smoothed, 1, axis=0) * 2
            + smoothed * 3
            + np.roll(smoothed, -1, axis=0) * 2
            + np.roll(smoothed, -2, axis=0)
        ) / 9.0
    smoothed = _normalize(smoothed)
    smooth_span = smoothed.max(axis=0) - smoothed.min(axis=0)
    if np.any(original_span <= 0) or np.any(smooth_span <= 0):
        raise ValueError("Contour dimensions must be positive.")
    smoothed *= original_span / smooth_span
    return _radial_offset(smoothed.astype(np.float32), clearance_mm)


def _outer(points: np.ndarray, width: float = RIM_WIDTH_MM) -> np.ndarray:
    center = points.mean(axis=0)
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1)
    safe = np.where(lengths[:, None] > 1e-6, vectors / lengths[:, None], np.array([1.0, 0.0]))
    return points + safe * width


def _normal(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> tuple[float, float, float]:
    ab = np.asarray(b) - np.asarray(a)
    ac = np.asarray(c) - np.asarray(a)
    n = np.cross(ab, ac)
    length = sqrt(float(np.dot(n, n)))
    if length <= 1e-9:
        return (0.0, 0.0, 0.0)
    n /= length
    return (float(n[0]), float(n[1]), float(n[2]))


def _triangle(lines: list[str], a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> None:
    nx, ny, nz = _normal(a, b, c)
    lines.append(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}")
    lines.append("    outer loop")
    for x, y, z in (a, b, c):
        lines.append(f"      vertex {x:.4f} {y:.4f} {z:.4f}")
    lines.append("    endloop")
    lines.append("  endfacet")


def _ring(lines: list[str], inner: np.ndarray, outer: np.ndarray, z0: float, z1: float) -> None:
    n = len(inner)
    for i in range(n):
        j = (i + 1) % n
        ib0 = (float(inner[i, 0]), float(inner[i, 1]), z0)
        ib1 = (float(inner[j, 0]), float(inner[j, 1]), z0)
        it0 = (float(inner[i, 0]), float(inner[i, 1]), z1)
        it1 = (float(inner[j, 0]), float(inner[j, 1]), z1)
        ob0 = (float(outer[i, 0]), float(outer[i, 1]), z0)
        ob1 = (float(outer[j, 0]), float(outer[j, 1]), z0)
        ot0 = (float(outer[i, 0]), float(outer[i, 1]), z1)
        ot1 = (float(outer[j, 0]), float(outer[j, 1]), z1)
        _triangle(lines, ot0, ot1, it1)
        _triangle(lines, ot0, it1, it0)
        _triangle(lines, ob1, ob0, ib0)
        _triangle(lines, ob1, ib0, ib1)
        _triangle(lines, ob0, ot0, it0)
        _triangle(lines, ob0, it0, ib0)
        _triangle(lines, ob1, ib1, it1)
        _triangle(lines, ob1, it1, ot1)


def _box(lines: list[str], x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> None:
    p = {
        "000": (x0, y0, z0), "100": (x1, y0, z0), "110": (x1, y1, z0), "010": (x0, y1, z0),
        "001": (x0, y0, z1), "101": (x1, y0, z1), "111": (x1, y1, z1), "011": (x0, y1, z1),
    }
    faces = [
        ("000", "100", "110", "010"), ("001", "011", "111", "101"),
        ("000", "001", "101", "100"), ("100", "101", "111", "110"),
        ("110", "111", "011", "010"), ("010", "011", "001", "000"),
    ]
    for a, b, c, d in faces:
        _triangle(lines, p[a], p[b], p[c])
        _triangle(lines, p[a], p[c], p[d])


def _extrude_polygon(lines: list[str], polygon: np.ndarray, z0: float, z1: float) -> None:
    if _area(polygon) < 0:
        polygon = polygon[::-1]
    center = polygon.mean(axis=0)
    top_center = (float(center[0]), float(center[1]), z1)
    bottom_center = (float(center[0]), float(center[1]), z0)
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        bottom_a = (float(polygon[i, 0]), float(polygon[i, 1]), z0)
        bottom_b = (float(polygon[j, 0]), float(polygon[j, 1]), z0)
        top_a = (float(polygon[i, 0]), float(polygon[i, 1]), z1)
        top_b = (float(polygon[j, 0]), float(polygon[j, 1]), z1)
        _triangle(lines, top_center, top_a, top_b)
        _triangle(lines, bottom_center, bottom_b, bottom_a)
        _triangle(lines, bottom_a, bottom_b, top_b)
        _triangle(lines, bottom_a, top_b, top_a)


def _capsule(x0: float, x1: float, y: float, radius: float, count: int = 24) -> np.ndarray:
    left, right = sorted((x0, x1))
    if right - left < radius * 2:
        midpoint = (left + right) / 2
        left, right = midpoint - radius, midpoint + radius
    angles_right = np.linspace(-pi / 2, pi / 2, count // 2, endpoint=True)
    angles_left = np.linspace(pi / 2, 3 * pi / 2, count // 2, endpoint=True)
    right_arc = np.column_stack([right + np.cos(angles_right) * radius, y + np.sin(angles_right) * radius])
    left_arc = np.column_stack([left + np.cos(angles_left) * radius, y + np.sin(angles_left) * radius])
    return np.vstack([right_arc, left_arc]).astype(np.float32)


def generate_frame_stl(
    left_contour_mm: list[tuple[float, float]],
    right_contour_mm: list[tuple[float, float]],
    bridge_width_mm: float = 18.0,
    rim_width_mm: float = RIM_WIDTH_MM,
    thickness_mm: float = FRAME_THICKNESS_MM,
    clearance_mm: float = 0.3,
) -> str:
    left_inner = _smooth_lens(left_contour_mm, clearance_mm)
    right_inner = _smooth_lens(right_contour_mm, clearance_mm)
    left_outer = _outer(left_inner, rim_width_mm)
    right_outer = _outer(right_inner, rim_width_mm)

    left_half = float(left_inner[:, 0].max())
    right_half = float(-right_inner[:, 0].min())
    left_shift = np.array([-(left_half + bridge_width_mm / 2), 0.0], dtype=np.float32)
    right_shift = np.array([right_half + bridge_width_mm / 2, 0.0], dtype=np.float32)
    left_inner += left_shift
    left_outer += left_shift
    right_inner += right_shift
    right_outer += right_shift

    lines = ["solid optiframe"]
    _ring(lines, left_inner, left_outer, 0.0, thickness_mm)
    _ring(lines, right_inner, right_outer, 0.0, thickness_mm)

    bridge_radius = max(2.8, rim_width_mm * .85)
    tab_radius = max(2.8, rim_width_mm * .80)
    _extrude_polygon(lines, _capsule(float(left_outer[:, 0].max()) - 1.0, float(right_outer[:, 0].min()) + 1.0, 0.0, bridge_radius), 0.0, thickness_mm)
    _extrude_polygon(lines, _capsule(float(left_outer[:, 0].min()) - 10.0, float(left_outer[:, 0].min()) + 1.0, 0.0, tab_radius), 0.0, thickness_mm)
    _extrude_polygon(lines, _capsule(float(right_outer[:, 0].max()) - 1.0, float(right_outer[:, 0].max()) + 10.0, 0.0, tab_radius), 0.0, thickness_mm)
    lines.append("endsolid optiframe")
    return "\n".join(lines) + "\n"
