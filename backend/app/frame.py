from __future__ import annotations

from math import pi, sqrt

import numpy as np


RIM_WIDTH_MM = 4.0
FRAME_THICKNESS_MM = 3.0
POINT_COUNT = 224


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


def _landscape(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    if len(centered) >= 3:
        covariance = np.cov(centered.T)
        values, vectors = np.linalg.eigh(covariance)
        major = vectors[:, int(np.argmax(values))]
        angle = np.arctan2(float(major[1]), float(major[0]))
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        points = centered @ rotation.T
    else:
        points = centered
    span = points.max(axis=0) - points.min(axis=0)
    if span[1] > span[0]:
        points = np.column_stack([points[:, 1], -points[:, 0]]).astype(np.float32)
    return _normalize(points.astype(np.float32))


def _radial_offset(points: np.ndarray, offset: float) -> np.ndarray:
    center = points.mean(axis=0)
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1)
    safe = np.where(lengths[:, None] > 1e-6, vectors / lengths[:, None], np.array([1.0, 0.0]))
    return points + safe * offset


def _smooth_lens(points: list[tuple[float, float]], clearance_mm: float, count: int = POINT_COUNT) -> np.ndarray:
    """Preserve measured lens shape, but remove pixel-scale contour noise."""
    sampled = _landscape(_resample(points, count))
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
    smoothed = _landscape(smoothed)
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


def _beam(
    lines: list[str],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    width: float,
    height: float,
) -> None:
    """Rectangular beam between two 3D points with a stable y-up section."""
    a = np.asarray(start, dtype=np.float32)
    b = np.asarray(end, dtype=np.float32)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length <= 1e-6:
        return
    direction = axis / length
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    side = np.cross(direction, up)
    if float(np.linalg.norm(side)) <= 1e-6:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side = side / np.linalg.norm(side)
    up = np.cross(side, direction)
    up = up / np.linalg.norm(up)
    side *= width / 2
    up *= height / 2

    corners = {
        "a00": tuple((a - side - up).tolist()), "a10": tuple((a + side - up).tolist()),
        "a11": tuple((a + side + up).tolist()), "a01": tuple((a - side + up).tolist()),
        "b00": tuple((b - side - up).tolist()), "b10": tuple((b + side - up).tolist()),
        "b11": tuple((b + side + up).tolist()), "b01": tuple((b - side + up).tolist()),
    }
    faces = [
        ("a00", "a10", "a11", "a01"), ("b00", "b01", "b11", "b10"),
        ("a00", "b00", "b10", "a10"), ("a10", "b10", "b11", "a11"),
        ("a11", "b11", "b01", "a01"), ("a01", "b01", "b00", "a00"),
    ]
    for p0, p1, p2, p3 in faces:
        _triangle(lines, corners[p0], corners[p1], corners[p2])
        _triangle(lines, corners[p0], corners[p2], corners[p3])


def _ring_basis(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-6)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    side = np.cross(tangent, up)
    if float(np.linalg.norm(side)) <= 1e-6:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side = side / np.linalg.norm(side)
    normal = np.cross(side, tangent)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
    return side, normal


def _tube_path(lines: list[str], points: list[tuple[float, float, float]], radius: float, sides: int = 14) -> None:
    if len(points) < 2:
        return
    pts = [np.asarray(point, dtype=np.float32) for point in points]
    rings: list[list[tuple[float, float, float]]] = []
    for index, point in enumerate(pts):
        if index == 0:
            tangent = pts[1] - point
        elif index == len(pts) - 1:
            tangent = point - pts[index - 1]
        else:
            tangent = pts[index + 1] - pts[index - 1]
        side, normal = _ring_basis(tangent)
        ring = []
        for step in range(sides):
            angle = 2 * pi * step / sides
            vertex = point + side * np.cos(angle) * radius + normal * np.sin(angle) * radius
            ring.append((float(vertex[0]), float(vertex[1]), float(vertex[2])))
        rings.append(ring)

    for index in range(len(rings) - 1):
        current = rings[index]
        following = rings[index + 1]
        for step in range(sides):
            nxt = (step + 1) % sides
            _triangle(lines, current[step], following[step], following[nxt])
            _triangle(lines, current[step], following[nxt], current[nxt])

    start_center = tuple(float(value) for value in pts[0])
    end_center = tuple(float(value) for value in pts[-1])
    for step in range(sides):
        nxt = (step + 1) % sides
        _triangle(lines, start_center, rings[0][nxt], rings[0][step])
        _triangle(lines, end_center, rings[-1][step], rings[-1][nxt])


def _catmull_rom(points: list[tuple[float, float, float]], samples: int = 10) -> list[tuple[float, float, float]]:
    if len(points) < 3:
        return points
    pts = [np.asarray(point, dtype=np.float32) for point in points]
    expanded = [pts[0], *pts, pts[-1]]
    curve: list[tuple[float, float, float]] = []
    for index in range(1, len(expanded) - 2):
        p0, p1, p2, p3 = expanded[index - 1], expanded[index], expanded[index + 1], expanded[index + 2]
        for step in range(samples):
            t = step / samples
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            curve.append((float(point[0]), float(point[1]), float(point[2])))
    curve.append(points[-1])
    return curve


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


def _ellipse(cx: float, cy: float, rx: float, ry: float, count: int = 36) -> np.ndarray:
    angles = np.linspace(0, 2 * pi, count, endpoint=False)
    return np.column_stack([cx + np.cos(angles) * rx, cy + np.sin(angles) * ry]).astype(np.float32)


def _edge_x_at_y(points: np.ndarray, y: float, side: str, band: float) -> float:
    nearby = points[np.abs(points[:, 1] - y) <= band]
    if len(nearby) < 2:
        nearby = points
    return float(nearby[:, 0].max() if side == "right" else nearby[:, 0].min())


def _temple_with_ear_grip(
    lines: list[str],
    hinge_x: float,
    side: int,
    rim_width_mm: float,
    thickness_mm: float,
) -> None:
    """Add a swept temple arm and ear hook without changing lens openings."""
    hinge_depth = max(4.5, thickness_mm * 1.4)
    arm_radius = max(1.75, rim_width_mm * 0.42)
    hinge_radius = max(2.25, rim_width_mm * 0.55)

    # A single rounded hinge boss keeps the model readable in STL viewers; the
    # side tab from the rim already provides the flat printable connection.
    _tube_path(
        lines,
        [(hinge_x + side * 1.8, -4.5, -hinge_depth * 0.32), (hinge_x + side * 1.8, 4.5, -hinge_depth * 0.32)],
        hinge_radius * 0.5,
        18,
    )

    # The main arm is a continuous rounded sweep: slightly flared at the hinge,
    # straight through the temple, then softly curved around the ear.
    shoulder = (hinge_x + side * 4.2, 0.0, -hinge_depth * 0.95)
    temple_controls = [
        shoulder,
        (hinge_x + side * 13.0, -0.4, -28.0),
        (hinge_x + side * 18.0, -0.7, -72.0),
        (hinge_x + side * 17.0, 1.1, -112.0),
        (hinge_x + side * 14.5, 5.8, -136.0),
        (hinge_x + side * 10.5, 9.4, -146.0),
    ]
    temple = _catmull_rom(temple_controls, 8)
    _tube_path(lines, temple, arm_radius, 18)

    # A subtle sleeve at the hinge makes the transition intentional instead of
    # looking like a thin rod pasted onto the rim.
    _tube_path(
        lines,
        [(hinge_x - side * 0.8, 0.0, thickness_mm * 0.25), shoulder],
        max(2.0, hinge_radius * 0.72),
        18,
    )

    # Rounded ear tip, angled inward and thicker so it reads as a grip.
    tip = temple_controls[-1]
    _tube_path(
        lines,
        [
            (tip[0], tip[1] - 0.6, tip[2] + 1.0),
            (tip[0] - side * 3.5, tip[1] + 1.5, tip[2] - 2.8),
            (tip[0] - side * 8.5, tip[1] + 1.0, tip[2] - 5.0),
        ],
        arm_radius * 1.25,
        18,
    )


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
    average_height = (float(left_outer[:, 1].max() - left_outer[:, 1].min()) + float(right_outer[:, 1].max() - right_outer[:, 1].min())) / 2
    bridge_y = -average_height * 0.23
    bridge_band = max(bridge_radius * 1.6, average_height * 0.08)
    bridge_left_x = _edge_x_at_y(left_outer, bridge_y, "right", bridge_band)
    bridge_right_x = _edge_x_at_y(right_outer, bridge_y, "left", bridge_band)
    _extrude_polygon(lines, _capsule(bridge_left_x - 1.0, bridge_right_x + 1.0, bridge_y, bridge_radius), 0.0, thickness_mm)
    # Nose pads below the bridge make the generated artifact read as a real
    # glasses frame, not just two measured rings.
    pad_y = average_height * 0.22
    _extrude_polygon(lines, _ellipse(-5.5, pad_y, 2.2, 4.4), -1.2, thickness_mm * 0.75)
    _extrude_polygon(lines, _ellipse(5.5, pad_y, 2.2, 4.4), -1.2, thickness_mm * 0.75)
    left_hinge_x = float(left_outer[:, 0].min()) - 10.0
    right_hinge_x = float(right_outer[:, 0].max()) + 10.0
    _extrude_polygon(lines, _capsule(left_hinge_x, float(left_outer[:, 0].min()) + 1.0, 0.0, tab_radius), 0.0, thickness_mm)
    _extrude_polygon(lines, _capsule(float(right_outer[:, 0].max()) - 1.0, right_hinge_x, 0.0, tab_radius), 0.0, thickness_mm)
    _temple_with_ear_grip(lines, left_hinge_x, -1, rim_width_mm, thickness_mm)
    _temple_with_ear_grip(lines, right_hinge_x, 1, rim_width_mm, thickness_mm)
    lines.append("endsolid optiframe")
    return "\n".join(lines) + "\n"
