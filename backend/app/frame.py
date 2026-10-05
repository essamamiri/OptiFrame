"""Parametric frame generator.

The frame front is built the way eyewear CAD models a rim: a 2D rim profile
(rounded section with a V-groove for the lens bevel) swept along the measured
lens outline. Bridge, end pieces, nose pads and a three-knuckle hinge are swept
or primitive solids, and everything is boolean-unioned into one watertight
solid with manifold3d. Temples are separate parts that print flat.

Accuracy contract: the groove bottom (the "lens seat") is exactly the measured
lens outline grown by ``clearance_mm``; styling never moves it. Coordinates are
millimetres, y up, the frame front faces +z, temples run toward -z, and the
wearer's right lens sits at -x (as seen when looking at the wearer).

This is prototype geometry, not validated eyewear CAD.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from math import pi

import manifold3d as mf
import numpy as np
import pyclipper


RIM_WIDTH_MM = 4.0
FRAME_THICKNESS_MM = 3.0
POINT_COUNT = 256
GROOVE_DEPTH_MM = 0.6
BROW_EXTRA_MM = 1.2
PIN_DIAMETER_MM = 1.75  # a piece of 1.75 mm filament works as the hinge pin
HINGE_HEIGHT_MM = 7.0
KNUCKLE_RADIUS_MM = 2.3
TEMPLE_KNUCKLE_HEIGHT_MM = 2.2
HINGE_GAP_MM = 0.25
TEMPLE_LENGTH_MM = 140.0


# --------------------------------------------------------------------------
# Lens outline preparation


def _area(points: np.ndarray) -> float:
    return float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2


def _convex_hull(points: np.ndarray) -> np.ndarray:
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) <= 3:
        return np.asarray(ordered, dtype=np.float64)

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

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _resample(points, count: int = POINT_COUNT) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
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
    edge = np.clip(np.searchsorted(cumulative, targets, side="right") - 1, 0, len(lengths) - 1)
    t = (targets - cumulative[edge]) / np.maximum(lengths[edge], 1e-9)
    return closed[edge] * (1 - t[:, None]) + closed[edge + 1] * t[:, None]


def _normalize(points: np.ndarray) -> np.ndarray:
    return points - (points.min(axis=0) + points.max(axis=0)) / 2


def _landscape(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    if len(centered) >= 3:
        values, vectors = np.linalg.eigh(np.cov(centered.T))
        major = vectors[:, int(np.argmax(values))]
        angle = np.arctan2(float(major[1]), float(major[0]))
        # An eigenvector's sign is arbitrary; level the lens with the smallest
        # rotation so the photo's top stays on top (no silent 180° flips).
        angle = (angle + pi / 2) % pi - pi / 2
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points = centered @ rotation.T
    else:
        points = centered
    span = points.max(axis=0) - points.min(axis=0)
    if span[1] > span[0]:
        points = np.column_stack([points[:, 1], -points[:, 0]])
    return _normalize(points)


def _offset(points: np.ndarray, distance: float, count: int) -> np.ndarray:
    """True (normal-direction) polygon offset with round joins."""
    if abs(distance) < 1e-9:
        return points
    scale = 10_000
    offsetter = pyclipper.PyclipperOffset(miter_limit=2, arc_tolerance=0.005 * scale)
    offsetter.AddPath(np.rint(points * scale).astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solutions = offsetter.Execute(distance * scale)
    if not solutions:
        raise ValueError("Lens contour collapsed during offset.")
    solution = np.asarray(max(solutions, key=lambda s: abs(pyclipper.Area(s))), dtype=np.float64) / scale
    return _resample(solution, count)


def lens_seat(points, clearance_mm: float, count: int = POINT_COUNT) -> np.ndarray:
    """Measured outline (sheet coords, y down) -> lens seat (y up, CCW, centred).

    Only pixel-scale noise is removed; the boxing width/height of the outline
    is restored exactly before the clearance offset is applied.
    """
    sampled = _landscape(_resample(points, count))
    original_span = sampled.max(axis=0) - sampled.min(axis=0)
    if np.any(original_span <= 0):
        raise ValueError("Contour dimensions must be positive.")
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
    smoothed *= original_span / (smoothed.max(axis=0) - smoothed.min(axis=0))
    # Sheet coordinates are y-down; the model is y-up.
    smoothed[:, 1] *= -1
    if _area(smoothed) < 0:
        smoothed = smoothed[::-1]
    seat = _offset(smoothed, clearance_mm, count)
    if _area(seat) < 0:
        seat = seat[::-1]
    return seat


def _outward_normals(points: np.ndarray) -> np.ndarray:
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)


# --------------------------------------------------------------------------
# Mesh helpers


def _solid(vertices: np.ndarray, faces: np.ndarray) -> mf.Manifold:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    if float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) < 0:
        faces = faces[:, ::-1]
    solid = mf.Manifold(mf.Mesh(vert_properties=vertices.astype(np.float32), tri_verts=faces.astype(np.uint32)))
    if solid.status() != mf.Error.NoError or solid.is_empty():
        raise ValueError(f"Frame geometry is invalid ({solid.status()}).")
    return solid


def _grid_faces(rows: int, cols: int, wrap_rows: bool) -> list[tuple[int, int, int]]:
    faces = []
    row_limit = rows if wrap_rows else rows - 1
    for i in range(row_limit):
        i2 = (i + 1) % rows
        for j in range(cols):
            j2 = (j + 1) % cols
            a, b, c, d = i * cols + j, i2 * cols + j, i2 * cols + j2, i * cols + j2
            faces += [(a, b, c), (a, c, d)]
    return faces


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, k: int) -> list[tuple[float, float]]:
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(a0, a1, k)]


def _rounded_rect(width: float, height: float, radius: float, k: int = 6) -> np.ndarray:
    w, h = width / 2, height / 2
    r = max(0.05, min(radius, w * 0.95, h * 0.95))
    points = (
        _arc(w - r, -h + r, r, -pi / 2, 0, k)
        + _arc(w - r, h - r, r, 0, pi / 2, k)
        + _arc(-w + r, h - r, r, pi / 2, pi, k)
        + _arc(-w + r, -h + r, r, pi, 3 * pi / 2, k)
    )
    return np.asarray(points)


def _rim_profile(outer: float, depth: float, groove: float, k: int = 6) -> np.ndarray:
    """Rim cross-section in (n, z): n=0 is the lens seat (groove bottom),
    n<0 points into the lens opening, z=0 is the front face."""
    n_in = -groove
    r_front = min(1.3, depth * 0.42, outer * 0.4)
    r_back = min(0.8, depth * 0.3, outer * 0.3)
    r_in = min(0.35, depth * 0.12)
    z_mid = -depth * 0.45
    half = max(0.3, min(0.8, (depth - 2 * r_in) / 2 - 0.25))
    points = (
        _arc(n_in + r_in, -depth + r_in, r_in, pi, 1.5 * pi, k)
        + _arc(outer - r_back, -depth + r_back, r_back, 1.5 * pi, 2 * pi, k)
        + _arc(outer - r_front, -r_front, r_front, 0, 0.5 * pi, k + 4)
        + _arc(n_in + r_in, -r_in, r_in, 0.5 * pi, pi, k)
    )
    if groove > 0:
        points += [(n_in, z_mid + half), (0.0, z_mid), (n_in, z_mid - half)]
    return np.asarray(points)


def _rim(seat: np.ndarray, width: float, depth: float, groove: float, brow: float) -> mf.Manifold:
    normals = _outward_normals(seat)
    height = float(np.ptp(seat[:, 1]))
    # Brow line: the upper rim is a little heavier, a classic frame cue that
    # never touches the seat.
    upness = np.clip((seat[:, 1] - seat[:, 1].min()) / max(height, 1e-6), 0, 1)
    extra = brow * np.clip((upness - 0.55) / 0.45, 0, 1) ** 1.5
    rings = []
    for point, normal, add in zip(seat, normals, extra):
        profile = _rim_profile(width + add, depth, groove)
        xy = point[None, :] + profile[:, :1] * normal[None, :]
        rings.append(np.column_stack([xy, profile[:, 1]]))
    vertices = np.vstack(rings)
    return _solid(vertices, np.asarray(_grid_faces(len(seat), len(rings[0]), True)))


def _frames_along(path: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    a1 = reference[None, :] - (tangent @ reference)[:, None] * tangent
    a1 /= np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-9)
    a2 = np.cross(tangent, a1)
    return a1, a2


def _sweep(path: np.ndarray, profiles: np.ndarray, reference) -> mf.Manifold:
    """Sweep per-station 2D profiles (u along `reference`, v across) along a path."""
    path = np.asarray(path, dtype=np.float64)
    a1, a2 = _frames_along(path, np.asarray(reference, dtype=np.float64))
    rings = [p + prof[:, :1] * u + prof[:, 1:] * v for p, prof, u, v in zip(path, profiles, a1, a2)]
    rows, cols = len(rings), len(rings[0])
    vertices = np.vstack(rings + [path[:1], path[-1:]])
    faces = _grid_faces(rows, cols, False)
    start, end = rows * cols, rows * cols + 1
    last = (rows - 1) * cols
    for j in range(cols):
        j2 = (j + 1) % cols
        faces += [(start, j, j2), (end, last + j2, last + j)]
    return _solid(vertices, np.asarray(faces))


def _catmull_rom(points, samples: int = 8) -> np.ndarray:
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    if len(pts) < 3:
        return np.asarray(pts)
    expanded = [pts[0], *pts, pts[-1]]
    curve = []
    for index in range(1, len(expanded) - 2):
        p0, p1, p2, p3 = expanded[index - 1:index + 3]
        for t in np.linspace(0, 1, samples, endpoint=False):
            curve.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    curve.append(pts[-1])
    return np.asarray(curve)


def _y_cylinder(center, radius: float, height: float, segments: int = 48) -> mf.Manifold:
    return mf.Manifold.cylinder(height, radius, radius, segments, True).rotate((-90, 0, 0)).translate(tuple(float(v) for v in center))


def _seat_x_at(seat: np.ndarray, y: float, side: str) -> float:
    band = np.abs(seat[:, 1] - y) <= 1.0
    nearby = seat[band] if band.sum() >= 2 else seat
    return float(nearby[:, 0].max() if side == "max" else nearby[:, 0].min())


# --------------------------------------------------------------------------
# Frame assembly


@dataclass
class FrameModel:
    front: mf.Manifold
    temples: list[mf.Manifold]
    right_seat: np.ndarray
    left_seat: np.ndarray
    hinge_axes: list[np.ndarray]


def build_frame(
    left_contour_mm,
    right_contour_mm,
    bridge_width_mm: float = 18.0,
    rim_width_mm: float = RIM_WIDTH_MM,
    thickness_mm: float = FRAME_THICKNESS_MM,
    clearance_mm: float = 0.3,
) -> FrameModel:
    depth = thickness_mm
    groove = GROOVE_DEPTH_MM
    right = lens_seat(right_contour_mm, clearance_mm)
    left = lens_seat(left_contour_mm, clearance_mm)
    # Bridge width (DBL) is the gap between the two lens seats.
    right = right - [float(right[:, 0].max()) + bridge_width_mm / 2, 0.0]
    left = left - [float(left[:, 0].min()) - bridge_width_mm / 2, 0.0]

    parts = [
        _rim(right, rim_width_mm, depth, groove, BROW_EXTRA_MM),
        _rim(left, rim_width_mm, depth, groove, BROW_EXTRA_MM),
    ]
    cuts: list[mf.Manifold] = []
    height = (float(np.ptp(right[:, 1])) + float(np.ptp(left[:, 1]))) / 2
    z_mid = -depth / 2
    embed = rim_width_mm * 0.5

    # Bridge: a gently arched swept bar, ends buried in the rims.
    y_bridge = height * 0.2
    x_start = _seat_x_at(right, y_bridge, "max") + embed
    x_end = _seat_x_at(left, y_bridge, "min") - embed
    x_center = (x_start + x_end) / 2
    controls = [
        (x_start, y_bridge, z_mid),
        (x_start + 2.0, y_bridge + 0.5, z_mid),
        (x_center, y_bridge + 1.6, z_mid),
        (x_end - 2.0, y_bridge + 0.5, z_mid),
        (x_end, y_bridge, z_mid),
    ]
    path = _catmull_rom(controls, 10)
    span = np.abs(path[:, 0] - x_center) / max((x_end - x_start) / 2, 1e-6)
    profiles = np.stack([_rounded_rect(3.2 + 1.2 * s**2, depth * 0.86, 1.1) for s in span])
    parts.append(_sweep(path, profiles, (0.0, 1.0, 0.0)))

    # End pieces + hinge knuckles.
    hinge_axes = []
    y_lug = height * 0.16
    for seat, side in ((right, -1.0), (left, 1.0)):
        x_seat = _seat_x_at(seat, y_lug, "min" if side < 0 else "max")
        axis = np.array([x_seat + side * (rim_width_mm + 3.6), y_lug, -depth - 4.8])
        lug_controls = [
            (x_seat + side * embed, y_lug, z_mid),
            (x_seat + side * (rim_width_mm + 1.6), y_lug, z_mid),
            (axis[0] - side * 0.2, y_lug, -depth - 1.2),
            tuple(axis),
        ]
        lug_path = _catmull_rom(lug_controls, 8)
        stations = np.linspace(0, 1, len(lug_path))
        profiles = np.stack([_rounded_rect(HINGE_HEIGHT_MM - 0.6 * np.sin(pi * s), depth * 0.95 + 0.9 * s, 1.2) for s in stations])
        parts.append(_sweep(lug_path, profiles, (0.0, 1.0, 0.0)))
        parts.append(_y_cylinder(axis, KNUCKLE_RADIUS_MM, HINGE_HEIGHT_MM))
        cuts.append(_y_cylinder(axis, KNUCKLE_RADIUS_MM + 0.45, TEMPLE_KNUCKLE_HEIGHT_MM + 2 * HINGE_GAP_MM))
        cuts.append(_y_cylinder(axis, PIN_DIAMETER_MM / 2 + 0.1, HINGE_HEIGHT_MM + 1.0, 24))
        hinge_axes.append(axis)

        # Integrated nose pad on the rim's back face, kept outside the opening.
        y_pad = -height * 0.12
        x_nasal = _seat_x_at(seat, y_pad, "max" if side < 0 else "min")
        pad = (
            mf.Manifold.sphere(1.0, 48)
            .scale((1.7, 4.6, 1.6))
            .rotate((0, -side * 28.0, 0))
            .translate((x_nasal - side * rim_width_mm * 0.6, y_pad, -depth - 0.6))
        )
        parts.append(pad)

    front = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    if cuts:
        front = front - mf.Manifold.batch_boolean(cuts, mf.OpType.Add)

    temples = [_temple(axis, side) for axis, side in zip(hinge_axes, (-1.0, 1.0))]
    return FrameModel(front=front, temples=temples, right_seat=right, left_seat=left, hinge_axes=hinge_axes)


def _temple(axis: np.ndarray, side: float) -> mf.Manifold:
    """Temple arm in the assembled (open) position, swept in a vertical plane
    so it prints flat on its side."""
    relative = [(0, 0), (0, -6), (-0.4, -40), (-1.2, -92), (-3.2, -110), (-10.5, -126), (-21.0, -TEMPLE_LENGTH_MM + 4)]
    controls = [(axis[0], axis[1] + dy, axis[2] + dz) for dy, dz in relative]
    path = _catmull_rom(controls, 12)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    profiles = []
    for s in arc:
        ramp = np.clip((s - 3.4) / 6.0, 0, 1)
        h = TEMPLE_KNUCKLE_HEIGHT_MM + (6.0 - TEMPLE_KNUCKLE_HEIGHT_MM) * (ramp * ramp * (3 - 2 * ramp))
        h -= 2.0 * np.clip((s - 20) / 80, 0, 1)
        h += 0.4 * np.clip((s - 115) / 25, 0, 1)
        width = 2.8 if s > 3.4 else 2.6
        profiles.append(_rounded_rect(width, h, 0.9))
    arm = _sweep(path, np.stack(profiles), (1.0, 0.0, 0.0))
    tangent = path[-1] - path[-2]
    tangent /= np.linalg.norm(tangent)
    tip_center = path[-1] + tangent * 0.6
    tip = mf.Manifold.sphere(1.0, 40).scale((1.6, 2.4, 2.4)).translate(tuple(tip_center))
    knuckle = _y_cylinder(axis, KNUCKLE_RADIUS_MM, TEMPLE_KNUCKLE_HEIGHT_MM)
    pin = _y_cylinder(axis, PIN_DIAMETER_MM / 2 + 0.1, TEMPLE_KNUCKLE_HEIGHT_MM + 1.0, 24)
    return (arm + tip + knuckle) - pin


# --------------------------------------------------------------------------
# Layouts and file formats


def _mesh_arrays(solid: mf.Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = solid.to_mesh()
    return np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3], np.asarray(mesh.tri_verts, dtype=np.int64)


def print_plate(model: FrameModel) -> list[mf.Manifold]:
    """Front face-down plus both temples lying on their side, all on z=0."""
    front = model.front.rotate((0, 180, 0))
    fmin = np.asarray(front.bounding_box()[:3])
    front = front.translate((0.0, 0.0, -fmin[2]))
    pieces = [front]
    fb = np.asarray(front.bounding_box())
    y_cursor = fb[1] - 6.0
    for temple in model.temples:
        laid = temple.rotate((0, 90, 0))
        bb = np.asarray(laid.bounding_box())
        laid = laid.translate((-(bb[0] + bb[3]) / 2, y_cursor - bb[4], -bb[2]))
        y_cursor -= (bb[4] - bb[1]) + 4.0
        pieces.append(laid)
    return pieces


def _binary_stl(solids: list[mf.Manifold], name: str) -> bytes:
    triangles = []
    for solid in solids:
        vertices, faces = _mesh_arrays(solid)
        triangles.append(vertices[faces])
    tris = np.concatenate(triangles).astype(np.float32)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    record = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["n"] = normals
    record["v"] = tris
    header = f"OptiFrame {name} prototype - units mm".encode()[:80].ljust(80, b" ")
    return header + struct.pack("<I", len(tris)) + record.tobytes()


def generate_frame_stl(
    left_contour_mm,
    right_contour_mm,
    bridge_width_mm: float = 18.0,
    rim_width_mm: float = RIM_WIDTH_MM,
    thickness_mm: float = FRAME_THICKNESS_MM,
    clearance_mm: float = 0.3,
    layout: str = "assembled",
) -> bytes:
    model = build_frame(left_contour_mm, right_contour_mm, bridge_width_mm, rim_width_mm, thickness_mm, clearance_mm)
    if layout == "print":
        return _binary_stl(print_plate(model), "print plate")
    return _binary_stl([model.front, *model.temples], "assembled frame")


def _glb(parts: list[tuple[str, mf.Manifold, tuple[float, float, float, float], float, float]]) -> bytes:
    """Minimal glTF 2.0 binary: one mesh per part with smooth normals split at
    sharp edges. Units are converted to metres for viewers."""
    blob = bytearray()
    views, accessors, meshes, materials, nodes = [], [], [], [], []

    def add_view(data: bytes, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    for name, solid, color, metallic, roughness in parts:
        with_normals = solid.calculate_normals(0, 40.0)
        mesh = with_normals.to_mesh()
        props = np.asarray(mesh.vert_properties, dtype=np.float32)
        positions = np.ascontiguousarray(props[:, :3] / 1000.0)
        normals = props[:, 3:6]
        normals = np.ascontiguousarray(normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9))
        indices = np.ascontiguousarray(np.asarray(mesh.tri_verts, dtype=np.uint32).reshape(-1))
        pos_view = add_view(positions.tobytes(), 34962)
        nor_view = add_view(normals.astype(np.float32).tobytes(), 34962)
        idx_view = add_view(indices.tobytes(), 34963)
        accessors += [
            {"bufferView": pos_view, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": nor_view, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": idx_view, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ]
        base = len(accessors) - 3
        material = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": list(color), "metallicFactor": metallic, "roughnessFactor": roughness}}
        if color[3] < 1:
            material["alphaMode"] = "BLEND"
            material["doubleSided"] = True
        materials.append(material)
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": base, "NORMAL": base + 1}, "indices": base + 2, "material": len(materials) - 1}]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})

    while len(blob) % 4:
        blob.append(0)
    document = {
        "asset": {"version": "2.0", "generator": "OptiFrame prototype"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": meshes, "materials": materials,
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(blob)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
        + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    )


def generate_frame_glb(
    left_contour_mm,
    right_contour_mm,
    bridge_width_mm: float = 18.0,
    rim_width_mm: float = RIM_WIDTH_MM,
    thickness_mm: float = FRAME_THICKNESS_MM,
    clearance_mm: float = 0.3,
) -> bytes:
    model = build_frame(left_contour_mm, right_contour_mm, bridge_width_mm, rim_width_mm, thickness_mm, clearance_mm)
    acetate = (0.09, 0.075, 0.07, 1.0)
    glass = (0.82, 0.9, 0.95, 0.22)
    parts = [
        ("front", model.front, acetate, 0.0, 0.35),
        ("temple_right", model.temples[0], acetate, 0.0, 0.35),
        ("temple_left", model.temples[1], acetate, 0.0, 0.35),
    ]
    # Display-only lenses: the measured outline (seat minus clearance) sitting
    # in the groove. They are never part of the STL.
    for name, seat in (("lens_right", model.right_seat), ("lens_left", model.left_seat)):
        parts.append((name, _display_lens(seat, clearance_mm, thickness_mm), glass, 0.0, 0.05))
    return _glb(parts)


def _display_lens(seat: np.ndarray, clearance_mm: float, depth: float) -> mf.Manifold:
    outline = _offset(seat, -clearance_mm, len(seat)) if clearance_mm > 0 else seat
    lens_thickness = min(2.0, depth * 0.6)
    section = mf.CrossSection([outline.tolist()], mf.FillRule.Positive)
    return section.extrude(lens_thickness).translate((0.0, 0.0, -depth * 0.45 - lens_thickness / 2))
