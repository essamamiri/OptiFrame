"""Frame generator: the lens seat must stay exactly the measured outline plus clearance."""
import struct

import numpy as np
import pytest

from app.frame import (
    GROOVE_DEPTH_MM,
    build_frame,
    generate_frame_glb,
    generate_frame_stl,
    lens_seat,
    print_plate,
)


def lens_outline(width=52.0, height=38.0, angle_deg=0.0, origin=(100.0, 140.0), count=360):
    """Rectangular-panto lens in sheet coordinates (mm, y down); the top
    (small y) is flatter and wider, like a real lens."""
    t = np.linspace(0, 2 * np.pi, count, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = np.sign(c) * np.abs(c) ** (2 / 3.4) * width / 2
    y = np.sign(s) * np.abs(s) ** (2 / (3.4 - 1.2 * (s > 0))) * height / 2
    points = np.column_stack([x, y])
    a = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (points @ rotation.T + origin).tolist()


def holes_at(solid, z):
    """Inner loops (lens openings) of a horizontal section through the front."""
    polygons = [np.asarray(p) for p in solid.slice(z).to_polygons()]
    def signed_area(p):
        return 0.5 * float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    holes = [p for p in polygons if signed_area(p) < 0 and abs(signed_area(p)) > 500]
    return sorted(holes, key=lambda p: p[:, 0].mean())


@pytest.mark.parametrize("width,height", [(52, 38), (48, 30), (56, 44)])
@pytest.mark.parametrize("clearance", [0.0, 0.3, 0.8])
@pytest.mark.parametrize("angle", [0, 9, -14])
def test_seat_is_measured_box_plus_clearance(width, height, clearance, angle):
    seat = lens_seat(lens_outline(width, height, angle), clearance)
    box = np.ptp(seat, axis=0)
    assert box[0] == pytest.approx(width + 2 * clearance, abs=0.03)
    assert box[1] == pytest.approx(height + 2 * clearance, abs=0.03)


@pytest.mark.parametrize("thickness,rim,bridge", [(3.0, 4.0, 18.0), (4.5, 6.0, 22.0), (2.0, 2.5, 12.0)])
def test_front_is_one_watertight_solid_with_exact_openings(thickness, rim, bridge):
    right, left = lens_outline(54, 40), lens_outline(50, 36)
    model = build_frame(left, right, bridge, rim, thickness, 0.3)
    assert len(model.front.decompose()) == 1
    assert all(len(temple.decompose()) == 1 for temple in model.temples)

    # Groove bottom (z at groove centre) is the lens seat.
    seat_holes = holes_at(model.front, -thickness * 0.45)
    assert len(seat_holes) == 2
    for hole, seat in zip(seat_holes, (model.right_seat, model.left_seat)):
        assert np.ptp(hole, axis=0) == pytest.approx(np.ptp(seat, axis=0), abs=0.05)
    assert np.ptp(seat_holes[0], axis=0) == pytest.approx([54.6, 40.6], abs=0.05)
    assert np.ptp(seat_holes[1], axis=0) == pytest.approx([50.6, 36.6], abs=0.05)

    # The visible opening is smaller than the lens, so the rim holds it.
    face_holes = holes_at(model.front, -0.3)
    for face, seat in zip(face_holes, seat_holes):
        assert np.all(np.ptp(face, axis=0) < np.ptp(seat, axis=0) - 2 * GROOVE_DEPTH_MM + 0.1)

    # Bridge width (DBL) is the gap between the two seats.
    gap = seat_holes[1][:, 0].min() - seat_holes[0][:, 0].max()
    assert gap == pytest.approx(bridge, abs=0.05)


def test_orientation_y_up_and_wearer_sides():
    right, left = lens_outline(56, 40), lens_outline(48, 34)
    model = build_frame(left, right)
    # Wearer's right lens is on -x when looking at the frame from the front.
    assert model.right_seat[:, 0].mean() < 0 < model.left_seat[:, 0].mean()
    # Sheet y-down becomes model y-up: the flatter, wider top edge stays on top.
    seat = model.right_seat
    y0, y1 = seat[:, 1].min(), seat[:, 1].max()

    def chord(y):
        band = seat[np.abs(seat[:, 1] - y) < 0.6]
        return np.ptp(band[:, 0])

    assert chord(y1 - 3) > chord(y0 + 3) + 1
    # Bridge sits above the lens centre line.
    xs = np.vstack([np.asarray(p) for p in model.front.slice(-1.5).to_polygons()])
    near_center = xs[np.abs(xs[:, 0]) < 1.0]
    assert len(near_center) and near_center[:, 1].min() > 0


def test_print_plate_sits_on_bed():
    model = build_frame(lens_outline(), lens_outline())
    for piece in print_plate(model):
        assert piece.bounding_box()[2] == pytest.approx(0.0, abs=1e-4)


def test_binary_stl_and_glb_are_well_formed():
    stl = generate_frame_stl(lens_outline(), lens_outline())
    (count,) = struct.unpack("<I", stl[80:84])
    assert len(stl) == 84 + 50 * count and count > 1000
    plate = generate_frame_stl(lens_outline(), lens_outline(), layout="print")
    assert plate[:80] != stl[:80]
    glb = generate_frame_glb(lens_outline(), lens_outline())
    magic, version, length = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(glb)


def test_rejects_degenerate_contours():
    with pytest.raises(ValueError):
        build_frame([[0, 0], [1, 1]], lens_outline())
