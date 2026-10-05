import numpy as np
import struct
from io import BytesIO
import re

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import app, MAX_IMAGE_BYTES
from app.upload_limit import MAX_BODY_BYTES
from .synthetic import scene

client = TestClient(app)


def upload(image, mode="loose", eye="left", format="PNG", mime="image/png", **save_options):
    buffer = BytesIO()
    image.save(buffer, format=format, **save_options)
    return client.post("/api/captures", data={"capture_mode": mode, "eye": eye},
                       files={"image": ("capture", buffer.getvalue(), mime)})


@pytest.mark.parametrize("mode,eye", [("loose", "left"), ("loose", "right"), ("framed", "left"), ("framed", "right")])
def test_success_preserves_selections(mode, eye):
    response = upload(scene()[0], mode, eye)
    assert response.status_code == 200
    result = response.json()
    assert (result["capture_mode"], result["eye"]) == (mode, eye)
    assert result["status"] == "measured"
    assert result["next_step"] == "capture_other_eye"
    assert len(result["calibration"]["corners_px"]) == 4
    expected_width = 40 if mode == "loose" else 37
    assert result["measurement"]["width_a_mm"] == pytest.approx(expected_width, abs=.75)
    assert result["measurement"]["approximate"] is (mode == "framed")
    assert len(response.content) < 4_000_000


def test_retry_is_structured_and_preserves_selections():
    response = upload(Image.new("RGB", (1000, 800)), "framed", "right")
    assert response.status_code == 200
    result = response.json()
    assert (result["capture_mode"], result["eye"]) == ("framed", "right")
    assert result["status"] == "retry"
    assert result["next_step"] == "retry_capture"
    assert result["calibration"]["reason"] == "card_missing"
    assert result["measurement"] is None
    assert "preview_data_url" not in result["calibration"]
    assert "retake" in result["calibration"]["message"]


def test_measurement_retry_keeps_successful_calibration():
    response = upload(scene(lens=False)[0])
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "retry"
    assert result["next_step"] == "retry_capture"
    assert result["calibration"]["status"] == "calibrated"
    assert result["measurement"]["status"] == "retry"
    assert result["measurement"]["reason"] == "lens_missing"


@pytest.mark.parametrize("format,mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("HEIF", "image/heif"), ("HEIF", "image/heic")])
def test_supported_formats(format, mime):
    response = upload(scene()[0], format=format, mime=mime)
    assert response.status_code == 200
    assert response.json()["status"] == "measured"


def test_exif_orientation_uses_display_coordinates():
    image = scene()[0]
    exif = Image.Exif()
    exif[274] = 6  # rotate clockwise for display
    response = upload(image, format="JPEG", mime="image/jpeg", exif=exif)
    assert response.status_code == 200
    result = response.json()
    assert (result["width_px"], result["height_px"]) == (1100, 1400)
    assert result["status"] == "measured"
    corners = result["calibration"]["corners_px"]
    assert all(0 <= x < 1100 and 0 <= y < 1400 for x, y in corners)


@pytest.mark.parametrize("payload,mime,code", [(b"garbage", "image/png", 400),
    (b"x", "text/plain", 415), (b"x" * (MAX_IMAGE_BYTES + 1), "image/jpeg", 413)],
    ids=["corrupt", "unsupported", "oversized"])
def test_invalid_uploads(payload, mime, code):
    response = client.post("/api/captures", data={"capture_mode": "loose", "eye": "left"},
                           files={"image": ("bad", payload, mime)})
    assert response.status_code == code
    assert isinstance(response.json()["detail"], str)


def test_spoofed_image_type():
    assert upload(Image.new("RGB", (1000, 800)), format="GIF").status_code == 415


def test_minimum_dimensions():
    assert upload(Image.new("RGB", (639, 480))).status_code == 422
    assert upload(Image.new("RGB", (480, 640))).status_code == 200


def test_pixel_limit():
    assert upload(Image.new("RGB", (6000, 4100))).status_code == 422


def test_invalid_selections():
    assert upload(scene()[0], mode="automatic").status_code == 422


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_frame_stl_endpoint_returns_downloadable_mesh():
    contour = [[0, 0], [34, 0], [34, 51], [0, 51]]
    body = {
        "left_contour_mm": contour,
        "right_contour_mm": contour,
        "bridge_width_mm": 18,
        "rim_width_mm": 4,
        "thickness_mm": 3,
        "clearance_mm": 0.3,
    }
    response = client.post("/api/frame-stl", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("model/stl")
    assert 'filename="monture.stl"' in response.headers["content-disposition"]
    data = response.content
    (count,) = struct.unpack("<I", data[80:84])
    assert count > 1000 and len(data) == 84 + 50 * count
    vertices = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"].reshape(-1, 3)
    span = vertices.max(axis=0) - vertices.min(axis=0)
    assert span[0] > span[1]

    plate = client.post("/api/frame-stl", json={**body, "layout": "print"})
    assert plate.status_code == 200
    assert 'filename="monture-print-plate.stl"' in plate.headers["content-disposition"]

    glb = client.post("/api/frame-glb", json=body)
    assert glb.status_code == 200
    assert glb.headers["content-type"].startswith("model/gltf-binary")
    assert glb.content[:4] == b"glTF"

    assert client.post("/api/frame-glb", json={**body, "rim_width_mm": 9}).status_code == 422

def test_streamed_body_limit_before_multipart_parsing():
    response = client.post("/api/captures", content=iter([b"x" * MAX_BODY_BYTES, b"x"]),
                           headers={"content-type": "multipart/form-data; boundary=invalid"})
    assert response.status_code == 413


def test_transparent_rgb_is_not_detected_as_card():
    image = scene()[0].convert("RGBA")
    image.putalpha(0)
    response = upload(image)
    assert response.status_code == 200
    assert response.json()["status"] == "retry"
