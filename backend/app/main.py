from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4
import warnings

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from .calibration import CalibrationReady, CalibrationRetry, calibrate
from .aruco_sheet import SheetReady, calibrate_sheet
from .frame import generate_frame_stl
from .measurement import LensMeasurement, MeasurementRetry, measure_lens
from .upload_limit import MAX_IMAGE_BYTES, UploadLimitMiddleware

register_heif_opener()

MAX_IMAGE_PIXELS = 24_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "HEIF", "HEIC"}
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}

CaptureMode = Literal["loose", "framed"]
Eye = Literal["left", "right"]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class CaptureReceipt(BaseModel):
    capture_id: str
    capture_mode: CaptureMode
    eye: Eye
    filename: str
    width_px: int
    height_px: int
    size_bytes: int
    status: Literal["measured", "retry"]
    next_step: Literal["capture_other_eye", "retry_capture"]
    calibration: CalibrationReady | CalibrationRetry
    measurement: LensMeasurement | MeasurementRetry | None = None


class FrameRequest(BaseModel):
    left_contour_mm: list[tuple[float, float]]
    right_contour_mm: list[tuple[float, float]]
    bridge_width_mm: float = 18.0
    rim_width_mm: float = 4.0
    thickness_mm: float = 3.0
    clearance_mm: float = 0.3


app = FastAPI(
    title="OptiFrame API",
    version="0.4.0",
    description="Reference-card calibration and ML-assisted lens measurement for OptiFrame.",
)

app.add_middleware(UploadLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/captures", response_model=CaptureReceipt)
async def create_capture(
    image: UploadFile = File(...),
    capture_mode: CaptureMode = Form(...),
    eye: Eye = Form(...),
) -> CaptureReceipt:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        await image.close()
        raise HTTPException(
            status_code=415,
            detail="Choose an original JPG, PNG, HEIC, HEIF, or WebP image.",
        )

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    await image.close()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="The image must be 4 MB or smaller.")

    # Decode and CPU-bound OpenCV work off the event loop. Images remain in memory.
    width_px, height_px, calibration, measurement = await run_in_threadpool(
        process_image, image_bytes, capture_mode
    )
    measured = measurement is not None and measurement.status == "measured"

    return CaptureReceipt(
        capture_id=str(uuid4()),
        capture_mode=capture_mode,
        eye=eye,
        filename=image.filename or "camera-capture",
        width_px=width_px,
        height_px=height_px,
        size_bytes=len(image_bytes),
        status="measured" if measured else "retry",
        next_step="capture_other_eye" if measured else "retry_capture",
        calibration=calibration,
        measurement=measurement,
    )


@app.post("/api/frame-stl")
async def create_frame_stl(request: FrameRequest) -> Response:
    if not 12 <= request.bridge_width_mm <= 28:
        raise HTTPException(status_code=422, detail="Bridge width must be between 12 and 28 mm.")
    if not 2.5 <= request.rim_width_mm <= 7:
        raise HTTPException(status_code=422, detail="Rim width must be between 2.5 and 7 mm.")
    if not 2 <= request.thickness_mm <= 5:
        raise HTTPException(status_code=422, detail="Frame thickness must be between 2 and 5 mm.")
    if not 0 <= request.clearance_mm <= 1:
        raise HTTPException(status_code=422, detail="Clearance must be between 0 and 1 mm.")
    if len(request.left_contour_mm) < 3 or len(request.right_contour_mm) < 3:
        raise HTTPException(status_code=422, detail="Measure both lenses before generating the frame.")
    try:
        stl = await run_in_threadpool(
            generate_frame_stl,
            request.left_contour_mm,
            request.right_contour_mm,
            request.bridge_width_mm,
            request.rim_width_mm,
            request.thickness_mm,
            request.clearance_mm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=stl,
        media_type="model/stl",
        headers={"Content-Disposition": 'attachment; filename="monture.stl"'},
    )


def process_image(
    image_bytes: bytes, capture_mode: CaptureMode
) -> tuple[int, int, CalibrationReady | CalibrationRetry, LensMeasurement | MeasurementRetry | None]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as opened:
                if opened.format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(status_code=415, detail="Choose an original JPG, PNG, HEIC, HEIF, or WebP image.")
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=422, detail="This image has too many pixels. Use a photo of 24 megapixels or fewer.")
                opened.verify()
            with Image.open(BytesIO(image_bytes)) as opened:
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                if min(width, height) < 480 or max(width, height) < 640:
                    raise HTTPException(status_code=422, detail="The image is too small for calibration. Use at least 640 × 480 pixels (or portrait equivalent).")
                # Composite transparency consistently; never treat invisible RGB as a card.
                if oriented.mode in ("RGBA", "LA", "P"):
                    rgba = oriented.convert("RGBA")
                    rgb = Image.new("RGB", oriented.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    rgb = oriented.convert("RGB")
                calibration = calibrate_with_supported_references(rgb)
                measurement = (
                    measure_lens(rgb, calibration, capture_mode)
                    if calibration.status == "calibrated" else None
                )
                return width, height, calibration, measurement
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=422, detail="This image has too many pixels. Use a photo of 24 megapixels or fewer.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image. Retake it or choose another original.") from exc


def calibrate_with_supported_references(image: Image.Image) -> CalibrationReady | CalibrationRetry:
    sheet = calibrate_sheet(image)
    if isinstance(sheet, SheetReady):
        return CalibrationReady(
            card_size_mm=(210.0, 297.0),
            reference_type="aruco_sheet",
            reference_label=f"ArUco capture sheet ({len(sheet.markers)} marker{'s' if len(sheet.markers) != 1 else ''})",
            corners_px=sheet.sheet_corners_px,
            rectified_card_corners_px=sheet.rectified_sheet_corners_px,
            exclusion_zones_px=sheet.rectified_marker_corners_px,
            source_to_rectified=sheet.source_to_rectified,
            source_to_card_mm=sheet.source_to_sheet_mm,
            millimetres_per_pixel=sheet.millimetres_per_pixel,
            confidence=sheet.confidence,
            preview_width_px=sheet.preview_width_px,
            preview_height_px=sheet.preview_height_px,
            preview_data_url=sheet.preview_data_url,
        )
    return calibrate(image)


static_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
