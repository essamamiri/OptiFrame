"""Lens/rim contour measurement on a calibrated image plane.

Primary path is the trained ML mask (app/ml_segment.py, tiny UNet ONNX);
the deterministic Canny/Otsu baseline remains as automatic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import pyclipper
from PIL import Image
from pydantic import BaseModel

from .calibration import CalibrationReady
from . import ml_segment

FRAME_INSET_MM = 1.5


class MeasurementRetry(BaseModel):
    status: Literal["retry"] = "retry"
    reason: Literal["lens_missing", "lens_cropped", "lens_blurry", "ambiguous_contour", "low_confidence"]
    message: str


class LensMeasurement(BaseModel):
    status: Literal["measured"] = "measured"
    method: Literal["direct_lens_contour", "outer_rim_inset", "ml_lens_mask", "ml_rim_inset"]
    approximate: bool
    inset_mm: float
    width_a_mm: float
    height_b_mm: float
    perimeter_mm: float
    confidence: float
    contour_px: list[tuple[float, float]]
    contour_mm: list[tuple[float, float]]
    detected_outer_contour_px: list[tuple[float, float]] | None = None


MESSAGES = {
    "lens_missing": "We calibrated the card but could not find a complete lens edge. Put one target lens or rim beside the card on a plain contrasting surface, avoid glare, and retake the photo.",
    "lens_cropped": "The target lens or rim touches the photo edge. Move back until its complete outline and a clear margin are visible, then retake the photo.",
    "lens_blurry": "The lens or rim edge is too soft to measure reliably. Add even light, tap the edge to focus, hold still, and retake the photo.",
    "ambiguous_contour": "More than one lens-sized outline was found. Photograph one target lens or rim beside the card and remove other rounded objects.",
    "low_confidence": "We found an outline but cannot measure it reliably. Use a plain contrasting surface, separate the lens from the card, remove glare and shadows, and retake from overhead.",
}


def retry(reason: str) -> MeasurementRetry:
    return MeasurementRetry(reason=reason, message=MESSAGES[reason])


@dataclass
class Candidate:
    contour: np.ndarray
    score: float
    sharpness: float
    area: float
    center: np.ndarray
    size: np.ndarray


def _rectify(image: Image.Image, calibration: CalibrationReady) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.warpPerspective(
        rgb,
        np.asarray(calibration.source_to_rectified, dtype=np.float64),
        (calibration.preview_width_px, calibration.preview_height_px),
        flags=cv2.INTER_LINEAR,
        borderValue=(235, 239, 235),
    )


def _card_mask(shape: tuple[int, int], calibration: CalibrationReady, padding_px: int) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    exclusion_zones = getattr(calibration, "exclusion_zones_px", [])
    if exclusion_zones:
        for zone in exclusion_zones:
            polygon = np.asarray(zone, np.float32)
            center = polygon.mean(axis=0)
            distances = np.linalg.norm(polygon - center, axis=1)
            expanded = center + (polygon - center) * (1 + padding_px / max(float(distances.min()), 1))
            cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 255)
    else:
        card = np.asarray(calibration.rectified_card_corners_px, np.float32)
        center = card.mean(axis=0)
        distances = np.linalg.norm(card - center, axis=1)
        expanded = center + (card - center) * (1 + padding_px / max(float(distances.min()), 1))
        cv2.fillConvexPoly(mask, np.rint(expanded).astype(np.int32), 255)
    return mask


def _edge_sharpness(gray: np.ndarray, contour: np.ndarray) -> float:
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    points = np.rint(contour.reshape(-1, 2)).astype(np.int32)
    if len(points) > 300:
        points = points[:: max(1, len(points) // 300)]
    values = gradient[
        np.clip(points[:, 1], 0, gray.shape[0] - 1),
        np.clip(points[:, 0], 0, gray.shape[1] - 1),
    ]
    return float(np.percentile(values, 30)) if len(values) else 0.0


def _candidate_contours(rectified: np.ndarray, calibration: CalibrationReady) -> tuple[list[Candidate], bool]:
    gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    median = float(np.median(blurred))
    lower = int(max(12, .45 * median))
    upper = int(min(240, max(lower + 30, 1.35 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    close_radius = max(1, round(.45 / calibration.millimetres_per_pixel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    card_mask = _card_mask(gray.shape, calibration, round(4 / calibration.millimetres_per_pixel))
    edges[card_mask > 0] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    height, width = gray.shape
    margin = max(8, round(2 / calibration.millimetres_per_pixel))
    candidates: list[Candidate] = []
    cropped = False
    mm = calibration.millimetres_per_pixel
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
        if len(contour) < 40:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        size_mm = np.array([w * mm, h * mm])
        if not (18 <= size_mm[0] <= 90 and 12 <= size_mm[1] <= 75):
            continue
        if x <= margin or y <= margin or x + w >= width - margin or y + h >= height - margin:
            cropped = True
            continue
        area = float(abs(cv2.contourArea(contour)))
        area_mm2 = area * mm * mm
        if not 180 <= area_mm2 <= 5200:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        extent = area / float(w * h)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        if solidity < .72 or not .42 <= extent <= .92 or circularity < .30:
            continue
        # Reject anything intersecting the expanded reference-card region.
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [contour], -1, 255, 2)
        if np.any((contour_mask > 0) & (card_mask > 0)):
            continue
        sharpness = _edge_sharpness(gray, contour)
        geometry = min(solidity / .95, 1) * .45 + min(circularity / .75, 1) * .35 + min(area_mm2 / 800, 1) * .20
        score = float(geometry * min(sharpness / 100, 1))
        moments = cv2.moments(contour)
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        candidate = Candidate(contour, score, sharpness, area, center, size_mm)
        duplicate_index = next((i for i, existing in enumerate(candidates)
                                if np.linalg.norm(existing.center - center) < max(w, h) * .08
                                and np.max(np.abs(existing.size - size_mm)) < 10), None)
        if duplicate_index is None:
            candidates.append(candidate)
        elif candidate.area > candidates[duplicate_index].area:
            candidates[duplicate_index] = candidate
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True), cropped


def _metric_to_rectified(calibration: CalibrationReady) -> np.ndarray:
    return np.asarray(calibration.source_to_rectified) @ np.linalg.inv(
        np.asarray(calibration.source_to_card_mm)
    )


def _sheet_zone_candidates(rectified: np.ndarray, calibration: CalibrationReady) -> list[Candidate]:
    """Low-contrast search constrained to the printed ArUco sheet's lens zone."""
    if getattr(calibration, "reference_type", "blank_card") != "aruco_sheet":
        return []
    gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    mm_to_rectified = _metric_to_rectified(calibration)
    zone_mm = np.float32([[[12.0, 63.5], [128.0, 63.5], [128.0, 163.5], [12.0, 163.5]]])
    zone_px = cv2.perspectiveTransform(zone_mm, mm_to_rectified)[0]
    x, y, w, h = cv2.boundingRect(np.rint(zone_px).astype(np.int32))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, gray.shape[1]), min(y + h, gray.shape[0])
    if x1 - x0 < 40 or y1 - y0 < 40:
        return []

    crop = gray[y0:y1, x0:x1]
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blurred, 20, 60)
    close_radius = max(2, round(1.0 / calibration.millimetres_per_pixel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[Candidate] = []
    mm = calibration.millimetres_per_pixel
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        if len(contour) < 40:
            continue
        contour = contour + np.array([[[x0, y0]]], dtype=np.int32)
        bx, by, bw, bh = cv2.boundingRect(contour)
        size_mm = np.array([bw * mm, bh * mm])
        if not (18 <= size_mm[0] <= 90 and 12 <= size_mm[1] <= 75):
            continue
        if bx <= x0 + close_radius or by <= y0 + close_radius or bx + bw >= x1 - close_radius or by + bh >= y1 - close_radius:
            continue
        area = float(abs(cv2.contourArea(contour)))
        area_mm2 = area * mm * mm
        if not 180 <= area_mm2 <= 5200:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        extent = area / float(bw * bh)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        if solidity < .72 or not .40 <= extent <= .92 or circularity < .28:
            continue
        sharpness = _edge_sharpness(gray, contour)
        geometry = min(solidity / .95, 1) * .35 + min(circularity / .70, 1) * .25 + min(area_mm2 / 1200, 1) * .40
        score = float(geometry * (.45 + .55 * min(sharpness / 30, 1)))
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        candidates.append(Candidate(contour, score, sharpness, area, center, size_mm))
    return sorted(candidates, key=lambda candidate: (candidate.area, candidate.score), reverse=True)


def _prune_spurs(contour: np.ndarray, shape: tuple[int, int], calibration: CalibrationReady) -> np.ndarray | None:
    """Fill the outline and open it with a ~1.5 mm disc. Paper texture and shadow specks can
    join the rim as thin branches; the convex envelope later would stretch over them and make
    the lens look several mm too big. Opening cuts anything thinner than the disc."""
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
    radius = max(2, round(1.5 / calibration.millimetres_per_pixel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return None
    best = max(found, key=cv2.contourArea)
    return best if len(best) >= 40 else None


def _snap_to_dark_ridge(gray: np.ndarray, contour: np.ndarray, calibration: CalibrationReady) -> np.ndarray:
    """Move each contour point along the radial direction to the darkest pixel
    of the rim line (sub-pixel), searching inward from the outer band edge."""
    mm = calibration.millimetres_per_pixel
    points = _resample_closed(contour, 360)
    center = points.mean(axis=0)
    radial = center - points
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
    steps = np.arange(-round(0.5 / mm), round(3.5 / mm) + 1, dtype=np.float32)
    samples_x = (points[:, None, 0] + radial[:, None, 0] * steps[None]).astype(np.float32)
    samples_y = (points[:, None, 1] + radial[:, None, 1] * steps[None]).astype(np.float32)
    profile = cv2.remap(gray.astype(np.float32), samples_x, samples_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    best = np.argmin(profile, axis=1)
    # Parabolic sub-pixel refinement around the minimum.
    idx = np.clip(best, 1, len(steps) - 2)
    rows = np.arange(len(points))
    left, mid, right = profile[rows, idx - 1], profile[rows, idx], profile[rows, idx + 1]
    denom = left - 2 * mid + right
    offset = np.divide(0.5 * (left - right), denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-6)
    shift = steps[idx] + np.clip(offset, -1, 1)
    # Tinted / dark lenses have no thin rim line: the profile drops from paper to dark and
    # stays dark. Snapping to the minimum there would pull the outline millimetres inside the
    # lens, so for those points use the step edge (where it first gets halfway dark) instead.
    paper = profile[:, 0]
    darkest = profile[rows, best]
    inner = profile[:, -max(3, len(steps) // 4):].mean(axis=1)
    is_step = (inner - darkest) < 0.35 * np.maximum(paper - darkest, 1e-3)
    halfway = (paper + darkest) / 2
    below = profile < halfway[:, None]
    first = np.where(below.any(axis=1), below.argmax(axis=1), best)
    j = np.clip(first, 1, len(steps) - 1)
    hi, lo = profile[rows, j - 1], profile[rows, j]
    frac = np.divide(hi - halfway, hi - lo, out=np.zeros_like(hi), where=np.abs(hi - lo) > 1e-6)
    step_shift = steps[j - 1] + np.clip(frac, 0, 1)
    shift = np.where(is_step, step_shift, shift)
    # A few outliers (glare, print) shouldn't drag the outline: median-filter the shifts.
    padded = np.concatenate([shift[-3:], shift, shift[:3]])
    shift = np.median(np.lib.stride_tricks.sliding_window_view(padded, 7), axis=1)
    snapped = points + radial * shift[:, None]
    return snapped.astype(np.float32).reshape(-1, 1, 2)


def _sheet_dark_rim_candidates(rectified: np.ndarray, calibration: CalibrationReady) -> list[Candidate]:
    """Detect the dark lens rim on the controlled printed sheet.

    This is intentionally sheet-only: the printed markers give scale and a
    known search zone, while black-hat morphology suppresses broad glare and
    keeps the darker physical rim.
    """
    if getattr(calibration, "reference_type", "blank_card") != "aruco_sheet":
        return []
    gray = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    mm_to_rectified = _metric_to_rectified(calibration)
    zone_mm = np.float32([[[12.0, 63.5], [128.0, 63.5], [128.0, 163.5], [12.0, 163.5]]])
    zone_px = cv2.perspectiveTransform(zone_mm, mm_to_rectified)[0]
    x, y, w, h = cv2.boundingRect(np.rint(zone_px).astype(np.int32))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, gray.shape[1]), min(y + h, gray.shape[0])
    if x1 - x0 < 40 or y1 - y0 < 40:
        return []

    crop = gray[y0:y1, x0:x1]
    rim_kernel = max(3, round(2.0 / calibration.millimetres_per_pixel))
    if rim_kernel % 2 == 0:
        rim_kernel += 1
    blackhat = cv2.morphologyEx(
        crop,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_kernel, rim_kernel)),
    )
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    _, mask = cv2.threshold(blackhat, float(np.percentile(blackhat, 92)), 255, cv2.THRESH_BINARY)
    close_radius = max(3, round(2.0 / calibration.millimetres_per_pixel))
    if close_radius % 2 == 0:
        close_radius += 1
    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius, close_radius)),
    )

    # RETR_EXTERNAL: the thresholded rim is a ring, and RETR_LIST returned
    # both its outer and inner boundary as separate, equally scored candidates
    # (~2-3 mm apart). Keep only the outer one, then snap it onto the centre
    # of the dark rim line so the result doesn't depend on the band width.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates: list[Candidate] = []
    mm = calibration.millimetres_per_pixel
    smooth_gray = cv2.GaussianBlur(gray, (0, 0), max(0.6, 0.15 / mm))
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        if len(contour) < 80:
            continue
        contour = contour + np.array([[[x0, y0]]], dtype=np.int32)
        contour = _prune_spurs(contour, gray.shape, calibration)
        if contour is None:
            continue
        contour = _snap_to_dark_ridge(smooth_gray, contour, calibration)
        bx, by, bw, bh = cv2.boundingRect(contour)
        size_mm = np.array([bw * mm, bh * mm])
        if not (18 <= size_mm[0] <= 90 and 12 <= size_mm[1] <= 75):
            continue
        area = float(abs(cv2.contourArea(contour)))
        area_mm2 = area * mm * mm
        if not 450 <= area_mm2 <= 3600:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        if hull_area <= 0:
            continue
        solidity = area / hull_area
        extent = area / float(bw * bh)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        if solidity < .80 or not .55 <= extent <= .86 or circularity < .16:
            continue
        finalized = _finalize(contour, calibration, "loose", .5, ml=False)
        if not isinstance(finalized, LensMeasurement):
            continue
        sharpness = _edge_sharpness(gray, contour)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        area_prior = max(0.45, min(1.0, 1500.0 / max(area_mm2, 1.0)))
        shape_score = min(solidity / .92, 1) * .35 + min(extent / .70, 1) * .25 + min(circularity / .32, 1) * .40
        score = float(shape_score * area_prior * (.35 + .65 * min(sharpness / 20, 1)))
        candidates.append(Candidate(contour, score, sharpness, area, center, size_mm))
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    polygon = points.reshape(-1, 2).astype(np.float32)
    closed = np.vstack([polygon, polygon[0]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 0:
        return polygon
    targets = np.linspace(0, perimeter, count, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    sampled: list[np.ndarray] = []
    edge = 0
    for target in targets:
        while edge < len(lengths) - 1 and cumulative[edge + 1] < target:
            edge += 1
        span = max(float(lengths[edge]), 1e-6)
        t = (target - cumulative[edge]) / span
        sampled.append(closed[edge] * (1 - t) + closed[edge + 1] * t)
    return np.asarray(sampled, dtype=np.float32)


def _rounded_outer_contour(contour_px: np.ndarray, calibration: CalibrationReady) -> np.ndarray:
    # The raw edge detector can see glare and internal reflection lines. The
    # usable lens shape for SVG/STL is the smooth outside envelope, so use the
    # hull as a robust envelope and then smooth it without reducing it to a
    # faceted low-point polygon.
    hull = cv2.convexHull(contour_px.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3:
        return contour_px.reshape(-1, 2).astype(np.float32)

    target_count = 160
    original_min = hull.min(axis=0)
    original_max = hull.max(axis=0)
    original_center = (original_min + original_max) / 2
    original_span = original_max - original_min

    rounded = _resample_closed(hull, target_count)
    # Circular fairing: enough to remove saw teeth and polygon corners, not
    # enough to change the measured boxing dimensions.
    for _ in range(18):
        rounded = (
            np.roll(rounded, 2, axis=0)
            + np.roll(rounded, 1, axis=0) * 3
            + rounded * 4
            + np.roll(rounded, -1, axis=0) * 3
            + np.roll(rounded, -2, axis=0)
        ) / 12.0

    rounded_min = rounded.min(axis=0)
    rounded_max = rounded.max(axis=0)
    rounded_center = (rounded_min + rounded_max) / 2
    rounded_span = rounded_max - rounded_min
    if np.all(original_span > 0) and np.all(rounded_span > 0):
        rounded = (rounded - rounded_center) * (original_span / rounded_span) + original_center

    # Keep point spacing around 0.5 mm or better for clean SVG/STL curves.
    spacing_count = max(target_count, int(cv2.arcLength(hull.reshape(-1, 1, 2), True) / max(2.0, 0.5 / calibration.millimetres_per_pixel)))
    if spacing_count > target_count:
        rounded = _resample_closed(rounded, min(spacing_count, 256))
    return rounded.astype(np.float32)


def _simplify_metric(contour_px: np.ndarray, calibration: CalibrationReady) -> tuple[np.ndarray, np.ndarray]:
    simplified_px = _rounded_outer_contour(contour_px, calibration)
    rectified_to_mm = np.asarray(calibration.source_to_card_mm) @ np.linalg.inv(
        np.asarray(calibration.source_to_rectified)
    )
    contour_mm = cv2.perspectiveTransform(simplified_px[None], rectified_to_mm)[0]
    return simplified_px, contour_mm


def _inset_polygon(contour_mm: np.ndarray) -> np.ndarray | None:
    scale = 10_000
    path = np.rint(contour_mm * scale).astype(np.int64).tolist()
    offsetter = pyclipper.PyclipperOffset(miter_limit=2, arc_tolerance=.02 * scale)
    offsetter.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    solutions = offsetter.Execute(-FRAME_INSET_MM * scale)
    if not solutions:
        return None
    solution = max(solutions, key=lambda points: abs(pyclipper.Area(points)))
    if len(solution) < 3:
        return None
    return np.asarray(solution, np.float32) / scale


def _oriented_boxing_dimensions(contour_mm: np.ndarray) -> tuple[float, float]:
    """Boxing size of the lens on its own axes: A is the long (horizontal-when-worn) side,
    B the short side. Uses the minimum-area rectangle, so it doesn't depend on how the lens
    was rotated on the sheet or next to the card (a lens laid down sideways still gets A > B)."""
    rect = cv2.minAreaRect(contour_mm.astype(np.float32).reshape(-1, 1, 2))
    side_a, side_b = sorted(rect[1], reverse=True)
    if side_a <= 0 or side_b <= 0:
        return 0.0, 0.0
    return float(side_a), float(side_b)


def _finalize(
    contour_preview_px: np.ndarray,
    calibration: CalibrationReady,
    capture_mode: Literal["loose", "framed"],
    confidence: float,
    ml: bool,
) -> LensMeasurement | MeasurementRetry:
    contour_preview_px = np.asarray(contour_preview_px, dtype=np.float32).reshape(-1, 1, 2)
    # Margin / size sanity in preview pixels (mirrors deterministic filters).
    h, w = calibration.preview_height_px, calibration.preview_width_px
    mm = calibration.millimetres_per_pixel
    x, y, bw, bh = cv2.boundingRect(contour_preview_px.astype(np.int32))
    margin = max(8, round(2 / mm))
    if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
        return retry("lens_cropped")
    size_mm = np.array([bw * mm, bh * mm])
    if not (18 <= size_mm[0] <= 90 and 12 <= size_mm[1] <= 75):
        return retry("lens_missing")

    outer_px, outer_mm = _simplify_metric(contour_preview_px, calibration)
    measured_mm = outer_mm
    measured_px = outer_px
    if capture_mode == "framed":
        inset = _inset_polygon(outer_mm)
        if inset is None:
            return retry("low_confidence")
        measured_mm = inset
        mm_to_rectified = np.linalg.inv(
            np.asarray(calibration.source_to_card_mm) @ np.linalg.inv(
                np.asarray(calibration.source_to_rectified)
            )
        )
        measured_px = cv2.perspectiveTransform(inset.astype(np.float32)[None], mm_to_rectified)[0]

    width_a, height_b = _oriented_boxing_dimensions(measured_mm)
    perimeter = cv2.arcLength(measured_mm.astype(np.float32).reshape(-1, 1, 2), True)
    if width_a <= 0 or height_b <= 0 or perimeter <= 0:
        return retry("low_confidence")
    if capture_mode == "loose":
        method = "ml_lens_mask" if ml else "direct_lens_contour"
    else:
        method = "ml_rim_inset" if ml else "outer_rim_inset"
    return LensMeasurement(
        method=method,
        approximate=capture_mode == "framed",
        inset_mm=FRAME_INSET_MM if capture_mode == "framed" else 0,
        width_a_mm=round(float(width_a), 2),
        height_b_mm=round(float(height_b), 2),
        perimeter_mm=round(float(perimeter), 2),
        confidence=round(float(confidence), 3),
        contour_px=measured_px.tolist(),
        contour_mm=measured_mm.tolist(),
        detected_outer_contour_px=outer_px.tolist() if capture_mode == "framed" else None,
    )


def _ml_rescue(
    rectified: np.ndarray,
    calibration: CalibrationReady,
    capture_mode: Literal["loose", "framed"],
    candidates: list[Candidate],
) -> LensMeasurement | None:
    """Run the UNet on tight crops around deterministic candidates.

    Crops match the training distribution (one dominant lens filling the
    frame); full scenes do not, so the model never runs on full scenes.
    Returns the most confident valid ML measurement, or None.
    """
    height, width = rectified.shape[:2]
    best_ml: LensMeasurement | None = None
    for cand in candidates[:3]:
        x, y, w, h = cv2.boundingRect(cand.contour)
        pad = int(max(w, h) * 0.4)
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, width), min(y + h + pad, height)
        if x1 - x0 < 32 or y1 - y0 < 32:
            continue
        result = ml_segment.predict_contour(rectified[y0:y1, x0:x1])
        if result is None:
            continue
        contour_crop, ml_conf = result
        contour_preview = contour_crop.reshape(-1, 1, 2).astype(np.float32)
        contour_preview[:, :, 0] += x0
        contour_preview[:, :, 1] += y0
        finalized = _finalize(contour_preview, calibration, capture_mode, ml_conf, ml=True)
        if not isinstance(finalized, LensMeasurement):
            continue
        # The deterministic candidate anchors the result: ML refines, never
        # reinvents. Reject masks that disagree with the ROI source size.
        anchor = np.asarray(cand.size, dtype=float)
        got = np.array([finalized.width_a_mm, finalized.height_b_mm])
        if np.any(np.abs(got - anchor) / np.maximum(anchor, 1e-6) > 0.25):
            continue
        if best_ml is None or finalized.confidence > best_ml.confidence:
            best_ml = finalized
    return best_ml


def measure_lens(
    image: Image.Image,
    calibration: CalibrationReady,
    capture_mode: Literal["loose", "framed"],
) -> LensMeasurement | MeasurementRetry:
    rectified = _rectify(image, calibration)
    if capture_mode == "loose":
        rim_candidates = _sheet_dark_rim_candidates(rectified, calibration)
        if rim_candidates:
            best_rim = rim_candidates[0]
            return _finalize(best_rim.contour, calibration, capture_mode, best_rim.score, ml=False)

    candidates, cropped = _candidate_contours(rectified, calibration)
    if not candidates:
        sheet_candidates = _sheet_zone_candidates(rectified, calibration)
        if sheet_candidates:
            best_sheet = sheet_candidates[0]
            if not (
                len(sheet_candidates) > 1
                and sheet_candidates[1].area >= best_sheet.area * .65
                and sheet_candidates[1].score >= best_sheet.score * .80
            ):
                return _finalize(best_sheet.contour, calibration, capture_mode, best_sheet.score, ml=False)
        return retry("lens_cropped" if cropped else "lens_missing")
    best = candidates[0]
    # Deterministic success stays primary; ML rescues weak-edge rejects.
    if best.sharpness >= 35 and best.score >= .48 and not (
        len(candidates) > 1 and candidates[1].score >= best.score * .82
    ):
        return _finalize(best.contour, calibration, capture_mode, best.score, ml=False)
    if ml_segment.available() and not (
        len(candidates) > 1 and candidates[1].score >= best.score * .82
    ):
        rescued = _ml_rescue(rectified, calibration, capture_mode, candidates)
        if rescued is not None:
            return rescued
    if best.sharpness < 35:
        return retry("lens_blurry")
    if best.score < .48:
        return retry("low_confidence")
    return retry("ambiguous_contour")