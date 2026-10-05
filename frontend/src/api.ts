export type CaptureMode = "loose" | "framed";
export type Eye = "left" | "right";

export interface CalibrationReady {
  status: "calibrated";
  card_size_mm: [number, number];
  reference_type: "blank_card" | "aruco_sheet";
  reference_label: string;
  corners_px: [number, number][];
  rectified_card_corners_px: [number, number][];
  exclusion_zones_px: [number, number][][];
  source_to_rectified: number[][];
  source_to_card_mm: number[][];
  millimetres_per_pixel: number;
  confidence: number;
  preview_width_px: number;
  preview_height_px: number;
  preview_data_url: string;
}

export interface CalibrationRetry {
  status: "retry";
  reason: "card_missing" | "card_cropped" | "image_blurry" | "low_confidence";
  message: string;
}

export interface LensMeasurement {
  status: "measured";
  method: "direct_lens_contour" | "outer_rim_inset" | "ml_lens_mask" | "ml_rim_inset";
  approximate: boolean;
  inset_mm: number;
  width_a_mm: number;
  height_b_mm: number;
  perimeter_mm: number;
  confidence: number;
  contour_px: [number, number][];
  contour_mm: [number, number][];
  detected_outer_contour_px: [number, number][] | null;
}

export interface MeasurementRetry {
  status: "retry";
  reason: "lens_missing" | "lens_cropped" | "lens_blurry" | "ambiguous_contour" | "low_confidence";
  message: string;
}

export interface CaptureReceipt {
  capture_id: string;
  capture_mode: CaptureMode;
  eye: Eye;
  filename: string;
  width_px: number;
  height_px: number;
  size_bytes: number;
  status: "measured" | "retry";
  next_step: "capture_other_eye" | "retry_capture";
  calibration: CalibrationReady | CalibrationRetry;
  measurement: LensMeasurement | MeasurementRetry | null;
}

export async function submitCapture(
  image: File,
  captureMode: CaptureMode,
  eye: Eye,
): Promise<CaptureReceipt> {
  const body = new FormData();
  body.append("image", image);
  body.append("capture_mode", captureMode);
  body.append("eye", eye);

  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch("/api/captures", {
      method: "POST",
      body,
      signal: controller.signal,
    });

    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as
        | { detail?: unknown }
        | null;
      throw new Error(typeof payload?.detail === "string" ? payload.detail : "The image could not be measured. Please try again.");
    }

    return await response.json() as CaptureReceipt;
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error("Measurement took too long. Check your connection and try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export interface FrameParams {
  bridgeWidthMm: number;
  rimWidthMm: number;
  thicknessMm: number;
  clearanceMm: number;
}

async function requestFrame(
  endpoint: "/api/frame-stl" | "/api/frame-glb",
  left: LensMeasurement,
  right: LensMeasurement,
  params: FrameParams,
  layout: "assembled" | "print" = "assembled",
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      left_contour_mm: left.contour_mm,
      right_contour_mm: right.contour_mm,
      bridge_width_mm: params.bridgeWidthMm,
      rim_width_mm: params.rimWidthMm,
      thickness_mm: params.thicknessMm,
      clearance_mm: params.clearanceMm,
      layout,
    }),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: unknown }
      | null;
    throw new Error(typeof payload?.detail === "string" ? payload.detail : "The frame could not be generated.");
  }
  return await response.blob();
}

export function downloadFrameStl(
  left: LensMeasurement,
  right: LensMeasurement,
  params: FrameParams,
  layout: "assembled" | "print" = "assembled",
): Promise<Blob> {
  return requestFrame("/api/frame-stl", left, right, params, layout);
}

export function fetchFrameGlb(
  left: LensMeasurement,
  right: LensMeasurement,
  params: FrameParams,
  signal?: AbortSignal,
): Promise<Blob> {
  return requestFrame("/api/frame-glb", left, right, params, "assembled", signal);
}
