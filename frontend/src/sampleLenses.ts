import type { LensMeasurement } from "./api";

// Synthetic rectangular-panto lenses so the frame preview can be explored
// without a printed sheet. Clearly labelled as samples in the UI.
function sampleLens(width: number, height: number): LensMeasurement {
  const points: [number, number][] = [];
  for (let i = 0; i < 240; i += 1) {
    const t = (i / 240) * Math.PI * 2;
    const c = Math.cos(t);
    const s = Math.sin(t);
    const x = Math.sign(c) * Math.abs(c) ** (2 / 3.4) * width / 2;
    const y = Math.sign(s) * Math.abs(s) ** (2 / (s > 0 ? 2.2 : 3.4)) * height / 2;
    points.push([105 + x, 148 + y]);
  }
  let perimeter = 0;
  points.forEach(([x, y], i) => {
    const [nx, ny] = points[(i + 1) % points.length];
    perimeter += Math.hypot(nx - x, ny - y);
  });
  return {
    status: "measured",
    method: "direct_lens_contour",
    approximate: false,
    inset_mm: 0,
    width_a_mm: width,
    height_b_mm: height,
    perimeter_mm: Math.round(perimeter * 100) / 100,
    confidence: 1,
    contour_px: [],
    contour_mm: points,
    detected_outer_contour_px: null,
  };
}

export const SAMPLE_LENSES = { left: sampleLens(51, 37), right: sampleLens(51, 37) };
