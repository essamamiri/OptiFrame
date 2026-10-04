import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  Check,
  CircleHelp,
  Glasses,
  ImagePlus,
  RotateCcw,
  ScanLine,
  ShieldCheck,
  Download,
  LoaderCircle,
} from "lucide-react";
import {
  CaptureReceipt,
  Eye,
  LensMeasurement,
  downloadFrameStl,
  submitCapture,
} from "./api";

const MAX_FILE_SIZE = 4 * 1024 * 1024;

function normalizeContour(points: [number, number][]) {
  const xs = points.map(point => point[0]);
  const ys = points.map(point => point[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  return points.map(([x, y]) => [x - cx, y - cy] as [number, number]);
}

function landscapeContour(points: [number, number][]) {
  const normalized = normalizeContour(points);
  const xs = normalized.map(point => point[0]);
  const ys = normalized.map(point => point[1]);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  if (height <= width) return normalized;
  return normalizeContour(normalized.map(([x, y]) => [y, -x] as [number, number]));
}

function offsetContour(points: [number, number][], offset: number) {
  return points.map(([x, y]) => {
    const length = Math.hypot(x, y) || 1;
    return [x + (x / length) * offset, y + (y / length) * offset] as [number, number];
  });
}

function contourPath(points: [number, number][], dx = 0, dy = 0) {
  if (!points.length) return "";
  return points.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${(x + dx).toFixed(2)} ${(y + dy).toFixed(2)}`).join(" ") + " Z";
}

function App() {
  const [eye, setEye] = useState<Eye>("left");
  const [images, setImages] = useState<Record<Eye, File | null>>({ left: null, right: null });
  const imageFile = images[eye];
  const captureMode = "loose";
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGeneratingFrame, setIsGeneratingFrame] = useState(false);
  const [bridgeWidth, setBridgeWidth] = useState(18);
  const [rimWidth, setRimWidth] = useState(4);
  const [frameThickness, setFrameThickness] = useState(3);
  const [clearance, setClearance] = useState(0.3);
  const [receipts, setReceipts] = useState<Record<Eye, CaptureReceipt | null>>({ left: null, right: null });
  const receipt = receipts[eye];
  function setImageFile(file: File | null) { setImages(previous => ({ ...previous, [eye]: file })); }
  function setReceipt(value: CaptureReceipt | null) { setReceipts(previous => ({ ...previous, [eye]: value })); }
  const measuredCount = (["left", "right"] as Eye[]).filter(side => receipts[side]?.measurement?.status === "measured").length;
  const cameraInput = useRef<HTMLInputElement>(null);
  const galleryInput = useRef<HTMLInputElement>(null);
  const calibration = receipt?.calibration;
  const measurement = receipt?.measurement;
  const leftMeasurement = receipts.left?.measurement?.status === "measured" ? receipts.left.measurement as LensMeasurement : null;
  const rightMeasurement = receipts.right?.measurement?.status === "measured" ? receipts.right.measurement as LensMeasurement : null;
  const framePreview = leftMeasurement && rightMeasurement ? {
    leftWidth: leftMeasurement.width_a_mm,
    leftHeight: leftMeasurement.height_b_mm,
    rightWidth: rightMeasurement.width_a_mm,
    rightHeight: rightMeasurement.height_b_mm,
    bridge: bridgeWidth,
    rim: rimWidth,
    clearance,
    bridgeY: -((leftMeasurement.height_b_mm + rightMeasurement.height_b_mm) / 2) * 0.23,
    leftPath: landscapeContour(leftMeasurement.contour_mm),
    rightPath: landscapeContour(rightMeasurement.contour_mm),
  } : null;
  const fitReport = leftMeasurement && rightMeasurement ? {
    symmetryDelta: Math.abs(leftMeasurement.width_a_mm - rightMeasurement.width_a_mm) + Math.abs(leftMeasurement.height_b_mm - rightMeasurement.height_b_mm),
    averageConfidence: (leftMeasurement.confidence + rightMeasurement.confidence) / 2,
    printableWidth: leftMeasurement.width_a_mm + rightMeasurement.width_a_mm + bridgeWidth + rimWidth * 4 + 20,
  } : null;
  const deliveryChecks = [
    { label: "Mobile photo capture", done: true },
    { label: "Reference correction", done: calibration?.status === "calibrated" },
    { label: "A / B / perimeter", done: measurement?.status === "measured" },
    { label: "1:1 SVG export", done: measurement?.status === "measured" },
    { label: "Two-lens STL", done: Boolean(leftMeasurement && rightMeasurement) },
  ];

  const previewUrl = useMemo(
    () => (imageFile ? URL.createObjectURL(imageFile) : ""),
    [imageFile],
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"].includes(file.type)) {
      setError("Choose an original JPG, PNG, HEIC, HEIF, or WebP image.");
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError("This image is larger than 4 MB. Retake it at a smaller resolution or choose a smaller original.");
      return;
    }

    setError("");
    setReceipt(null);
    setImageFile(file);
  }

  function resetCapture() {
    setImageFile(null);
    setReceipt(null);
    setError("");
  }

  async function continueCapture() {
    if (!imageFile) {
      setError("Take or choose a photo first.");
      return;
    }

    setIsSubmitting(true);
    setError("");
    setReceipt(null);
    try {
      const response = await submitCapture(imageFile, captureMode, eye);
      setReceipt(response);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "We could not validate the image. Please try again.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  function downloadOutline() {
    if (measurement?.status !== "measured" || measurement.contour_mm.length < 3) return;
    const points = measurement.contour_mm;
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const minX = Math.min(...xs), minY = Math.min(...ys);
    const width = Math.max(...xs) - minX + 4, height = Math.max(...ys) - minY + 4;
    if (![width, height].every(v => Number.isFinite(v) && v > 0)) return;
    const contour = points.map(([x,y]) => `${x-minX+2},${y-minY+2}`).join(" ");
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}mm" height="${height}mm" viewBox="0 0 ${width} ${height}"><polygon points="${contour}" fill="none" stroke="black" stroke-width="0.2" /></svg>`;
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    const link = document.createElement("a"); link.href = url; link.download = `${eye}-lens-1to1.svg`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function downloadFrame() {
    if (!leftMeasurement || !rightMeasurement) {
      setError("Measure both lenses first.");
      return;
    }
    setIsGeneratingFrame(true);
    setError("");
    try {
      const blob = await downloadFrameStl(leftMeasurement, rightMeasurement, bridgeWidth, rimWidth, frameThickness, clearance);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "monture.stl";
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The frame STL could not be generated.");
    } finally {
      setIsGeneratingFrame(false);
    }
  }

  async function copyDemoSummary() {
    if (!leftMeasurement || !rightMeasurement || !fitReport) return;
    const summary = [
      "OptiFrame demo:",
      `Left lens: ${leftMeasurement.width_a_mm.toFixed(2)} x ${leftMeasurement.height_b_mm.toFixed(2)} mm`,
      `Right lens: ${rightMeasurement.width_a_mm.toFixed(2)} x ${rightMeasurement.height_b_mm.toFixed(2)} mm`,
      `CAD: bridge ${bridgeWidth.toFixed(1)} mm, rim ${rimWidth.toFixed(1)} mm, thickness ${frameThickness.toFixed(1)} mm, clearance ${clearance.toFixed(1)} mm`,
      `Fit confidence: ${(fitReport.averageConfidence * 100).toFixed(0)}%`,
      "Export: 1:1 SVG contours + printable STL frame front",
    ].join("\n");
    await navigator.clipboard.writeText(summary).catch(() => undefined);
  }

  function downloadFitCheck() {
    if (!leftMeasurement || !rightMeasurement) return;
    const left = landscapeContour(leftMeasurement.contour_mm);
    const right = landscapeContour(rightMeasurement.contour_mm);
    const leftOpening = offsetContour(left, clearance);
    const rightOpening = offsetContour(right, clearance);
    const leftX = leftMeasurement.width_a_mm / 2 + rimWidth + 4;
    const rightX = leftX + leftMeasurement.width_a_mm / 2 + rightMeasurement.width_a_mm / 2 + bridgeWidth + rimWidth * 2;
    const height = Math.max(leftMeasurement.height_b_mm, rightMeasurement.height_b_mm) + rimWidth * 4 + 10;
    const width = rightX + rightMeasurement.width_a_mm / 2 + rimWidth + 4;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width.toFixed(2)}mm" height="${height.toFixed(2)}mm" viewBox="0 ${(-height / 2).toFixed(2)} ${width.toFixed(2)} ${height.toFixed(2)}">
<rect width="100%" height="100%" y="${(-height / 2).toFixed(2)}" fill="white"/>
<path d="${contourPath(left, leftX, 0)}" fill="none" stroke="black" stroke-width="0.25"/>
<path d="${contourPath(right, rightX, 0)}" fill="none" stroke="black" stroke-width="0.25"/>
<path d="${contourPath(leftOpening, leftX, 0)}" fill="none" stroke="#00a36c" stroke-width="0.35"/>
<path d="${contourPath(rightOpening, rightX, 0)}" fill="none" stroke="#00a36c" stroke-width="0.35"/>
<text x="2" y="${(-height / 2 + 5).toFixed(2)}" font-size="3" fill="black">Black = measured lens. Green = frame opening with ${clearance.toFixed(1)} mm clearance. Print at 100%.</text>
</svg>`;
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "fit-check-1to1.svg";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="OptiFrame home">
          <span className="brand-mark"><Glasses size={25} strokeWidth={2.25} /></span>
          <span>OptiFrame</span>
        </a>
        <a className="help-button" href="#photo-guide" aria-label="Jump to capture guide">
          <CircleHelp size={20} />
        </a>
      </header>

      <main id="top">
        <section className="progress-wrap" aria-label="Progress">
          <div className="progress-copy">
            <span>{measuredCount} / 2 lenses measured</span>
            <span>Measure</span>
          </div>
          <div className="progress-track"><span style={{width: `${measuredCount * 50}%`}} /></div>
          <ol className="progress-labels">
            <li>Capture</li>
            <li className="active">Measure</li>
            <li>Review</li>
            <li>STL</li>
          </ol>
        </section>

        <section className="hero">
          <div className="eyebrow"><ScanLine size={14} /> Precision lens workspace</div>
          <h1>Old lenses.<br /><span>New possibilities.</span></h1>
          <p>Choose a lens. Take a photo. Reveal its shape and dimensions.</p>
        </section>

        <section className="demo-status" aria-label="Challenge deliverables">
          <div>
            <span className="status-kicker">Consignes check</span>
            <strong>Photo {"->"} millimetres {"->"} STL</strong>
          </div>
          <ul>
            {deliveryChecks.map(item => (
              <li key={item.label} className={item.done ? "done" : ""}>
                <Check size={15} />
                {item.label}
              </li>
            ))}
          </ul>
        </section>

        <section className="capture-card">
          <fieldset className="field-group compact" disabled={isSubmitting}>
            <legend>1. Choose a lens</legend>
            <div className="segmented-control">
              <button
                type="button"
                className={eye === "left" ? "active" : ""}
                aria-pressed={eye === "left"}
                onClick={() => { setEye("left"); setError(""); }}
              >Left lens <small>{receipts.left?.measurement?.status === "measured" ? "Measured" : images.left ? "Photo ready" : "Add photo"}</small></button>
              <button
                type="button"
                className={eye === "right" ? "active" : ""}
                aria-pressed={eye === "right"}
                onClick={() => { setEye("right"); setError(""); }}
              >Right lens <small>{receipts.right?.measurement?.status === "measured" ? "Measured" : images.right ? "Photo ready" : "Add photo"}</small></button>
            </div>
            <p className="orientation-note">Choose from the wearer’s point of view.</p>
          </fieldset>

          <div className="field-group" id="photo-guide">
            <div className="section-heading">
              <h2>2. Capture {eye} lens</h2>
              <span>Original images only</span>
            </div>

            {!imageFile ? (
              <div className="capture-stage">
                <div className="scan-target" aria-hidden="true">
                  <Camera size={46} strokeWidth={1.4} />
                  <span>READY TO CAPTURE</span>
                  <i className="corner top-left" /><i className="corner top-right" />
                  <i className="corner bottom-left" /><i className="corner bottom-right" />
                </div>
                <p>Keep one loose lens and all four sheet markers visible.</p>
                <div className="capture-actions">
                  <button className="primary-button" type="button" onClick={() => cameraInput.current?.click()}>
                    <Camera size={20} /> Take photo
                  </button>
                  <button className="secondary-button" type="button" onClick={() => galleryInput.current?.click()}>
                    <ImagePlus size={20} /> Upload photo
                  </button>
                </div>
              </div>
            ) : (
              <div className={`preview-stage ${isSubmitting ? "scanning" : ""}`} aria-busy={isSubmitting}>
                {isSubmitting && <div className="scan-overlay"><div className="scan-sweep" /><span><LoaderCircle size={16} /> Detecting reference and lens…</span></div>}
                {imageFile.type === "image/heic" || imageFile.type === "image/heif" ? (
                  <p className="preview-placeholder">HEIC/HEIF selected. The calibrated preview will appear after processing.</p>
                ) : <img src={previewUrl} alt="Selected lens and reference card" />}
                <button className="replace-button" type="button" disabled={isSubmitting} onClick={resetCapture}>
                  <RotateCcw size={17} /> Replace
                </button>
                <div className="file-summary">
                  <span>{imageFile.name}</span>
                  <span>{(imageFile.size / 1024 / 1024).toFixed(1)} MB</span>
                </div>
              </div>
            )}

            <input ref={cameraInput} hidden disabled={isSubmitting} type="file" accept="image/*" capture="environment" onChange={chooseImage} />
            <input ref={galleryInput} hidden disabled={isSubmitting} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif" onChange={chooseImage} />
          </div>

          <details className="photo-tips">
            <summary>How to get an accurate photo</summary>
            <ul className="checklist" aria-label="Photo checklist">
              <li><Check size={16} /> Use the printed marker sheet. Keep all four black markers visible.</li>
              <li><Check size={16} /> Place one loose lens inside the lens zone.</li>
              <li><Check size={16} /> Photograph from above; keep the sheet flat.</li>
              <li><Check size={16} /> Keep the whole lens visible, with no blur or strong glare.</li>
            </ul>
          </details>

          {error && <p className="error-message" role="alert">{error}</p>}

          {calibration?.status === "retry" && (
            <div className="error-message" role="alert">
              <strong>Retake photo — calibration incomplete</strong>
              <p>{calibration.message}</p>
              <button className="secondary-button" type="button" onClick={resetCapture}>Choose a new photo</button>
            </div>
          )}

          {measurement?.status === "retry" && (
            <div className="error-message" role="alert">
              <strong>Retake photo — lens measurement incomplete</strong>
              <p>{measurement.message}</p>
              <button className="secondary-button" type="button" onClick={resetCapture}>Choose a new photo</button>
            </div>
          )}

          {calibration?.status === "calibrated" && receipt ? (
            <section className="calibration-result" aria-label="Calibration result">
              <div className="success-card" role="status">
                <span><Check size={20} /></span>
                <div>
                  <strong>{measurement?.status === "measured" ? "Shape captured." : "Reference card calibrated"}</strong>
                  <p>{receipt.eye} eye · loose lens only</p>
                </div>
              </div>
              <figure className="rectified-preview">
                <div className="rectified-image">
                  <img src={calibration.preview_data_url} alt="Top-down rectified photo with detected reference card and lens overlays" />
                  <svg viewBox={`0 0 ${calibration.preview_width_px} ${calibration.preview_height_px}`} aria-hidden="true">
                    <polygon className="card-outline" points={calibration.rectified_card_corners_px.map(point => point.join(",")).join(" ")} />
                    {calibration.rectified_card_corners_px.map(([x, y], index) => (
                      <circle className="card-corner" key={index} cx={x} cy={y} r={calibration.preview_width_px / 130} />
                    ))}
                    {calibration.exclusion_zones_px.map((zone, index) => (
                      <polygon className="marker-outline" key={index} points={zone.map(point => point.join(",")).join(" ")} />
                    ))}
                    {measurement?.status === "measured" && measurement.detected_outer_contour_px && (
                      <polygon className="outer-contour" points={measurement.detected_outer_contour_px.map(point => point.join(",")).join(" ")} />
                    )}
                    {measurement?.status === "measured" && (
                      <polygon className="lens-contour" points={measurement.contour_px.map(point => point.join(",")).join(" ")} />
                    )}
                  </svg>
                </div>
                <figcaption>
                  Top-down preview · green marks the reference · {measurement?.status === "measured" ? "orange marks the measured lens contour" : "no reliable lens contour was found"}
                </figcaption>
              </figure>
              {measurement?.status === "measured" && (
                <dl className="measurement-grid" aria-label="Lens measurements">
                  <div><dt>Width A</dt><dd>{measurement.width_a_mm.toFixed(2)} <small>mm</small></dd></div>
                  <div><dt>Height B</dt><dd>{measurement.height_b_mm.toFixed(2)} <small>mm</small></dd></div>
                  <div><dt>Perimeter</dt><dd>{measurement.perimeter_mm.toFixed(2)} <small>mm</small></dd></div>
                </dl>
              )}
              <dl className="calibration-details">
                <div><dt>Reference</dt><dd>{calibration.reference_label}</dd></div>
                <div><dt>Preview scale</dt><dd>{calibration.millimetres_per_pixel.toFixed(4)} mm / pixel</dd></div>
                {measurement?.status === "measured" && <div><dt>Method</dt><dd>{measurement.method === "ml_lens_mask" ? "ML lens mask" : "Direct detected contour"}</dd></div>}
              </dl>
              {measurement?.status === "measured" && (
                <div className="pipeline-view" aria-label="Step by step image pipeline">
                  <div>
                    <span>1</span>
                    <strong>Reference</strong>
                    <small>{calibration.reference_label}</small>
                  </div>
                  <div>
                    <span>2</span>
                    <strong>Redressed</strong>
                    <small>{calibration.millimetres_per_pixel.toFixed(4)} mm / px</small>
                  </div>
                  <div>
                    <span>3</span>
                    <strong>Contour</strong>
                    <small>{measurement.contour_mm.length} points</small>
                  </div>
                  <div>
                    <span>4</span>
                    <strong>CAD ready</strong>
                    <small>{measurement.width_a_mm.toFixed(1)} × {measurement.height_b_mm.toFixed(1)} mm</small>
                  </div>
                </div>
              )}
              <p className="calibration-note">
                {measurement?.status === "measured"
                  ? "Confirm the orange contour follows the loose lens edge. Real-photo accuracy has not been validated yet."
                  : "Card calibration succeeded, but the lens contour was not reliable enough to report dimensions."}
              </p>
              {measurement?.status === "measured" && <div className="result-actions">
                <button className="primary-button" type="button" onClick={downloadOutline}><Download size={18} /> Download outline SVG</button>
                <button className="secondary-button" type="button" onClick={() => { setEye(eye === "left" ? "right" : "left"); setError(""); }}>View {eye === "left" ? "right" : "left"} lens</button>
              </div>}
              <button className="secondary-button" type="button" onClick={resetCapture}><RotateCcw size={17} /> Retake this lens</button>
            </section>
          ) : (
            <button
              className="continue-button"
              type="button"
              disabled={!imageFile || isSubmitting}
              onClick={continueCapture}
            >
              {isSubmitting ? "Calibrating and measuring…" : receipt?.status === "retry" ? "Try measurement again" : "Measure lens"}
              {isSubmitting && <LoaderCircle size={20} className="spinner" />}
            </button>
          )}
        </section>

        {leftMeasurement && rightMeasurement && (
          <section className="frame-panel" aria-label="Frame export">
            {framePreview && (
              <div className="frame-previews">
                <svg className="frame-preview" viewBox="-62 -34 124 68" role="img" aria-label="Top view preview of generated frame">
                  <line className="frame-bridge" x1={-framePreview.bridge / 2} y1={framePreview.bridgeY} x2={framePreview.bridge / 2} y2={framePreview.bridgeY} />
                  <line className="frame-tab" x1={-(framePreview.bridge / 2 + framePreview.leftWidth + 10)} y1="0" x2={-(framePreview.bridge / 2 + framePreview.leftWidth / 2 + 1)} y2="0" />
                  <line className="frame-tab" x1={framePreview.bridge / 2 + framePreview.rightWidth / 2 + 1} y1="0" x2={framePreview.bridge / 2 + framePreview.rightWidth + 10} y2="0" />
                  <path
                    className="frame-rim"
                    d={contourPath(offsetContour(framePreview.leftPath, framePreview.rim + framePreview.clearance), -(framePreview.bridge / 2 + framePreview.leftWidth / 2), 0)}
                  />
                  <path
                    className="frame-opening"
                    d={contourPath(offsetContour(framePreview.leftPath, framePreview.clearance), -(framePreview.bridge / 2 + framePreview.leftWidth / 2), 0)}
                  />
                  <path
                    className="frame-rim"
                    d={contourPath(offsetContour(framePreview.rightPath, framePreview.rim + framePreview.clearance), framePreview.bridge / 2 + framePreview.rightWidth / 2, 0)}
                  />
                  <path
                    className="frame-opening"
                    d={contourPath(offsetContour(framePreview.rightPath, framePreview.clearance), framePreview.bridge / 2 + framePreview.rightWidth / 2, 0)}
                  />
                </svg>
                <svg className="frame-preview frame-preview-3d" viewBox="-62 -44 124 88" role="img" aria-label="Angled 3D preview of generated frame">
                  <g className="frame-shadow" transform="translate(5 9) scale(1 .72)">
                    <line className="frame-bridge" x1={-framePreview.bridge / 2} y1={framePreview.bridgeY} x2={framePreview.bridge / 2} y2={framePreview.bridgeY} />
                    <line className="frame-tab" x1={-(framePreview.bridge / 2 + framePreview.leftWidth + 10)} y1="0" x2={-(framePreview.bridge / 2 + framePreview.leftWidth / 2 + 1)} y2="0" />
                    <line className="frame-tab" x1={framePreview.bridge / 2 + framePreview.rightWidth / 2 + 1} y1="0" x2={framePreview.bridge / 2 + framePreview.rightWidth + 10} y2="0" />
                    <path className="frame-rim" d={contourPath(offsetContour(framePreview.leftPath, framePreview.rim + framePreview.clearance), -(framePreview.bridge / 2 + framePreview.leftWidth / 2), 0)} />
                    <path className="frame-rim" d={contourPath(offsetContour(framePreview.rightPath, framePreview.rim + framePreview.clearance), framePreview.bridge / 2 + framePreview.rightWidth / 2, 0)} />
                  </g>
                  <g transform="translate(0 -2) scale(1 .72)">
                    <line className="frame-bridge" x1={-framePreview.bridge / 2} y1={framePreview.bridgeY} x2={framePreview.bridge / 2} y2={framePreview.bridgeY} />
                    <line className="frame-tab" x1={-(framePreview.bridge / 2 + framePreview.leftWidth + 10)} y1="0" x2={-(framePreview.bridge / 2 + framePreview.leftWidth / 2 + 1)} y2="0" />
                    <line className="frame-tab" x1={framePreview.bridge / 2 + framePreview.rightWidth / 2 + 1} y1="0" x2={framePreview.bridge / 2 + framePreview.rightWidth + 10} y2="0" />
                    <path className="frame-rim" d={contourPath(offsetContour(framePreview.leftPath, framePreview.rim + framePreview.clearance), -(framePreview.bridge / 2 + framePreview.leftWidth / 2), 0)} />
                    <path className="frame-opening" d={contourPath(offsetContour(framePreview.leftPath, framePreview.clearance), -(framePreview.bridge / 2 + framePreview.leftWidth / 2), 0)} />
                    <path className="frame-rim" d={contourPath(offsetContour(framePreview.rightPath, framePreview.rim + framePreview.clearance), framePreview.bridge / 2 + framePreview.rightWidth / 2, 0)} />
                    <path className="frame-opening" d={contourPath(offsetContour(framePreview.rightPath, framePreview.clearance), framePreview.bridge / 2 + framePreview.rightWidth / 2, 0)} />
                  </g>
                </svg>
              </div>
            )}
            <div className="cad-controls">
              <label>
                <span>Bridge</span>
                <input type="range" min="12" max="28" step="0.5" value={bridgeWidth} onChange={event => setBridgeWidth(Number(event.target.value))} />
                <strong>{bridgeWidth.toFixed(1)} mm</strong>
              </label>
              <label>
                <span>Rim</span>
                <input type="range" min="2.5" max="7" step="0.5" value={rimWidth} onChange={event => setRimWidth(Number(event.target.value))} />
                <strong>{rimWidth.toFixed(1)} mm</strong>
              </label>
              <label>
                <span>Thickness</span>
                <input type="range" min="2" max="5" step="0.5" value={frameThickness} onChange={event => setFrameThickness(Number(event.target.value))} />
                <strong>{frameThickness.toFixed(1)} mm</strong>
              </label>
              <label>
                <span>Clearance</span>
                <input type="range" min="0" max="1" step="0.1" value={clearance} onChange={event => setClearance(Number(event.target.value))} />
                <strong>{clearance.toFixed(1)} mm</strong>
              </label>
            </div>
            {fitReport && (
              <div className="fit-report">
                <div><span>Fit check</span><strong>{fitReport.averageConfidence >= 0.6 ? "Ready" : "Retake suggested"}</strong></div>
                <div><span>L/R delta</span><strong>{fitReport.symmetryDelta.toFixed(1)} mm</strong></div>
                <div><span>Frame width</span><strong>{fitReport.printableWidth.toFixed(1)} mm</strong></div>
              </div>
            )}
            <div>
              <span>Bridge</span>
              <strong>{bridgeWidth.toFixed(1)} mm</strong>
            </div>
            <div>
              <span>Left</span>
              <strong>{leftMeasurement.width_a_mm.toFixed(1)} × {leftMeasurement.height_b_mm.toFixed(1)} mm</strong>
            </div>
            <div>
              <span>Right</span>
              <strong>{rightMeasurement.width_a_mm.toFixed(1)} × {rightMeasurement.height_b_mm.toFixed(1)} mm</strong>
            </div>
            <button className="primary-button" type="button" onClick={downloadFrame} disabled={isGeneratingFrame}>
              <Download size={18} /> {isGeneratingFrame ? "Generating STL…" : "Download monture.stl"}
            </button>
            <button className="secondary-button" type="button" onClick={downloadFitCheck}>
              <Download size={18} /> Download 1:1 fit check
            </button>
            <button className="secondary-button" type="button" onClick={copyDemoSummary}>
              <Check size={18} /> Copy demo summary
            </button>
          </section>
        )}

        <p className="privacy-note"><ShieldCheck size={15} /> Photos are processed in memory and are not saved.</p>
      </main>
    </div>
  );
}

export default App;
