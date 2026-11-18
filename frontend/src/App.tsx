import { ChangeEvent, Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { Camera, Check, Download, ImagePlus, LoaderCircle, RotateCcw } from "lucide-react";
import {
  CaptureReceipt,
  Eye,
  LensMeasurement,
  downloadFrameStl,
  fetchFrameGlb,
  submitCapture,
} from "./api";
import { SAMPLE_LENSES } from "./sampleLenses";

const FrameViewer = lazy(() => import("./FrameViewer"));
const HeroScene = lazy(() => import("./HeroScene"));

const MAX_FILE_SIZE = 4 * 1024 * 1024;
type Step = "home" | Eye | "frame";
type Point = [number, number];

const STEPS: { id: Exclude<Step, "home">; label: string }[] = [
  { id: "left", label: "Left lens" },
  { id: "right", label: "Right lens" },
  { id: "frame", label: "Frame" },
];

/** Phone cameras often save 5–12 MB photos. Re-encode big ones in the browser (max 3000 px,
 *  JPEG 92%) so they fit the 4 MB upload limit; ~3000 px is still far more detail than needed. */
async function shrinkPhoto(file: File): Promise<File> {
  if (file.size <= MAX_FILE_SIZE) return file;
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, 3000 / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    bitmap.close();
    for (const quality of [0.92, 0.85, 0.75]) {
      const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, "image/jpeg", quality));
      if (blob && blob.size <= MAX_FILE_SIZE) {
        return new File([blob], file.name.replace(/\.[^.]+$/, "") + ".jpg", { type: "image/jpeg" });
      }
    }
  } catch {
    // Browser can't decode this format (e.g. HEIC outside Safari); fall through to the size check.
  }
  return file;
}

function normalizeContour(points: Point[]) {
  const xs = points.map(point => point[0]);
  const ys = points.map(point => point[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  return points.map(([x, y]) => [x - cx, y - cy] as Point);
}

function landscapeContour(points: Point[]) {
  const normalized = normalizeContour(points);
  if (normalized.length < 3) return normalized;
  const meanX = normalized.reduce((sum, [x]) => sum + x, 0) / normalized.length;
  const meanY = normalized.reduce((sum, [, y]) => sum + y, 0) / normalized.length;
  let xx = 0, xy = 0, yy = 0;
  for (const [x, y] of normalized) {
    const dx = x - meanX;
    const dy = y - meanY;
    xx += dx * dx;
    xy += dx * dy;
    yy += dy * dy;
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  let leveled = normalized.map(([x, y]) => {
    const dx = x - meanX;
    const dy = y - meanY;
    return [dx * cos - dy * sin, dx * sin + dy * cos] as Point;
  });
  const xs = leveled.map(point => point[0]);
  const ys = leveled.map(point => point[1]);
  if (Math.max(...ys) - Math.min(...ys) > Math.max(...xs) - Math.min(...xs)) {
    leveled = leveled.map(([x, y]) => [y, -x] as Point);
  }
  return normalizeContour(leveled);
}

function offsetContour(points: Point[], offset: number) {
  return points.map(([x, y]) => {
    const length = Math.hypot(x, y) || 1;
    return [x + (x / length) * offset, y + (y / length) * offset] as Point;
  });
}

function contourPath(points: Point[], dx = 0, dy = 0) {
  if (!points.length) return "";
  return points.map(([x, y], index) => `${index === 0 ? "M" : "L"} ${(x + dx).toFixed(2)} ${(y + dy).toFixed(2)}`).join(" ") + " Z";
}

function mm(value: number, digits = 2) {
  return value.toFixed(digits);
}

/** The measured lens drawn like an optician's tracing: real 5 mm grid, A and B dimension lines. */
function LensDrawing({ contour, widthA, heightB, placeholder = false }: {
  contour: Point[];
  widthA?: number;
  heightB?: number;
  placeholder?: boolean;
}) {
  const shape = useMemo(() => landscapeContour(contour), [contour]);
  const xs = shape.map(point => point[0]);
  const ys = shape.map(point => point[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const padLeft = 8, padTop = 8, padRight = 20, padBottom = 19;
  const viewX = minX - padLeft;
  const viewY = minY - padTop;
  const viewW = maxX - minX + padLeft + padRight;
  const viewH = maxY - minY + padTop + padBottom;
  const dimA = maxY + 11;
  const dimB = maxX + 12;
  const gap = 1.6;
  const labelA = placeholder ? "A" : `${mm(widthA ?? maxX - minX)}`;
  const labelB = placeholder ? "B" : `${mm(heightB ?? maxY - minY)}`;
  const gridStart = (value: number) => Math.floor(value / 5) * 5;

  return (
    <svg
      className={`lens-drawing ${placeholder ? "is-placeholder" : "is-measured"}`}
      viewBox={`${viewX} ${viewY} ${viewW} ${viewH}`}
      role="img"
      aria-label={placeholder ? "Example lens outline" : `Measured lens outline, width ${labelA} mm, height ${labelB} mm`}
    >
      <defs>
        <marker id="dim-arrow" viewBox="0 0 6 6" refX="5.4" refY="3" markerWidth="2.6" markerHeight="2.6" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
          <path d="M0 0.6 L5.6 3 L0 5.4" fill="none" className="dim-arrow" />
        </marker>
      </defs>
      <g className="mm-grid" aria-hidden="true">
        {Array.from({ length: Math.ceil(viewW / 5) + 2 }, (_, i) => gridStart(viewX) + i * 5).map(x => (
          <line key={`x${x}`} x1={x} x2={x} y1={viewY} y2={viewY + viewH} className={x % 10 === 0 ? "major" : ""} />
        ))}
        {Array.from({ length: Math.ceil(viewH / 5) + 2 }, (_, i) => gridStart(viewY) + i * 5).map(y => (
          <line key={`y${y}`} y1={y} y2={y} x1={viewX} x2={viewX + viewW} className={y % 10 === 0 ? "major" : ""} />
        ))}
      </g>
      <path className="lens-fill" d={contourPath(shape)} />
      <path className="lens-line" d={contourPath(shape)} pathLength={1} />
      <g className="dims">
        <line className="ext" x1={minX} x2={minX} y1={maxY + gap} y2={dimA + 2} />
        <line className="ext" x1={maxX} x2={maxX} y1={maxY + gap} y2={dimA + 2} />
        <line className="dim" x1={minX} x2={maxX} y1={dimA} y2={dimA} markerStart="url(#dim-arrow)" markerEnd="url(#dim-arrow)" />
        <text className="dim-label" x={(minX + maxX) / 2} y={dimA + 5.4} textAnchor="middle">{labelA}</text>
        <line className="ext" y1={minY} y2={minY} x1={maxX + gap} x2={dimB + 2} />
        <line className="ext" y1={maxY} y2={maxY} x1={maxX + gap} x2={dimB + 2} />
        <line className="dim" y1={minY} y2={maxY} x1={dimB} x2={dimB} markerStart="url(#dim-arrow)" markerEnd="url(#dim-arrow)" />
        <text className="dim-label" x={dimB + 5.2} y={(minY + maxY) / 2} textAnchor="middle" transform={`rotate(-90 ${dimB + 5.2} ${(minY + maxY) / 2})`}>{labelB}</text>
      </g>
    </svg>
  );
}

function App() {
  const [step, setStep] = useState<Step>(() => (new URLSearchParams(window.location.search).has("demo") ? "frame" : "home"));
  const eye: Eye = step === "right" ? "right" : "left";
  const [images, setImages] = useState<Record<Eye, File | null>>({ left: null, right: null });
  const [receipts, setReceipts] = useState<Record<Eye, CaptureReceipt | null>>({ left: null, right: null });
  const imageFile = images[eye];
  const receipt = receipts[eye];
  const captureMode = "loose";
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGeneratingFrame, setIsGeneratingFrame] = useState(false);
  const [bridgeWidth, setBridgeWidth] = useState(18);
  const [rimWidth, setRimWidth] = useState(4);
  const [frameThickness, setFrameThickness] = useState(3);
  const [clearance, setClearance] = useState(0.3);
  const cameraInput = useRef<HTMLInputElement>(null);
  const galleryInput = useRef<HTMLInputElement>(null);
  const calibration = receipt?.calibration;
  const measurement = receipt?.measurement;
  const measured = measurement?.status === "measured" ? measurement : null;
  const measuredLeft = receipts.left?.measurement?.status === "measured" ? receipts.left.measurement as LensMeasurement : null;
  const measuredRight = receipts.right?.measurement?.status === "measured" ? receipts.right.measurement as LensMeasurement : null;
  const [useSample, setUseSample] = useState(() => new URLSearchParams(window.location.search).has("demo"));
  const usingSample = useSample && !(measuredLeft && measuredRight);
  const leftMeasurement = measuredLeft ?? (usingSample ? SAMPLE_LENSES.left : null);
  const rightMeasurement = measuredRight ?? (usingSample ? SAMPLE_LENSES.right : null);
  const [glbUrl, setGlbUrl] = useState("");
  const [frameError, setFrameError] = useState("");
  const [heroGlb, setHeroGlb] = useState("");
  const storyRef = useRef<HTMLElement>(null);
  const storyProgress = useRef(0);
  const [chapter, setChapter] = useState(0);

  // Pinned scroll story on the landing page: 0..1 across the tall section.
  useEffect(() => {
    if (step !== "home") return;
    let frame = 0;
    function update() {
      frame = 0;
      const section = storyRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const travel = Math.max(rect.height - window.innerHeight, 1);
      const p = Math.min(1, Math.max(0, -rect.top / travel));
      storyProgress.current = p;
      setChapter(p < 0.24 ? 0 : p < 0.5 ? 1 : p < 0.76 ? 2 : 3);
    }
    function onScroll() { if (!frame) frame = requestAnimationFrame(update); }
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); cancelAnimationFrame(frame); };
  }, [step]);

  function setImageFile(file: File | null) { setImages(previous => ({ ...previous, [eye]: file })); }
  function setReceipt(value: CaptureReceipt | null) { setReceipts(previous => ({ ...previous, [eye]: value })); }
  function goTo(next: Step) {
    setStep(next);
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // The landing page's product shot: a frame built from the sample lenses.
  useEffect(() => {
    if (step !== "home" || heroGlb) return;
    const controller = new AbortController();
    fetchFrameGlb(SAMPLE_LENSES.left, SAMPLE_LENSES.right, {
      bridgeWidthMm: 18, rimWidthMm: 4, thicknessMm: 3, clearanceMm: 0.3,
    }, controller.signal)
      .then(blob => setHeroGlb(URL.createObjectURL(blob)))
      .catch(() => undefined);
    return () => controller.abort();
  }, [step, heroGlb]);

  useEffect(() => {
    if (!leftMeasurement || !rightMeasurement) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      fetchFrameGlb(leftMeasurement, rightMeasurement, {
        bridgeWidthMm: bridgeWidth, rimWidthMm: rimWidth, thicknessMm: frameThickness, clearanceMm: clearance,
      }, controller.signal)
        .then(blob => {
          setFrameError("");
          setGlbUrl(previous => {
            if (previous) URL.revokeObjectURL(previous);
            return URL.createObjectURL(blob);
          });
        })
        .catch(caught => {
          if (!controller.signal.aborted) setFrameError(caught instanceof Error ? caught.message : "The 3D preview could not be generated.");
        });
    }, 350);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [leftMeasurement, rightMeasurement, bridgeWidth, rimWidth, frameThickness, clearance]);

  const previewUrl = useMemo(() => (imageFile ? URL.createObjectURL(imageFile) : ""), [imageFile]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  async function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const picked = event.target.files?.[0];
    event.target.value = "";
    if (!picked) return;
    if (!["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"].includes(picked.type)) {
      setError("This file type isn't supported. Use a JPG, PNG, HEIC or WebP photo.");
      return;
    }
    const file = await shrinkPhoto(picked);
    if (file.size > MAX_FILE_SIZE) {
      setError("This photo is over 4 MB and couldn't be made smaller here. Take it again at a lower resolution.");
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

  async function measureLens() {
    if (!imageFile) return;
    setIsSubmitting(true);
    setError("");
    setReceipt(null);
    try {
      setReceipt(await submitCapture(imageFile, captureMode, eye));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The photo couldn't be sent. Check your connection and measure again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  function downloadOutline() {
    if (!measured || measured.contour_mm.length < 3) return;
    const points = measured.contour_mm;
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const minX = Math.min(...xs), minY = Math.min(...ys);
    const width = Math.max(...xs) - minX + 4, height = Math.max(...ys) - minY + 4;
    if (![width, height].every(v => Number.isFinite(v) && v > 0)) return;
    const contour = points.map(([x, y]) => `${x - minX + 2},${y - minY + 2}`).join(" ");
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}mm" height="${height}mm" viewBox="0 0 ${width} ${height}"><polygon points="${contour}" fill="none" stroke="black" stroke-width="0.2" /></svg>`;
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    const link = document.createElement("a"); link.href = url; link.download = `${eye}-lens-1to1.svg`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function downloadFrame(layout: "assembled" | "print" = "assembled") {
    if (!leftMeasurement || !rightMeasurement) return;
    setIsGeneratingFrame(true);
    setError("");
    try {
      const blob = await downloadFrameStl(leftMeasurement, rightMeasurement, {
        bridgeWidthMm: bridgeWidth, rimWidthMm: rimWidth, thicknessMm: frameThickness, clearanceMm: clearance,
      }, layout);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = layout === "print" ? "monture-print-plate.stl" : "monture.stl";
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The frame file couldn't be generated. Try again.");
    } finally {
      setIsGeneratingFrame(false);
    }
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
<path d="${contourPath(leftOpening, leftX, 0)}" fill="none" stroke="#0071e3" stroke-width="0.35"/>
<path d="${contourPath(rightOpening, rightX, 0)}" fill="none" stroke="#0071e3" stroke-width="0.35"/>
<text x="2" y="${(-height / 2 + 5).toFixed(2)}" font-size="3" fill="black">Black = measured lens. Blue = frame opening with ${clearance.toFixed(1)} mm clearance. Print at 100%.</text>
</svg>`;
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "fit-check-1to1.svg";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const stepDone = (id: Step) => (id === "left" || id === "right") && receipts[id]?.measurement?.status === "measured";
  const otherEye: Eye = eye === "left" ? "right" : "left";
  const nextStep: Step = eye === "left" ? (measuredRight ? "frame" : "right") : (measuredLeft ? "frame" : "left");
  const calibrationFailed = calibration?.status === "retry";
  const measurementFailed = measurement?.status === "retry";

  const nav = (
    <header className="nav">
      <div className="nav-inner">
        <button className="wordmark" type="button" onClick={() => goTo("home")} aria-label="OptiFrame home">
          <svg className="mark" viewBox="0 0 40 18" aria-hidden="true">
            <ellipse cx="10" cy="9" rx="8" ry="7" />
            <ellipse cx="30" cy="9" rx="8" ry="7" />
            <path d="M18 7.5 Q20 5.5 22 7.5" />
          </svg>
          OptiFrame
        </button>
        {step === "home" ? (
          <button className="btn small" type="button" onClick={() => goTo("left")}>Start measuring</button>
        ) : (
          <nav aria-label="Steps">
            <ol className="steps">
              {STEPS.map((item, index) => (
                <li key={item.id}>
                  <button
                    type="button"
                    className={`${step === item.id ? "is-current" : ""} ${stepDone(item.id) ? "is-done" : ""}`}
                    aria-current={step === item.id ? "step" : undefined}
                    onClick={() => goTo(item.id)}
                    disabled={isSubmitting}
                  >
                    <span className="step-dot">{stepDone(item.id) ? <Check size={11} strokeWidth={3.2} aria-hidden="true" /> : index + 1}</span>
                    <span className="step-label">{item.label}</span>
                  </button>
                </li>
              ))}
            </ol>
          </nav>
        )}
      </div>
    </header>
  );

  if (step === "home") {
    return (
      <div className="app">
        {nav}
        <main>
          <section className="scrolly" ref={storyRef} aria-label="How OptiFrame works">
            <div className="scrolly-pin" data-chapter={chapter}>
              <div className="scrolly-glow" aria-hidden="true" />
              {heroGlb && (
                <Suspense fallback={null}>
                  <HeroScene glbUrl={heroGlb} progress={storyProgress} />
                </Suspense>
              )}

              <div className={`chapter chapter-hero ${chapter === 0 ? "is-active" : ""}`}>
                <h1 className="hero-title">
                  <span className="line"><span>A new frame</span></span>
                  <span className="line"><span>for the lenses</span></span>
                  <span className="line"><span>you already have.</span></span>
                </h1>
                <p className="hero-sub">Photograph each lens next to a bank card. OptiFrame traces its exact outline and turns it into a frame you can 3D print.</p>
                <div className="hero-actions">
                  <button className="btn" type="button" onClick={() => goTo("left")}>Measure your lenses</button>
                  <button className="link" type="button" onClick={() => { setUseSample(true); goTo("frame"); }}>See a sample frame</button>
                </div>
              </div>

              {[
                { n: 1, title: "Keep the lenses.", body: "Old prescription glasses, a snapped frame, a pair you’ve outgrown. The lenses are usually fine, so they’re what you start with." },
                { n: 2, title: "Photograph each one next to a card.", body: "A bank card gives the photo its scale. OptiFrame finds the lens edge and measures its width, height and full outline in millimetres." },
                { n: 3, title: "Print a frame that fits them.", body: "Download the frame file, print it, and press the lenses into rims shaped from their own outlines." },
              ].map(item => (
                <div key={item.n} className={`chapter ${chapter === item.n ? "is-active" : ""}`}>
                  <p className="chapter-count"><span>{item.n}</span> / 3</p>
                  <h2>{item.title}</h2>
                  <p>{item.body}</p>
                  {item.n === 3 && <button className="btn" type="button" onClick={() => goTo("left")}>Measure your lenses</button>}
                </div>
              ))}

              <div className="scroll-cue" aria-hidden="true"><span style={{ transform: `scaleY(${0.08 + chapter / 3 * 0.92})` }} /></div>
            </div>
          </section>

          <section className="story">
            <div className="showcase">
              <div className="showcase-copy">
                <h2 className="story-title">What you get back from one photo</h2>
                <p>A tracing of the lens on a 5 mm grid, with its boxed width and height, the way an optician’s tracer would read it. Download it as a 1:1 outline, or carry it straight into the frame.</p>
              </div>
              <div className="showcase-art">
                <LensDrawing contour={SAMPLE_LENSES.left.contour_mm} widthA={SAMPLE_LENSES.left.width_a_mm} heightB={SAMPLE_LENSES.left.height_b_mm} />
              </div>
            </div>
          </section>

          <section className="closing">
            <h2>Got your lenses nearby?</h2>
            <p>It takes about a minute per lens.</p>
            <button className="btn" type="button" onClick={() => goTo("left")}>Measure your lenses</button>
          </section>
        </main>
        <footer className="footer">Photos are measured in memory and never stored.</footer>
      </div>
    );
  }

  return (
    <div className="app">
      {nav}

      {step !== "frame" ? (
        <main className="flow" key={eye}>
          <div className={`stage ${isSubmitting ? "is-scanning" : ""}`} aria-busy={isSubmitting}>
            {measured ? (
              <LensDrawing key={receipt?.capture_id} contour={measured.contour_mm} widthA={measured.width_a_mm} heightB={measured.height_b_mm} />
            ) : imageFile ? (
              <div className="stage-photo">
                {imageFile.type === "image/heic" || imageFile.type === "image/heif"
                  ? <p className="stage-note">HEIC photo selected. It will show here once measured.</p>
                  : <img src={previewUrl} alt={`Photo of the ${eye} lens next to the card`} />}
                {isSubmitting && <div className="scan-line" aria-hidden="true" />}
              </div>
            ) : (
              <LensDrawing contour={SAMPLE_LENSES.left.contour_mm} placeholder />
            )}
          </div>

          <section className="panel" aria-labelledby="step-title">
            <p className="step-count">Step {eye === "left" ? 1 : 2} of 3</p>
            <h1 id="step-title" className="title">{eye === "left" ? "Left lens" : "Right lens"}</h1>

            {measured ? (
              <>
                <dl className="readout">
                  <div><dt>Width A</dt><dd>{mm(measured.width_a_mm)}<span>mm</span></dd></div>
                  <div><dt>Height B</dt><dd>{mm(measured.height_b_mm)}<span>mm</span></dd></div>
                  <div><dt>Perimeter</dt><dd>{mm(measured.perimeter_mm, 1)}<span>mm</span></dd></div>
                </dl>
                <div className="actions">
                  <button className="btn wide" type="button" onClick={() => goTo(nextStep)}>
                    {nextStep === "frame" ? "Continue to frame" : `Measure the ${otherEye} lens`}
                  </button>
                  <div className="secondary-actions">
                    <button className="link" type="button" onClick={downloadOutline}><Download size={16} aria-hidden="true" /> Download outline</button>
                    <button className="link" type="button" onClick={resetCapture}><RotateCcw size={15} aria-hidden="true" /> Retake photo</button>
                  </div>
                </div>
                {calibration?.status === "calibrated" && (
                  <details className="check">
                    <summary>Check the outline on your photo</summary>
                    <figure>
                      <div className="check-image">
                        <img src={calibration.preview_data_url} alt="Your photo, flattened, with the detected lens outline" />
                        <svg viewBox={`0 0 ${calibration.preview_width_px} ${calibration.preview_height_px}`} aria-hidden="true">
                          <polygon className="ref-outline" points={calibration.rectified_card_corners_px.map(point => point.join(",")).join(" ")} />
                          {calibration.exclusion_zones_px.map((zone, index) => (
                            <polygon className="ref-outline" key={index} points={zone.map(point => point.join(",")).join(" ")} />
                          ))}
                          <polygon className="lens-outline" points={measured.contour_px.map(point => point.join(",")).join(" ")} />
                        </svg>
                      </div>
                      <figcaption>The pink line should sit right on the lens edge. If it doesn’t, retake the photo.</figcaption>
                    </figure>
                  </details>
                )}
              </>
            ) : calibrationFailed || measurementFailed ? (
              <>
                <div className="problem" role="alert">
                  <h2>{calibrationFailed ? "The photo couldn’t be measured" : "The lens couldn’t be measured"}</h2>
                  <p className="problem-tip">Try putting the lens and card on a sheet of white paper. It’s the fix for most failed photos.</p>
                  <p>{calibration?.status === "retry" ? calibration.message : measurement?.status === "retry" ? measurement.message : ""}</p>
                </div>
                <div className="actions">
                  <button className="btn wide" type="button" onClick={() => cameraInput.current?.click()}><Camera size={18} aria-hidden="true" /> Retake photo</button>
                  <button className="link center" type="button" onClick={() => galleryInput.current?.click()}>Choose another photo</button>
                </div>
              </>
            ) : imageFile ? (
              <>
                <p className="lede">Make sure the whole card and the whole lens are in the photo.</p>
                <div className="actions">
                  <button className="btn wide" type="button" onClick={measureLens} disabled={isSubmitting}>
                    {isSubmitting ? <><LoaderCircle size={18} className="spin" aria-hidden="true" /> Measuring</> : "Measure lens"}
                  </button>
                  <button className="link center" type="button" onClick={resetCapture} disabled={isSubmitting}>Replace photo</button>
                </div>
              </>
            ) : (
              <>
                <p className="lede">The one in front of your {eye} eye. Put it next to a bank card on white paper and take a photo from above.</p>
                <ol className="setup">
                  <li><strong>White paper.</strong> Put the lens and any bank-size card (debit, credit, transit, ID) on a sheet of white printer paper.</li>
                  <li><strong>Close together.</strong> Keep the lens about a finger’s width from the card.</li>
                  <li><strong>Straight above.</strong> Keep the card and lens fully inside the paper and the photo, with some space around them.</li>
                </ol>
                <div className="actions">
                  <button className="btn wide" type="button" onClick={() => cameraInput.current?.click()}><Camera size={18} aria-hidden="true" /> Take photo</button>
                  <button className="link center" type="button" onClick={() => galleryInput.current?.click()}><ImagePlus size={16} aria-hidden="true" /> Choose from library</button>
                </div>
              </>
            )}

            {error && <p className="problem inline" role="alert">{error}</p>}
            <input ref={cameraInput} hidden type="file" accept="image/*" capture="environment" onChange={chooseImage} />
            <input ref={galleryInput} hidden type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif" onChange={chooseImage} />
          </section>
        </main>
      ) : leftMeasurement && rightMeasurement ? (
        <main className="flow frame-flow">
          <div className="frame-head">
            <h1 className="title">{usingSample ? "Sample frame" : "Your frame"}</h1>
            <p className="lede">
              {usingSample
                ? "Built from two sample lenses. Measure your own to get a frame that fits them."
                : "Built around the two lenses you measured. Adjust it, then download it for your 3D printer."}
            </p>
          </div>
          <div className="viewer-tile">
            {glbUrl ? (
              <Suspense fallback={<div className="viewer-stage"><span className="viewer-loading">Loading 3D preview</span></div>}>
                <FrameViewer glbUrl={glbUrl} caption={`Bridge ${bridgeWidth.toFixed(1)} mm`} />
              </Suspense>
            ) : (
              <div className="viewer-stage"><span className="viewer-loading">{frameError || "Building 3D preview"}</span></div>
            )}
          </div>

          <div className="frame-grid">
            <section className="card" aria-label="Frame adjustments">
              <h2>Adjust</h2>
              {[
                { label: "Bridge", value: bridgeWidth, set: setBridgeWidth, min: 12, max: 28, step: 0.5, hint: "Gap between the lenses" },
                { label: "Rim", value: rimWidth, set: setRimWidth, min: 2.5, max: 7, step: 0.5, hint: "Frame width around each lens" },
                { label: "Thickness", value: frameThickness, set: setFrameThickness, min: 2, max: 5, step: 0.5, hint: "Depth of the frame front" },
                { label: "Clearance", value: clearance, set: setClearance, min: 0, max: 1, step: 0.1, hint: "Extra room so the lens clicks in" },
              ].map(control => (
                <label key={control.label} className="slider">
                  <span className="slider-head">
                    <span>{control.label}<small>{control.hint}</small></span>
                    <output>{control.value.toFixed(1)} mm</output>
                  </span>
                  <input type="range" min={control.min} max={control.max} step={control.step} value={control.value}
                    onChange={event => control.set(Number(event.target.value))} />
                </label>
              ))}
            </section>

            <section className="card" aria-label="Download">
              <h2>Download</h2>
              <dl className="summary">
                <div><dt>Left lens</dt><dd>{mm(leftMeasurement.width_a_mm, 1)} × {mm(leftMeasurement.height_b_mm, 1)} mm</dd></div>
                <div><dt>Right lens</dt><dd>{mm(rightMeasurement.width_a_mm, 1)} × {mm(rightMeasurement.height_b_mm, 1)} mm</dd></div>
                <div><dt>Frame width</dt><dd>{mm(leftMeasurement.width_a_mm + rightMeasurement.width_a_mm + bridgeWidth + rimWidth * 4 + 20, 1)} mm</dd></div>
              </dl>
              <button className="btn wide" type="button" onClick={() => downloadFrame("assembled")} disabled={isGeneratingFrame}>
                {isGeneratingFrame ? "Preparing file" : "Download frame (STL)"}
              </button>
              <div className="secondary-actions stacked">
                <button className="link" type="button" onClick={() => downloadFrame("print")} disabled={isGeneratingFrame}><Download size={16} aria-hidden="true" /> Flat print layout</button>
                <button className="link" type="button" onClick={downloadFitCheck}><Download size={16} aria-hidden="true" /> 1:1 fit check</button>
              </div>
              <p className="fine">Print the fit check at 100% and lay your lenses on it first. The blue line is where each lens will sit.</p>
              {usingSample && <button className="link" type="button" onClick={() => { setUseSample(false); goTo("left"); }}>Measure my own lenses</button>}
              {error && <p className="problem inline" role="alert">{error}</p>}
            </section>
          </div>
        </main>
      ) : (
        <main className="flow empty-flow">
          <h1 className="title">Your frame</h1>
          <p className="lede">It’s built from both lens outlines. {!measuredLeft && !measuredRight ? "Measure both lenses first." : `Measure the ${measuredLeft ? "right" : "left"} lens to continue.`}</p>
          <div className="hero-actions">
            <button className="btn" type="button" onClick={() => goTo(measuredLeft ? "right" : "left")}>Measure the {measuredLeft ? "right" : "left"} lens</button>
            <button className="link" type="button" onClick={() => setUseSample(true)}>See a sample frame</button>
          </div>
        </main>
      )}

      <footer className="footer">Photos are measured in memory and never stored.</footer>
    </div>
  );
}

export default App;