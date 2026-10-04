# OptiFrame

OptiFrame is a hackathon prototype that turns phone photos of loose recycled lenses into measured lens contours, printable fit checks, and a generated eyeglass-frame STL.

The app is built for the contest workflow: place one loose lens on the marker sheet, take a photo, repeat for the other lens, then export CAD files for a custom frame.

## What It Does

- Calibrates photos from a 4-marker ArUco capture sheet.
- Measures loose lens width A, height B, perimeter, and contour.
- Works with real phone photos, including rotated lenses and mild perspective.
- Shows the measured contour directly on the corrected top-down preview.
- Captures left and right lenses separately.
- Exports lens outlines as SVG.
- Generates a matching frame STL from the measured lens contours.
- Includes CAD controls for rim thickness, depth, bridge width, tab length, and lens clearance.
- Exports a 1:1 fit-check SVG so the lens can be compared before printing.

## Why It Is Different

Most AI hackathon demos stop at detection. OptiFrame connects the full pipeline:

photo -> calibrated measurement -> contour extraction -> fit validation -> CAD export.

That makes it easier to explain live: "I photographed these loose recycled lenses, the app measured them, and this downloadable STL was generated from those exact contours."

## Project Structure

```text
backend/    FastAPI + OpenCV measurement and STL generation
frontend/   React + Vite mobile-first capture interface
docs/       Capture-sheet PDF
vercel.json Vercel service routing
```

## Run Locally

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8765
```

Frontend, in a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`.

## Verify

```bash
cd backend
.venv/bin/python -m pytest -q

cd ../frontend
npm run build
```

Current local backend result: `95 passed`.

## API

- `GET /api/health`
- `POST /api/captures`
- `POST /api/frame-stl`

`/api/captures` accepts a photo plus the selected eye (`left` or `right`) and returns calibration data, measurements, preview overlay data, and contour coordinates.

`/api/frame-stl` accepts the measured left/right contours and CAD parameters, then returns a downloadable ASCII STL.

## Notes

- Photos are processed in memory.
- The intended capture method is loose contact/lens photos only, without the lens inside a frame.
- Real-world accuracy depends on lighting, glare, sharp focus, marker visibility, and the lens edge being visible.
- This is a hackathon prototype, not a certified medical or optical manufacturing tool.
