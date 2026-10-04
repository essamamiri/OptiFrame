# Frontend update

## What changed
- Dark precision-scanner theme, responsive camera and upload controls.
- Independent photo, measurement receipt, and capture mode state for each eye; switching eyes preserves each eye's work.
- Scan animation runs only while the existing measurement request is active, with reduced-motion support.
- Actual-size SVG download from the measured millimetre contour.
- Progress reflects measured lenses rather than implying completed CAD/export features.
- Existing backend API and all calibration/error displays preserved.

## Apply to the existing Vercel project
Copy frontend/src/App.tsx, frontend/src/styles.css, frontend/index.html and frontend/public/favicon.svg into your existing checkout. Review and commit those files, then push to the branch connected to Vercel. Keep backend files and deployment configuration unchanged. No new dependencies were added.

## Verify
Run npm ci and npm run build inside frontend. Build passed (TypeScript and Vite). Browser interaction QA could not run because a Chromium executable was unavailable in this environment. On your phone verify that left/right images remain separate, retaking only resets that lens, measurement errors display, and results persist when changing eyes. Test the SVG at 100% print scale.

## Important
The supplied frontend/backend uses card calibration with a standard 85.60 x 53.98 mm card. The dataset's ArUco work does not make the current capture endpoint accept marker-only photos. Coordinate reference handling with the measurement teammate.
The provided application still has no frame-generation endpoint or 3D/STL UI. Those functions were not invented in this redesign. Measurement accuracy and physical fitting still require validation.
