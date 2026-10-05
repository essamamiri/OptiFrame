import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

type Finish = { label: string; color: string; roughness: number; opacity: number; clearcoat: number };

const FINISHES: Finish[] = [
  { label: "Onyx", color: "#16110f", roughness: 0.28, opacity: 1, clearcoat: 1 },
  { label: "Havana", color: "#2e1a10", roughness: 0.3, opacity: 1, clearcoat: 1 },
  { label: "Crystal", color: "#d9e6ea", roughness: 0.08, opacity: 0.55, clearcoat: 1 },
  { label: "Sage", color: "#7fa894", roughness: 0.45, opacity: 1, clearcoat: 0.4 },
];

const VIEWS = {
  "3/4": new THREE.Vector3(0.62, 0.32, 1),
  Front: new THREE.Vector3(0, 0.04, 1),
  Side: new THREE.Vector3(1, 0.12, -0.35),
  Top: new THREE.Vector3(0.05, 1, 0.35),
} as const;
type View = keyof typeof VIEWS;

export default function FrameViewer({ glbUrl, caption }: { glbUrl: string; caption: string }) {
  const host = useRef<HTMLDivElement>(null);
  const api = useRef<{ setFinish: (f: Finish) => void; setView: (v: View) => void; snapshot: () => string } | null>(null);
  const [finish, setFinish] = useState(0);
  const [view, setView] = useState<View>("3/4");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const container = host.current;
    if (!container) return;
    setReady(false);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    const camera = new THREE.PerspectiveCamera(30, 1, 0.005, 5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.minDistance = 0.12;
    controls.maxDistance = 1.5;
    controls.autoRotate = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    controls.autoRotateSpeed = 0.6;
    controls.addEventListener("start", () => { controls.autoRotate = false; });

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(0.25, 0.6, 0.35);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 6;
    Object.assign(key.shadow.camera, { left: -0.2, right: 0.2, top: 0.2, bottom: -0.2, near: 0.01, far: 2 });
    scene.add(key, new THREE.HemisphereLight(0xdff1ff, 0x1a2230, 0.6));

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShadowMaterial({ opacity: 0.32 }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    const frameMaterial = new THREE.MeshPhysicalMaterial({ roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.12 });
    const glassMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xeaf6ff, roughness: 0.02, transmission: 0.92, thickness: 0.002, ior: 1.5,
      transparent: true, opacity: 0.35, side: THREE.DoubleSide,
    });
    let target = new THREE.Vector3();
    let radius = 0.1;

    function applyFinish(f: Finish) {
      frameMaterial.color.set(f.color);
      frameMaterial.roughness = f.roughness;
      frameMaterial.clearcoat = f.clearcoat;
      frameMaterial.transparent = f.opacity < 1;
      frameMaterial.opacity = f.opacity;
      frameMaterial.transmission = f.opacity < 1 ? 0.6 : 0;
      frameMaterial.needsUpdate = true;
    }
    function applyView(v: View) {
      const direction = VIEWS[v].clone().normalize();
      // Fit the model's bounding sphere to the narrower field of view so it
      // never crops on a portrait phone screen.
      const vertical = THREE.MathUtils.degToRad(camera.fov);
      const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * camera.aspect);
      const distance = (radius * 1.08) / Math.sin(Math.min(vertical, horizontal) / 2);
      camera.position.copy(target).addScaledVector(direction, distance);
      controls.target.copy(target);
      controls.update();
    }

    let disposed = false;
    new GLTFLoader().load(glbUrl, gltf => {
      if (disposed) return;
      const model = gltf.scene;
      model.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        const isLens = object.name.startsWith("lens");
        object.material = isLens ? glassMaterial : frameMaterial;
        object.castShadow = !isLens;
      });
      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      model.position.sub(center);
      model.position.y += (box.max.y - box.min.y) / 2 + 0.012;
      scene.add(model);
      target = new THREE.Vector3(0, (box.max.y - box.min.y) / 2 + 0.012, 0);
      radius = box.getSize(new THREE.Vector3()).length() / 2;
      applyView("3/4");
      setReady(true);
    });

    function resize() {
      const { clientWidth, clientHeight } = container!;
      renderer.setSize(clientWidth, clientHeight, false);
      camera.aspect = clientWidth / Math.max(clientHeight, 1);
      camera.updateProjectionMatrix();
    }
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

    api.current = {
      setFinish: applyFinish,
      setView: applyView,
      snapshot: () => {
        const previous = renderer.getPixelRatio();
        renderer.setPixelRatio(3);
        resize();
        renderer.render(scene, camera);
        const url = renderer.domElement.toDataURL("image/png");
        renderer.setPixelRatio(previous);
        resize();
        return url;
      },
    };
    applyFinish(FINISHES[0]);

    return () => {
      disposed = true;
      observer.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      pmrem.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      api.current = null;
    };
  }, [glbUrl]);

  useEffect(() => { api.current?.setFinish(FINISHES[finish]); }, [finish, ready]);

  function chooseView(v: View) { setView(v); api.current?.setView(v); }
  function saveRender() {
    const url = api.current?.snapshot();
    if (!url) return;
    const link = document.createElement("a");
    link.href = url;
    link.download = `optiframe-render-${view.replace("/", "-").toLowerCase()}.png`;
    link.click();
  }

  return (
    <div className="frame-viewer">
      <div className="viewer-stage" ref={host} aria-label="Interactive 3D preview of the generated frame">
        {!ready && <span className="viewer-loading">Building 3D preview…</span>}
        <span className="viewer-caption">{caption}</span>
      </div>
      <div className="viewer-toolbar">
        <div className="viewer-chips" role="group" aria-label="Camera view">
          {(Object.keys(VIEWS) as View[]).map(v => (
            <button key={v} type="button" className={v === view ? "active" : ""} aria-pressed={v === view} onClick={() => chooseView(v)}>{v}</button>
          ))}
        </div>
        <div className="viewer-chips" role="group" aria-label="Preview finish">
          {FINISHES.map((f, index) => (
            <button key={f.label} type="button" className={index === finish ? "active" : ""} aria-pressed={index === finish} onClick={() => setFinish(index)}>
              <i style={{ background: f.color }} />{f.label}
            </button>
          ))}
        </div>
        <button type="button" className="viewer-save" onClick={saveRender} disabled={!ready}>Save render</button>
      </div>
    </div>
  );
}
