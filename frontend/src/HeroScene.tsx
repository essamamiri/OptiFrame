import { MutableRefObject, useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

/**
 * Landing-page scene. Scroll drives one story: two loose lenses float on their own,
 * the printed frame rises beneath them, and the lenses settle into its rims.
 * `progress` is 0..1 across the pinned section; the scene eases toward it so the
 * motion has weight instead of snapping to the scrollbar.
 */
export default function HeroScene({ glbUrl, progress }: { glbUrl: string; progress: MutableRefObject<number> }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = host.current;
    if (!container) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    const key = new THREE.DirectionalLight(0xffd2a0, 2.4);
    key.position.set(0.3, 0.5, 0.6);
    const rim = new THREE.DirectionalLight(0xffb066, 1.6);
    rim.position.set(-0.6, 0.2, -0.4);
    scene.add(key, rim, new THREE.HemisphereLight(0xfff1e0, 0x1a120c, 0.5));

    const camera = new THREE.PerspectiveCamera(28, 1, 0.005, 5);

    const frameMaterial = new THREE.MeshPhysicalMaterial({
      color: "#3a2214", roughness: 0.32, clearcoat: 1, clearcoatRoughness: 0.1, transparent: true,
    });
    const glassMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xfff4e6, roughness: 0.03, transmission: 0.95, thickness: 0.002, ior: 1.52,
      transparent: true, opacity: 0.32, side: THREE.DoubleSide, clearcoat: 1,
      sheen: 1, sheenColor: new THREE.Color("#7fd6b0"), sheenRoughness: 0.3,
    });

    const root = new THREE.Group();
    scene.add(root);
    const frameParts: THREE.Object3D[] = [];
    const lenses: { mesh: THREE.Object3D; side: number; home: THREE.Vector3 }[] = [];
    let radius = 0.09;
    let disposed = false;

    new GLTFLoader().load(glbUrl, gltf => {
      if (disposed) return;
      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      model.position.sub(box.getCenter(new THREE.Vector3()));
      radius = box.getSize(new THREE.Vector3()).length() / 2;
      model.traverse(object => {
        if (!(object instanceof THREE.Mesh)) return;
        if (object.name.startsWith("lens")) {
          object.material = glassMaterial;
          lenses.push({ mesh: object, side: object.name.endsWith("left") ? 1 : -1, home: object.position.clone() });
        } else {
          object.material = frameMaterial;
          frameParts.push(object);
        }
      });
      root.add(model);
      resize();
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

    const pointer = new THREE.Vector2();
    const smoothPointer = new THREE.Vector2();
    function onPointer(event: PointerEvent) {
      pointer.set(event.clientX / window.innerWidth - 0.5, event.clientY / window.innerHeight - 0.5);
    }
    window.addEventListener("pointermove", onPointer, { passive: true });

    const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
    const ease = (v: number) => v * v * (3 - 2 * v);
    const span = (p: number, a: number, b: number) => ease(clamp01((p - a) / (b - a)));
    let eased = progress.current;
    const clock = new THREE.Clock();

    renderer.setAnimationLoop(() => {
      const t = clock.getElapsedTime();
      eased += (progress.current - eased) * 0.08;
      smoothPointer.lerp(pointer, 0.05);
      const p = eased;

      // 0 → .45 lenses alone; .35 → .75 frame rises and lenses settle; .75 → 1 the pair turns to face you.
      const rise = span(p, 0.32, 0.72);
      const settle = span(p, 0.42, 0.78);
      const turn = span(p, 0.72, 1);
      const idle = reduceMotion ? 0 : 1;

      for (const lens of lenses) {
        const loose = 1 - settle;
        lens.mesh.position.set(
          lens.home.x + lens.side * 0.012 * loose,
          lens.home.y + loose * (0.01 + Math.sin(t * 1.1 + lens.side) * 0.003 * idle),
          lens.home.z + 0.022 * loose,
        );
        lens.mesh.rotation.set(
          loose * (0.35 + Math.sin(t * 0.8 + lens.side) * 0.08 * idle),
          loose * lens.side * 0.55,
          loose * lens.side * 0.18,
        );
      }
      frameMaterial.opacity = rise;
      frameMaterial.depthWrite = rise > 0.98;
      for (const part of frameParts) part.visible = rise > 0.01;
      root.position.y = 0;
      for (const part of frameParts) part.position.y = (1 - rise) * -0.07;

      // Camera: straight on for the loose lenses, swinging to a three-quarter hero angle.
      const yaw = THREE.MathUtils.lerp(0.0, 0.62, turn) + smoothPointer.x * 0.25;
      const pitch = THREE.MathUtils.lerp(0.08, 0.3, turn) - smoothPointer.y * 0.12;
      const direction = new THREE.Vector3(Math.sin(yaw), pitch, Math.cos(yaw)).normalize();
      const vertical = THREE.MathUtils.degToRad(camera.fov);
      const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * camera.aspect);
      const distance = (radius * THREE.MathUtils.lerp(1.2, 1.1, rise)) / Math.sin(Math.min(vertical, horizontal) / 2);
      camera.position.copy(direction.multiplyScalar(distance));
      camera.lookAt(0, 0, 0);
      root.rotation.y = idle * Math.sin(t * 0.25) * 0.05 * (1 - turn * 0.6);

      renderer.render(scene, camera);
    });

    return () => {
      disposed = true;
      observer.disconnect();
      window.removeEventListener("pointermove", onPointer);
      renderer.setAnimationLoop(null);
      pmrem.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [glbUrl, progress]);

  return <div className="hero-scene" ref={host} aria-hidden="true" />;
}