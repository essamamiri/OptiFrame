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
    // Clear glass: almost no body colour, strong reflections, and a drawn edge so the
    // lens reads as a real cut piece with thickness rather than a frosted blob.
    const glassMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xf6fbfc, roughness: 0.02, metalness: 0, ior: 1.5,
      transparent: true, opacity: 0.24, depthWrite: false, side: THREE.DoubleSide,
      clearcoat: 1, clearcoatRoughness: 0, envMapIntensity: 3, specularIntensity: 1,
      iridescence: 0.6, iridescenceIOR: 1.3, iridescenceThicknessRange: [180, 420],
    });
    const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x7a8386, transparent: true, opacity: 0.9 });
    const bevelMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });

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
          object.renderOrder = 2;
          // Front and back outlines of the lens: a darker cut edge, plus a bright bevel just inside it.
          const edges = new THREE.LineSegments(new THREE.EdgesGeometry(object.geometry, 30), edgeMaterial);
          const bevel = new THREE.LineSegments(new THREE.EdgesGeometry(object.geometry, 30), bevelMaterial);
          object.geometry.computeBoundingBox();
          const centre = object.geometry.boundingBox!.getCenter(new THREE.Vector3());
          bevel.position.copy(centre).multiplyScalar(0.03);
          bevel.scale.setScalar(0.97);
          object.add(edges, bevel);
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

      // Story beats:
      //   0 → .32   the two lenses float on their own, tilted
      //   .30 → .58 the frame rises up behind them
      //   .42 → .62 the lenses straighten and line up in front of their rims
      //   .62 → .74 they push straight back and click into place (small overshoot)
      //   .76 → 1   the finished pair turns to a three-quarter view
      const rise = span(p, 0.3, 0.58);
      const align = span(p, 0.42, 0.62);
      const push = clamp01((p - 0.62) / 0.12);
      const turn = span(p, 0.76, 1);
      const idle = reduceMotion ? 0 : 1;
      // easeOutBack: goes a hair past the seat, then settles — the "click".
      const back = (v: number) => { const c = 1.2; const u = v - 1; return 1 + (c + 1) * u * u * u + c * u * u; };
      const seated = push <= 0 ? 0 : back(push);

      for (const lens of lenses) {
        const free = 1 - align;
        lens.mesh.position.set(
          lens.home.x + lens.side * 0.014 * free,
          lens.home.y + free * (0.008 + Math.sin(t * 1.1 + lens.side) * 0.003 * idle),
          lens.home.z + 0.034 * (1 - seated),
        );
        lens.mesh.rotation.set(
          free * (0.3 + Math.sin(t * 0.8 + lens.side) * 0.07 * idle),
          free * lens.side * 0.5,
          free * lens.side * 0.14,
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
      const distance = (radius * THREE.MathUtils.lerp(1.2, 1.08, rise)) / Math.sin(Math.min(vertical, horizontal) / 2);
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