import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Match, RackPoint } from "../types";

function makeGlowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (!context) return new THREE.Texture();

  const gradient = context.createRadialGradient(64, 64, 3, 64, 64, 62);
  gradient.addColorStop(0, "rgba(255,255,255,0.95)");
  gradient.addColorStop(0.22, "rgba(255,255,255,0.48)");
  gradient.addColorStop(0.58, "rgba(255,255,255,0.14)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function indexFromId(id: string) {
  return Array.from(id).reduce((total, char) => total + char.charCodeAt(0), 0);
}

export function VectorSpace({
  points,
  queryPoint,
  matches,
  selectedId,
  onSelect,
}: {
  points: RackPoint[];
  queryPoint: RackPoint | null;
  matches: Match[];
  selectedId: string | null;
  onSelect: (point: RackPoint | null) => void;
}) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const pointMapRef = useRef<Map<string, RackPoint>>(new Map());
  const selectedIdRef = useRef<string | null>(selectedId);

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    if (!mountRef.current) return;

    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x070806, 9, 26);

    const camera = new THREE.PerspectiveCamera(48, mount.clientWidth / mount.clientHeight, 0.1, 100);
    camera.position.set(0, 1.9, 9.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const frameGroup = new THREE.Group();
    const pointGroup = new THREE.Group();
    scene.add(frameGroup);
    scene.add(pointGroup);

    const grid = new THREE.GridHelper(10, 20, 0x3b82f6, 0x303126);
    grid.position.y = -2.8;
    grid.material.opacity = 0.28;
    grid.material.transparent = true;
    frameGroup.add(grid);

    const axes = [
      [new THREE.Vector3(-4.8, -2.6, -2.8), new THREE.Vector3(4.8, -2.6, -2.8), 0x3b82f6],
      [new THREE.Vector3(-4.8, -2.6, -2.8), new THREE.Vector3(-4.8, 2.8, -2.8), 0x9fc9a2],
      [new THREE.Vector3(-4.8, -2.6, -2.8), new THREE.Vector3(-4.8, -2.6, 2.8), 0x9fbbe0],
    ] as const;
    axes.forEach(([start, end, color]) => {
      const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
      const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 });
      frameGroup.add(new THREE.Line(geometry, material));
    });

    const backdropGeometry = new THREE.BufferGeometry();
    const backdropPositions = new Float32Array(150 * 3);
    for (let index = 0; index < 150; index += 1) {
      backdropPositions[index * 3] = (Math.random() - 0.5) * 12;
      backdropPositions[index * 3 + 1] = (Math.random() - 0.5) * 7;
      backdropPositions[index * 3 + 2] = (Math.random() - 0.5) * 9;
    }
    backdropGeometry.setAttribute("position", new THREE.BufferAttribute(backdropPositions, 3));
    const backdrop = new THREE.Points(
      backdropGeometry,
      new THREE.PointsMaterial({ color: 0xf7f7f4, size: 0.012, transparent: true, opacity: 0.34 })
    );
    frameGroup.add(backdrop);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const pointerTarget = new THREE.Vector2();
    const meshes: THREE.Mesh[] = [];
    const halos: THREE.Sprite[] = [];
    const objectsById = new Map<string, { halo?: THREE.Sprite; base: THREE.Vector3; orbit: number; phase: number; speed: number }>();
    const matchedIds = new Set(matches.map((match) => match.source_id));
    const glowTexture = makeGlowTexture();
    const allPoints = queryPoint ? [...points, queryPoint] : points;
    pointMapRef.current = new Map(allPoints.map((point) => [point.id, point]));

    allPoints.forEach((point) => {
      const position = new THREE.Vector3(point.projection.x * 1.35, point.projection.y * 1.35, point.projection.z * 1.35);
      const isQuery = point.modality === "query";
      const isMatched = matchedIds.has(point.source_id);
      if (isQuery || isMatched) {
        const haloMaterial = new THREE.SpriteMaterial({
          map: glowTexture,
          color: new THREE.Color(isQuery ? "#3b82f6" : point.color),
          transparent: true,
          opacity: isQuery ? 0.32 : 0.24,
          depthWrite: false,
        });
        const halo = new THREE.Sprite(haloMaterial);
        halo.position.copy(position);
        halo.scale.setScalar(isQuery ? 0.72 : 0.58);
        halo.userData.baseScale = isQuery ? 0.72 : 0.58;
        halo.userData.baseOpacity = isQuery ? 0.32 : 0.24;
        halo.userData.id = point.id;
        halos.push(halo);
        pointGroup.add(halo);
      }

      const geometry = new THREE.SphereGeometry(0.08, 24, 24);
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(point.color),
        transparent: true,
        opacity: point.modality === "query" ? 1 : 0.9,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(position);
      mesh.userData.id = point.id;
      meshes.push(mesh);
      pointGroup.add(mesh);
      objectsById.set(point.id, {
        halo: halos.find((item) => item.userData.id === point.id),
        base: position,
        orbit: isQuery ? 0.028 : 0.075 + (indexFromId(point.id) % 5) * 0.012,
        phase: (indexFromId(point.id) % 13) * 0.62,
        speed: isQuery ? 0.28 : 0.34 + (indexFromId(point.id) % 7) * 0.035,
      });
    });

    const handlePointer = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      pointerTarget.set(pointer.x, pointer.y);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(meshes)[0];
      renderer.domElement.style.cursor = hit ? "pointer" : "default";
      onSelect(hit ? pointMapRef.current.get(hit.object.userData.id) ?? null : null);
    };

    const handlePointerLeave = () => {
      pointerTarget.set(0, 0);
      renderer.domElement.style.cursor = "default";
      onSelect(null);
    };

    renderer.domElement.addEventListener("pointermove", handlePointer);
    renderer.domElement.addEventListener("pointerleave", handlePointerLeave);

    const resize = () => {
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", resize);

    let frame = 0;
    let animation = 0;
    const animate = () => {
      frame += 0.01;
      camera.position.x += (pointerTarget.x * 0.18 - camera.position.x) * 0.025;
      camera.position.y += (1.9 + pointerTarget.y * 0.1 - camera.position.y) * 0.025;
      camera.lookAt(0, 0, 0);
      meshes.forEach((mesh, index) => {
        const object = objectsById.get(mesh.userData.id);
        if (object) {
          const theta = frame * object.speed + object.phase;
          const bob = Math.sin(frame * object.speed * 1.7 + object.phase) * object.orbit * 0.48;
          mesh.position.set(
            object.base.x + Math.cos(theta) * object.orbit,
            object.base.y + bob,
            object.base.z + Math.sin(theta) * object.orbit
          );
          mesh.rotation.y += 0.012 + index * 0.0004;
          mesh.rotation.x += 0.006;
          object.halo?.position.copy(mesh.position);
        }
        const pulse = 1 + Math.sin(frame * 2.2 + index) * 0.055;
        mesh.scale.setScalar(mesh.userData.id === selectedIdRef.current ? 1.22 : pulse);
      });
      halos.forEach((halo, index) => {
        const base = halo.userData.baseScale || 0.58;
        const pulse = 1 + Math.sin(frame * 1.7 + index) * 0.08;
        halo.scale.setScalar(base * pulse);
        const material = halo.material as THREE.SpriteMaterial;
        material.opacity = halo.userData.id === selectedIdRef.current ? 0.46 : halo.userData.baseOpacity;
      });
      renderer.render(scene, camera);
      animation = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(animation);
      window.removeEventListener("resize", resize);
      renderer.domElement.removeEventListener("pointermove", handlePointer);
      renderer.domElement.removeEventListener("pointerleave", handlePointerLeave);
      mount.removeChild(renderer.domElement);
      glowTexture.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
          else object.material.dispose();
        }
        if (object instanceof THREE.Sprite) {
          object.material.dispose();
        }
      });
      renderer.dispose();
    };
  }, [points, queryPoint, matches, onSelect]);

  return <div className="vector-canvas" ref={mountRef} />;
}
