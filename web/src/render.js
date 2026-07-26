// three.js rendering driven entirely by the compiled MuJoCo model.
//
// Geometry is read out of the model — types, sizes, mesh vertices and faces — and
// posed each frame from `data.geom_xpos` and `data.geom_xmat`. Nothing is
// re-authored, so what is on screen is the scene the policy was trained in.

import * as THREE from '../vendor/three/three.module.js';
import { at } from './sim.js';

// mjtGeom values used by this scene.
const GEOM_PLANE = 0;
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;
const GEOM_MESH = 7;

function meshGeometry(model, dataId) {
  const vertAdr = at(model.mesh_vertadr, dataId);
  const vertNum = at(model.mesh_vertnum, dataId);
  const faceAdr = at(model.mesh_faceadr, dataId);
  const faceNum = at(model.mesh_facenum, dataId);

  const positions = new Float32Array(vertNum * 3);
  for (let i = 0; i < vertNum * 3; i++) positions[i] = at(model.mesh_vert, vertAdr * 3 + i);

  const indices = new Uint32Array(faceNum * 3);
  for (let i = 0; i < faceNum * 3; i++) indices[i] = at(model.mesh_face, faceAdr * 3 + i);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

/**
 * A checkerboard for the ground plane.
 *
 * Not decoration. The camera tracks the torso and MuJoCo's floor is a featureless
 * infinite plane, so a robot walking at 1 m/s renders as a robot standing still —
 * nothing in frame moves relative to it. The texture is what makes the gait read
 * as locomotion.
 */
function floorTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#1b222d';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#232c3a';
  ctx.fillRect(0, 0, size / 2, size / 2);
  ctx.fillRect(size / 2, size / 2, size / 2, size / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  // One tile per 0.5 m of a 50 m plane.
  texture.repeat.set(100, 100);
  texture.anisotropy = 8;
  return texture;
}

function primitiveGeometry(type, sx, sy, sz) {
  switch (type) {
    case GEOM_PLANE:
      // MuJoCo planes are infinite; size is a rendering hint. 0 means "large".
      return new THREE.PlaneGeometry(2 * (sx || 25), 2 * (sy || 25));
    case GEOM_SPHERE:
      return new THREE.SphereGeometry(sx, 16, 12);
    case GEOM_CAPSULE:
      return new THREE.CapsuleGeometry(sx, 2 * sy, 8, 8);
    case GEOM_ELLIPSOID: {
      const g = new THREE.SphereGeometry(1, 16, 12);
      g.scale(sx, sy, sz);
      return g;
    }
    case GEOM_CYLINDER:
      return new THREE.CylinderGeometry(sx, sx, 2 * sy, 16);
    case GEOM_BOX:
      return new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
    default:
      return null;
  }
}

export class Viewport {
  constructor(canvas, sim, label) {
    this.sim = sim;
    this.label = label;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x11151c);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    this.camera.up.set(0, 0, 1); // MuJoCo is z-up
    this.camera.position.set(1.05, -1.35, 0.78);

    this.scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x30302f, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(2, -3, 5);
    this.scene.add(sun);

    this.#buildGeoms();
    this.resize();
  }

  #buildGeoms() {
    const model = this.sim.model;
    this.meshes = [];

    for (let g = 0; g < model.ngeom; g++) {
      const type = at(model.geom_type, g);
      const sx = at(model.geom_size, g * 3 + 0);
      const sy = at(model.geom_size, g * 3 + 1);
      const sz = at(model.geom_size, g * 3 + 2);

      let geometry = null;
      if (type === GEOM_MESH) {
        const dataId = at(model.geom_dataid, g);
        if (dataId >= 0) geometry = meshGeometry(model, dataId);
      } else {
        geometry = primitiveGeometry(type, sx, sy, sz);
      }
      if (geometry === null) {
        this.meshes.push(null);
        continue;
      }

      const r = at(model.geom_rgba, g * 4 + 0);
      const gg = at(model.geom_rgba, g * 4 + 1);
      const b = at(model.geom_rgba, g * 4 + 2);
      const a = at(model.geom_rgba, g * 4 + 3);

      const material =
        type === GEOM_PLANE
          ? new THREE.MeshStandardMaterial({
              map: floorTexture(),
              roughness: 0.95,
              metalness: 0.0,
            })
          : new THREE.MeshStandardMaterial({
              color: new THREE.Color(r, gg, b),
              transparent: a < 1,
              opacity: a,
              roughness: 0.6,
              metalness: 0.15,
            });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.matrixAutoUpdate = false;
      this.scene.add(mesh);
      this.meshes.push(mesh);
    }
  }

  /** Pose every geom from the current simulation state. */
  sync() {
    const d = this.sim.data;
    const m = new THREE.Matrix4();
    for (let g = 0; g < this.meshes.length; g++) {
      const mesh = this.meshes[g];
      if (!mesh) continue;
      const px = at(d.geom_xpos, g * 3 + 0);
      const py = at(d.geom_xpos, g * 3 + 1);
      const pz = at(d.geom_xpos, g * 3 + 2);
      // geom_xmat is row-major 3x3; three.js `set` takes row-major too.
      const r = (i) => at(d.geom_xmat, g * 9 + i);
      m.set(
        r(0), r(1), r(2), px,
        r(3), r(4), r(5), py,
        r(6), r(7), r(8), pz,
        0, 0, 0, 1
      );
      mesh.matrix.copy(m);
    }

    // Follow the torso so the robot does not walk out of frame.
    const tx = at(d.xpos, 3);
    const ty = at(d.xpos, 4);
    const tz = at(d.xpos, 5);
    this.camera.position.set(tx + 1.05, ty - 1.35, tz + 0.55);
    this.camera.lookAt(tx, ty, tz - 0.05);
  }

  resize() {
    const canvas = this.renderer.domElement;
    const w = canvas.clientWidth || 320;
    const h = canvas.clientHeight || 240;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  draw() {
    this.renderer.render(this.scene, this.camera);
  }
}
