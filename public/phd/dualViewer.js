// dualViewer.js
// Reusable dual point-cloud viewer with synchronized cameras (Three.js).
// No dependency on Reveal.js. Pure "container + options" API.

// External deps loaded by the caller page as ESM:
//   - three
//   - OrbitControls
//   - PLYLoader
import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js";

export class DualViewer {
  /**
   * @param {HTMLElement} container - Element containing two <canvas> with IDs 'canvasRaw' & 'canvasPCA'
   * @param {{
   *   rawUrl: string,
   *   pcaUrl: string,
   *   pointSize?: number,
   *   background?: string | number,
   *   ctrlPan?: boolean,
   *   fitOnLeftLoad?: boolean,
   *   cameraInit?: { pos?: [number,number,number], fov?: number, near?: number, far?: number }
   * }} opts
   */
  constructor(container, opts) {
    this.container = container;
    this.opts = Object.assign({
      pointSize: 0.01,
      background: 0xffffff,
      ctrlPan: true,
      fitOnLeftLoad: true,
      cameraInit: { pos: [2.5, 2.0, 2.5], fov: 60, near: 0.01, far: 2000 }
    }, opts || {});
    this.running = false;

    // Grab canvases
    this.canvasA = container.querySelector("#canvasRaw");
    this.canvasB = container.querySelector("#canvasPCA");
    if (!this.canvasA || !this.canvasB) throw new Error("Missing #canvasRaw or #canvasPCA");

    // Renderers
    this.rendererA = new THREE.WebGLRenderer({ canvas: this.canvasA, antialias: true });
    this.rendererB = new THREE.WebGLRenderer({ canvas: this.canvasB, antialias: true });
    this.rendererA.setPixelRatio(devicePixelRatio);
    this.rendererB.setPixelRatio(devicePixelRatio);

    // Scenes
    this.sceneA = new THREE.Scene();
    this.sceneB = new THREE.Scene();
    const bg = this.opts.background;
    this.sceneA.background = new THREE.Color(bg);
    this.sceneB.background = new THREE.Color(bg);

    // Cameras
    const { fov, near, far, pos } = this.opts.cameraInit;
    this.cameraA = new THREE.PerspectiveCamera(fov, 1, near, far);
    this.cameraB = new THREE.PerspectiveCamera(fov, 1, near, far);
    this.cameraA.position.set(...pos);
    this.cameraB.position.copy(this.cameraA.position);

    // Controls (only on left; we mirror to right)
    this.controlsA = new OrbitControls(this.cameraA, this.rendererA.domElement);
    this.controlsA.enableDamping = true;
    this.controlsA.enablePan = true;

    if (this.opts.ctrlPan) {
      this._onPointerMove = (e) => { this.controlsA.enablePan = e.ctrlKey; };
      this.rendererA.domElement.addEventListener("pointermove", this._onPointerMove);
    }

    // Lights (simple)
    this.sceneA.add(new THREE.AmbientLight(0xffffff, 1.0));
    this.sceneB.add(new THREE.AmbientLight(0xffffff, 1.0));

    // Sync A → B
    this._syncCamera = () => {
      this.cameraB.position.copy(this.cameraA.position);
      this.cameraB.quaternion.copy(this.cameraA.quaternion);
      this.cameraB.zoom = this.cameraA.zoom;
      this.cameraB.projectionMatrix.copy(this.cameraA.projectionMatrix);
      this.cameraB.updateMatrixWorld(true);
    };
    this.controlsA.addEventListener("change", this._syncCamera);
    this._syncCamera();

    // Load clouds
    this.loader = new PLYLoader();
    this._addCloud(this.opts.rawUrl, this.sceneA, this.opts.fitOnLeftLoad);
    this._addCloud(this.opts.pcaUrl, this.sceneB, false);

    // Initial resize
    this.resize();
  }

  // Load one PLY and add to scene
  _addCloud(url, scene, fitAfterLoad) {
    this.loader.load(url, (geom) => {
      geom.computeBoundingBox();
      const mat = new THREE.PointsMaterial({
        size: this.opts.pointSize,
        vertexColors: true,
        sizeAttenuation: true
      });
      const pts = new THREE.Points(geom, mat);
      scene.add(pts);

      if (fitAfterLoad) {
        const bb = geom.boundingBox;
        const center = new THREE.Vector3().addVectors(bb.min, bb.max).multiplyScalar(0.5);
        this.controlsA.target.copy(center);
        this.cameraA.lookAt(center);
        // Optionally adjust distance to fit
        const radius = 0.5 * bb.getSize(new THREE.Vector3()).length();
        const fovRad = this.cameraA.fov * Math.PI / 180;
        const dist = radius / Math.tan(fovRad / 2);
        const dir = new THREE.Vector3().subVectors(this.cameraA.position, center).normalize();
        this.cameraA.position.copy(center.clone().addScaledVector(dir, dist * 1.2));
        this._syncCamera();
      }
    });
  }

  // Public API
  start() {
    if (this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      this.controlsA.update();
      this.rendererA.render(this.sceneA, this.cameraA);
      this.rendererB.render(this.sceneB, this.cameraB);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  stop() {
    this.running = false;
  }

  resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    const wHalf = Math.max(1, Math.floor(w / 2));
    this.rendererA.setSize(wHalf, h, false);
    this.rendererB.setSize(wHalf, h, false);
    this.cameraA.aspect = wHalf / h;
    this.cameraB.aspect = wHalf / h;
    this.cameraA.updateProjectionMatrix();
    this.cameraB.updateProjectionMatrix();
  }

  dispose() {
    this.stop();
    this.controlsA.removeEventListener("change", this._syncCamera);
    if (this._onPointerMove) {
      this.rendererA.domElement.removeEventListener("pointermove", this._onPointerMove);
    }
    [this.rendererA, this.rendererB].forEach(r => r.dispose());
    // Minimal GC safety for materials/geometries
    [this.sceneA, this.sceneB].forEach(scene => {
      scene.traverse(obj => {
        if (obj.geometry) obj.geometry.dispose?.();
        if (obj.material) obj.material.dispose?.();
      });
    });
  }
}
