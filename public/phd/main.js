// main.js
// Reveal bootstrap + multi-slide DualViewer lifecycle

import Reveal from "https://unpkg.com/reveal.js@5/dist/reveal.esm.js";
import { DualViewer } from "./dualViewer.js";

const deck = new Reveal({
  hash: true,
  width: 1280,
  height: 720,
  center: true
});

await deck.initialize();

// Maintain one DualViewer per slide element
const viewers = new Map(); // key: slideElement, value: { viewer, container }

function setupIfNeeded(slideEl) {
  if (viewers.has(slideEl)) return viewers.get(slideEl);

  const container = slideEl.querySelector(".dual-viewer");
  if (!container) return null;

  // Read data-* attributes
  const rawUrl = slideEl.dataset.raw || "raw.ply";
  const pcaUrl = slideEl.dataset.pca || "pca.ply";
  const pointSize = slideEl.dataset.pointSize ? parseFloat(slideEl.dataset.pointSize) : undefined;
  const background = slideEl.dataset.background || undefined;

  const viewer = new DualViewer(container, {
    rawUrl, pcaUrl, pointSize, background
  });

  viewers.set(slideEl, { viewer, container });
  return { viewer, container };
}

function onEnter(slideEl) {
  const v = setupIfNeeded(slideEl);
  if (v) v.viewer.start();
}

function onLeave(slideEl) {
  const v = viewers.get(slideEl);
  if (v) v.viewer.stop();
}

// Initial
onEnter(deck.getCurrentSlide());

// Events
deck.on("slidechanged", (e) => {
  if (e.previousSlide) onLeave(e.previousSlide);
  if (e.currentSlide) onEnter(e.currentSlide);
});

deck.on("resize", () => {
  const current = deck.getCurrentSlide();
  const v = current ? viewers.get(current) : null;
  if (v) v.viewer.resize();
});

// Optional: clean up if you dynamically remove slides
// function disposeSlide(slideEl) {
//   const v = viewers.get(slideEl);
//   if (v) { v.viewer.dispose(); viewers.delete(slideEl); }
// }
