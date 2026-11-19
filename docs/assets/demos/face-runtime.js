// Finding, aligning, and recognizing a face on the reader's device. It loads ONNX Runtime Web and the
// study's two OpenCV Zoo models only when a demo asks for them, checks the SHA-256 of every file it
// runs (the runtime's three files against pinned values, the two models against the study's) before
// using it, and never sends an image anywhere. The pure math (the YuNet decoder, the five-point
// alignment, SFace's input, and the cosine score) is in face-engine.js.
import * as engine from "./face-engine.js?v=58cf75d963";

const ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
// onnxruntime-web 1.31.1's browser files, with the SHA-256 of each as published.
const ORT_FILES = {
  main: { file: "ort.wasm.min.mjs", sha256: "c21d9c326d453a4a61398270c804d5979e118540597cb0c07f7b3d64e6098234", type: "text/javascript" },
  glue: { file: "ort-wasm-simd-threaded.mjs", sha256: "30ed0b5cd1182ede931fea448afb567d4dc705ec957af09f108a2be5f770dac3", type: "text/javascript" },
  wasm: { file: "ort-wasm-simd-threaded.wasm", sha256: "4b884c203b0fa48d057be3332c65513ff85aa3c5bddd6946acc0fab7ecda001d", type: "application/wasm" },
};
const CROP = 112;

let ortPromise = null;
const sessions = new Map();

const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

/** Fetch a file with progress, and refuse it unless its SHA-256 is the expected one. */
async function fetchChecked(url, sha256, name, onProgress, bytes) {
  const res = await fetch(url, { mode: "cors", credentials: "omit" });
  if (!res.ok) throw new Error(`${name}: the download failed (${res.status})`);
  const total = Number(res.headers.get("content-length")) || bytes || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    if (onProgress) onProgress(got, total);
  }
  const buf = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { buf.set(p, at); at += p.length; }
  if (hex(await crypto.subtle.digest("SHA-256", buf)) !== sha256) throw new Error(`${name}: its SHA-256 doesn't match, so it wasn't used`);
  return buf;
}

/** ONNX Runtime Web, run only from bytes whose hashes match; a failed load can be retried. */
function ort() {
  if (!ortPromise) {
    ortPromise = (async () => {
      const [main, glue, wasm] = await Promise.all(Object.values(ORT_FILES).map((f) => fetchChecked(ORT_BASE + f.file, f.sha256, f.file)));
      const url = (bytes, type) => URL.createObjectURL(new Blob([bytes], { type }));
      const m = await import(url(main, ORT_FILES.main.type));
      const o = m.default || m;
      o.env.wasm.wasmPaths = { mjs: url(glue, ORT_FILES.glue.type), wasm: url(wasm, ORT_FILES.wasm.type) };
      o.env.wasm.numThreads = 1;
      o.env.logLevel = "error";  // SFace's file lists its weights as graph inputs, which ONNX Runtime warns about
      return o;
    })().catch((e) => { ortPromise = null; throw e; });
  }
  return ortPromise;
}

async function session(info, onProgress) {
  if (!sessions.has(info.file)) {
    sessions.set(info.file, (async () => {
      const [o, bytes] = await Promise.all([ort(), fetchChecked(info.url, info.sha256, info.file, onProgress, info.bytes)]);
      return { o, s: await o.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all", logSeverityLevel: 3 }) };
    })().catch((e) => { sessions.delete(info.file); throw e; }));
  }
  return sessions.get(info.file);
}

const dims = (src) => [src.videoWidth || src.naturalWidth || src.width, src.videoHeight || src.naturalHeight || src.height];

/**
 * One still frame of the source (a video frame, photo, or bitmap), so detection and alignment use the
 * same pixels; a very large photo is scaled to 4,096 pixels on its long side, which still leaves a face
 * far more pixels than the 112-pixel crop needs. k is that scale.
 */
function snapshot(src) {
  const [w, h] = dims(src);
  const k = Math.min(1, 4096 / Math.max(w, h));
  const c = new OffscreenCanvas(Math.max(1, Math.round(w * k)), Math.max(1, Math.round(h * k)));
  const g = c.getContext("2d"); g.imageSmoothingQuality = "high";
  g.drawImage(src, 0, 0, c.width, c.height);
  c.k = [c.width / w, c.height / h];
  return c;
}

/**
 * Pixels of a region of a canvas, scaled by about k: { rgba, W, H, kx, ky }, where kx and ky are the
 * scales the rounded size actually gives on each axis.
 */
function region(canvas, x0, y0, rw, rh, k) {
  const W = Math.max(1, Math.round(rw * k)), H = Math.max(1, Math.round(rh * k));
  const c = new OffscreenCanvas(W, H);
  const g = c.getContext("2d", { willReadFrequently: true });
  g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
  g.drawImage(canvas, x0, y0, rw, rh, 0, 0, W, H);
  return { rgba: g.getImageData(0, 0, W, H).data, W, H, kx: W / rw, ky: H / rh };
}

/**
 * The face finder: YuNet with the study's settings. alignedFace(source) takes one frame, finds the
 * highest-scoring face, as the study did, and returns its 112-pixel aligned crop (RGBA), its five
 * landmarks in the source's pixels, and whether they were found in this frame or held from an earlier one.
 */
export async function faceFinder(det, onProgress) {
  const { o, s } = await session(det, onProgress);
  const opts = { scoreThreshold: det.score_threshold, nmsThreshold: det.nms, topK: 5000 };
  /** Detection in the frame's pixels (pass a snapshot) or, for any other source, in its own pixels. */
  async function detect(src) {
    if (!(src instanceof OffscreenCanvas)) {
      const frame = snapshot(src), [sx, sy] = frame.k, d = await detect(frame);
      return d && { box: [d.box[0] / sx, d.box[1] / sy, d.box[2] / sx, d.box[3] / sy], landmarks: d.landmarks.map(([x, y]) => [x / sx, y / sy]), score: d.score };
    }
    const frame = src;
    const [w, h] = [frame.width, frame.height];
    // The model takes a fixed 640 × 640 input: the frame, scaled so its long side is at most 640, sits
    // at the top left and the rest is zero, as OpenCV pads an image to the network's size.
    const { rgba, W, H, kx, ky } = region(frame, 0, 0, w, h, Math.min(1, 640 / Math.max(w, h)));
    const inp = engine.yunetInput(rgba, W, H, 640, 640);
    const out = await s.run({ [s.inputNames[0]]: new o.Tensor("float32", inp.data, [1, 3, inp.height, inp.width]) });
    const faces = engine.yunetDecode(Object.fromEntries(Object.entries(out).map(([n, t]) => [n, t.data])), inp.width, inp.height, opts);
    if (!faces.length) return null;
    const best = engine.bestFace(faces);
    return { box: [best[0] / kx, best[1] / ky, best[2] / kx, best[3] / ky], landmarks: [0, 1, 2, 3, 4].map((i) => [best[4 + 2 * i] / kx, best[5 + 2 * i] / ky]), score: best[14] };
  }
  async function alignedFace(src, { hold = null } = {}) {
    const frame = snapshot(src);
    const [sx, sy] = frame.k;  // source pixels to frame pixels
    let found = await detect(frame), fresh = true;
    if (!found && hold) { found = { landmarks: hold.map(([x, y]) => [x * sx, y * sy]), box: null }; fresh = false; }
    if (!found) return null;
    const lm = found.landmarks;
    // Copy only the part of the frame the crop can reach (with a margin), scaled so the face is about
    // as wide as in the study's 256-pixel photos, then align that copy.
    const M = engine.similarityMatrix(lm);
    const [a, b, c, d, e, f] = M.length === 6 ? M : M.flat();
    const det2 = a * e - b * d;
    const inv = (u, v) => [(e * (u - c) - b * (v - f)) / det2, (-d * (u - c) + a * (v - f)) / det2];
    const corners = [[0, 0], [CROP, 0], [0, CROP], [CROP, CROP]].map(([u, v]) => inv(u, v));
    const [w, h] = [frame.width, frame.height];
    const x0 = Math.max(0, Math.floor(Math.min(...corners.map((p) => p[0])) - 2)), y0 = Math.max(0, Math.floor(Math.min(...corners.map((p) => p[1])) - 2));
    const x1 = Math.min(w, Math.ceil(Math.max(...corners.map((p) => p[0])) + 2)), y1 = Math.min(h, Math.ceil(Math.max(...corners.map((p) => p[1])) + 2));
    if (x1 <= x0 || y1 <= y0) return null;
    const faceW = found.box ? found.box[2] : Math.hypot(lm[1][0] - lm[0][0], lm[1][1] - lm[0][1]) * 2.6;
    const r = region(frame, x0, y0, x1 - x0, y1 - y0, Math.min(1, 120 / Math.max(1, faceW)));
    const crop = engine.alignCrop(r.rgba, r.W, r.H, lm.map(([x, y]) => [(x - x0) * r.kx, (y - y0) * r.ky]));
    return { crop: toRGBA(crop), landmarks: lm.map(([x, y]) => [x / sx, y / sy]), fresh, score: found.score };
  }
  return { detect, alignedFace };
}

function toRGBA(crop) {
  const px = crop.data || crop;
  if (px.length === CROP * CROP * 4) return new Uint8ClampedArray(px);
  const out = new Uint8ClampedArray(CROP * CROP * 4);
  for (let i = 0, j = 0; i < CROP * CROP; i++, j += 3) { out[4 * i] = px[j]; out[4 * i + 1] = px[j + 1]; out[4 * i + 2] = px[j + 2]; out[4 * i + 3] = 255; }
  return out;
}

/** SFace: embed(cropRGBA) gives the 128-number feature the study compared by cosine. */
export async function recognizer(rec, onProgress) {
  const { o, s } = await session(rec, onProgress);
  async function embed(rgba) {
    const inp = engine.sfaceInput(rgba);
    const out = await s.run({ [s.inputNames[0]]: new o.Tensor("float32", inp, [1, 3, CROP, CROP]) });
    return Float32Array.from(out[s.outputNames[0]].data);
  }
  return { embed, cosine: engine.cosine };
}

/** Draw an occluder from the study's list onto a copy of an aligned crop (RGBA), as the study drew it. */
export function occlude(rgba, cond) {
  const out = new Uint8ClampedArray(rgba);
  if (!cond || !cond.rect) return out;
  const [x0, y0, w, h] = cond.rect;
  for (let y = y0; y < Math.min(CROP, y0 + h); y++) {
    for (let x = x0; x < Math.min(CROP, x0 + w); x++) {
      const i = 4 * (y * CROP + x);
      const v = cond.fill === "gray" ? 128 : cond.fill === "noise" ? -1 : 0;
      for (let c = 0; c < 3; c++) out[i + c] = v < 0 ? Math.floor(Math.random() * 256) : v;
    }
  }
  return out;
}
