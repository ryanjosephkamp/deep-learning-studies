// Face alignment, SFace input, cosine scoring, and YuNet decoding: ports of these operations in OpenCV 4.12
// FaceRecognizerSF::alignCrop, ::feature's input blob, and ::match(FR_COSINE)
// (objdetect/src/face_recognize.cpp), and FaceDetectorYN's padding, input blob, and
// post-processing with dnn::NMSBoxes (objdetect/src/face_detect.cpp, dnn/src/nms.inl.hpp). The networks
// themselves run in ONNX Runtime, at YuNet's fixed 640 x 640 input, so detections can differ slightly
// from OpenCV's, which runs YuNet at each image's own size.
// Pure math: no DOM and no ONNX Runtime. Images are canvas-style RGBA (getImageData().data);
// OpenCV held the same pixels as BGR, so channel order is handled here.

const f32 = Math.fround;

// ---- alignment -------------------------------------------------------------------------------

// The five points alignCrop maps landmarks onto, in a 112 x 112 crop: the eye on the image's
// left, the eye on the image's right, the nose tip, then the mouth corners in the same order.
export const TEMPLATE = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
// alignCrop uses this rounded mean of TEMPLATE, not the exact mean.
const TEMPLATE_MEAN = [56.0262, 71.9008];
export const CROP_SIZE = 112;

// Five landmarks as float32 [x, y] pairs. Takes five pairs, ten numbers, or a YuNet face row
// [x, y, w, h, 10 landmark coordinates, score] (alignCrop reads row entries 4 to 13).
function landmarkPairs(lm) {
  if (lm.length === 5 && typeof lm[0] !== "number") return lm.map((p) => [f32(p[0]), f32(p[1])]);
  const off = lm.length >= 14 ? 4 : 0;
  if (lm.length !== 10 && off === 0) throw new Error("landmarks: expected 5 points, 10 numbers, or a face row");
  const out = [];
  for (let i = 0; i < 5; i++) out.push([f32(lm[off + 2 * i]), f32(lm[off + 2 * i + 1])]);
  return out;
}

// The 2 x 3 similarity transform alignCrop applies (getSimilarityTransformMatrix). Means and
// demeaned points are float32 and the cross-covariance sums are double, as in OpenCV. OpenCV then
// takes the SVD of the 2 x 2 cross-covariance A and forms U diag(1, sign det A) V^T and the scale
// (s0 + sign(det A) s1) / (var_x + var_y). For 2 x 2 matrices that rotation has cos and sin
// proportional to (A00 + A11, A10 - A01), and s0 + sign(det A) s1 equals their hypotenuse, so the
// same matrix is computed here in closed form (it agrees with the SVD route to rounding).
export function similarityMatrix(src5) {
  const src = landmarkPairs(src5);
  let sx = src[0][0], sy = src[0][1];
  for (let i = 1; i < 5; i++) { sx = f32(sx + src[i][0]); sy = f32(sy + src[i][1]); }
  const mx = f32(sx / 5), my = f32(sy / 5);
  const dmx = f32(TEMPLATE_MEAN[0]), dmy = f32(TEMPLATE_MEAN[1]);
  const s = src.map((p) => [f32(p[0] - mx), f32(p[1] - my)]);
  const d = TEMPLATE.map((p) => [f32(f32(p[0]) - dmx), f32(f32(p[1]) - dmy)]);
  let a00 = 0, a01 = 0, a10 = 0, a11 = 0, var1 = 0, var2 = 0;
  for (let i = 0; i < 5; i++) a00 += f32(d[i][0] * s[i][0]);
  for (let i = 0; i < 5; i++) a01 += f32(d[i][0] * s[i][1]);
  for (let i = 0; i < 5; i++) a10 += f32(d[i][1] * s[i][0]);
  for (let i = 0; i < 5; i++) a11 += f32(d[i][1] * s[i][1]);
  for (let i = 0; i < 5; i++) var1 += f32(s[i][0] * s[i][0]);
  for (let i = 0; i < 5; i++) var2 += f32(s[i][1] * s[i][1]);
  a00 /= 5; a01 /= 5; a10 /= 5; a11 /= 5; var1 /= 5; var2 /= 5;
  const p = a00 + a11, q = a10 - a01;
  const n = Math.hypot(p, q);
  const c = n > 0 ? p / n : 1, sn = n > 0 ? q / n : 0;
  const scale = 1.0 / (var1 + var2) * n;
  const ts0 = c * mx + -sn * my;
  const ts1 = sn * mx + c * my;
  return [[c * scale, -sn * scale, dmx - scale * ts0], [sn * scale, c * scale, dmy - scale * ts1]];
}

// Round half to even and saturate, as cvRound / saturate_cast<int>(double).
function roundInt(v) {
  if (!(v === v)) return -2147483648;
  let r = Math.round(v);
  if (r - v === 0.5 && r % 2 !== 0) r -= 1;
  return r > 2147483647 ? 2147483647 : r < -2147483648 ? -2147483648 : r;
}

const satShort = (v) => (v > 32767 ? 32767 : v < -32768 ? -32768 : v);

// OpenCV's fixed-point bilinear weights (initInterTab2D(INTER_LINEAR, true)): 32 x 32 sub-pixel
// cells, four weights each, summing to 1 << 15. The (0, 0) cell saturates to 32767 and OpenCV's
// correction step adds the missing unit to its last weight; the result is kept as is.
let WTAB = null;
function bilinearTable() {
  if (WTAB) return WTAB;
  const T = 32, SCALE = 1 << 15;
  const t1 = new Float32Array(2 * T);
  for (let i = 0; i < T; i++) { const x = f32(i * f32(1 / T)); t1[2 * i] = f32(1 - x); t1[2 * i + 1] = x; }
  const itab = new Int16Array(T * T * 4 + 8);
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < T; j++) {
      const b = (i * T + j) * 4;
      let isum = 0;
      for (let k1 = 0; k1 < 2; k1++) {
        for (let k2 = 0; k2 < 2; k2++) {
          const v = f32(t1[2 * i + k1] * t1[2 * j + k2]);
          const w = satShort(roundInt(f32(v * SCALE)));
          itab[b + k1 * 2 + k2] = w;
          isum += w;
        }
      }
      if (isum !== SCALE) {
        const diff = isum - SCALE;
        let M1 = 1, M2 = 1, m1 = 1, m2 = 1;
        for (let k1 = 1; k1 < 3; k1++) {
          for (let k2 = 1; k2 < 3; k2++) {
            if (itab[b + k1 * 2 + k2] < itab[b + m1 * 2 + m2]) { m1 = k1; m2 = k2; }
            else if (itab[b + k1 * 2 + k2] > itab[b + M1 * 2 + M2]) { M1 = k1; M2 = k2; }
          }
        }
        if (diff < 0) itab[b + M1 * 2 + M2] -= diff;
        else itab[b + m1 * 2 + m2] -= diff;
      }
    }
  }
  WTAB = itab.subarray(0, T * T * 4);
  return WTAB;
}

const flat6 = (M) => (M.length === 2 ? [...M[0], ...M[1]] : [...M]).map(Number);

// cv2.warpAffine(src, M, (outW, outH), flags=INTER_LINEAR, borderMode=BORDER_CONSTANT, 0) on
// 8-bit pixels. M maps source to destination; it is inverted as OpenCV does, each destination
// pixel's source position is put on a 1/1024 grid and cut to 1/32 of a pixel, and the four
// neighbors (0 outside the image) are blended with 15-bit integer weights.
// src: RGBA or RGB bytes (srcChannels 4 or 3). Output: RGB (outChannels 3) or RGBA with alpha 255.
export function warpAffine(src, w, h, M, outW, outH, outChannels = 3, srcChannels = 4) {
  const m = flat6(M);
  let D = m[0] * m[4] - m[1] * m[3];
  D = D !== 0 ? 1 / D : 0;
  const a11 = m[4] * D, a22 = m[0] * D;
  m[0] = a11; m[1] *= -D; m[3] *= -D; m[4] = a22;
  const b1 = -m[0] * m[2] - m[1] * m[5];
  const b2 = -m[3] * m[2] - m[4] * m[5];
  m[2] = b1; m[5] = b2;

  const adelta = new Int32Array(outW), bdelta = new Int32Array(outW);
  for (let x = 0; x < outW; x++) {
    adelta[x] = roundInt(m[0] * x * 1024);
    bdelta[x] = roundInt(m[3] * x * 1024);
  }
  const tab = bilinearTable();
  const out = new Uint8ClampedArray(outW * outH * outChannels);
  const rowStride = w * srcChannels;
  const tap = (sx, sy, c) => (sx >= 0 && sx < w && sy >= 0 && sy < h ? src[sy * rowStride + sx * srcChannels + c] : 0);
  let o = 0;
  for (let y = 0; y < outH; y++) {
    const X0 = (roundInt((m[1] * y + m[2]) * 1024) + 16) | 0;
    const Y0 = (roundInt((m[4] * y + m[5]) * 1024) + 16) | 0;
    for (let x = 0; x < outW; x++) {
      const X = ((X0 + adelta[x]) | 0) >> 5;
      const Y = ((Y0 + bdelta[x]) | 0) >> 5;
      const sx = satShort(X >> 5), sy = satShort(Y >> 5);
      const k = ((Y & 31) * 32 + (X & 31)) * 4;
      const w0 = tab[k], w1 = tab[k + 1], w2 = tab[k + 2], w3 = tab[k + 3];
      for (let c = 0; c < 3; c++) {
        const v = w0 * tap(sx, sy, c) + w1 * tap(sx + 1, sy, c) + w2 * tap(sx, sy + 1, c) + w3 * tap(sx + 1, sy + 1, c);
        out[o + c] = (v + 16384) >> 15;
      }
      if (outChannels === 4) out[o + 3] = 255;
      o += outChannels;
    }
  }
  return out;
}

// FaceRecognizerSF::alignCrop: the 112 x 112 crop (RGB, or RGBA with outChannels 4) for five
// landmarks or a YuNet face row, from an RGBA image of size w x h.
export function alignCrop(rgba, w, h, landmarks5, outChannels = 3) {
  return warpAffine(rgba, w, h, similarityMatrix(landmarks5), CROP_SIZE, CROP_SIZE, outChannels, 4);
}

// ---- SFace -----------------------------------------------------------------------------------

// The input FaceRecognizerSF::feature builds: blobFromImage(bgrCrop, 1, (112, 112), (0, 0, 0),
// swapRB=true, crop=false), a 1 x 3 x 112 x 112 float tensor with planes R, G, B and values 0-255.
// crop: 112 x 112 RGB or RGBA bytes.
export function sfaceInput(crop) {
  const n = CROP_SIZE * CROP_SIZE;
  const ch = crop.length === n * 4 ? 4 : crop.length === n * 3 ? 3 : 0;
  if (!ch) throw new Error("sfaceInput: expected a 112 x 112 RGB or RGBA crop");
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = crop[i * ch];
    out[n + i] = crop[i * ch + 1];
    out[2 * n + i] = crop[i * ch + 2];
  }
  return out;
}
export const SFACE_DIMS = [1, 3, CROP_SIZE, CROP_SIZE];

// FaceRecognizerSF::match(a, b, FR_COSINE): each feature is L2-normalized (norm in double,
// scale 1/norm applied in float32), then the float32 products are summed in double.
export function cosine(a, b) {
  const unit = (v) => {
    let ss = 0;
    for (let i = 0; i < v.length; i++) { const t = f32(v[i]); ss += t * t; }
    const nrm = Math.sqrt(ss);
    const k = f32(nrm > 2.220446049250313e-16 ? 1 / nrm : 0);
    const out = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = f32(v[i]) * k;
    return out;
  };
  if (a.length !== b.length) throw new Error("cosine: features differ in length");
  const ua = unit(a), ub = unit(b);
  let sum = 0;
  for (let i = 0; i < ua.length; i++) sum += f32(ua[i] * ub[i]);
  return sum;
}

// ---- YuNet -----------------------------------------------------------------------------------

export const YUNET_STRIDES = [8, 16, 32];
export const YUNET_OUTPUTS = ["cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
  "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"];

// FaceDetectorYN pads its input size up to a multiple of 32 (setInputSize).
export function yunetPadSize(inW, inH) {
  return { padW: (Math.trunc((inW - 1) / 32) + 1) * 32, padH: (Math.trunc((inH - 1) / 32) + 1) * 32 };
}

// The tensor FaceDetectorYN feeds the network: the image at the detector's input size, zero-padded
// on the right and bottom to multiples of 32 (padWithDivisor), as blobFromImage(padded) with no
// scaling, no mean, and BGR planes: 1 x 3 x height x width floats, values 0-255.
// face_detection_yunet_2023mar.onnx declares a fixed 1 x 3 x 640 x 640 input in ONNX Runtime, hence
// the 640 x 640 default. An image that fits is only padded, which is exactly what detect() feeds
// with setInputSize((640, 640)). A larger image is first shrunk by one factor so it fits (aspect
// kept, box-filter averaging like cv2.resize INTER_AREA; FaceDetectorYN itself never resizes) and
// placed top left; toImageCoords maps the detections back. Returns {data, width, height} (the
// tensor's size, inW x inH rounded up to multiples of 32) plus dims and the scale used.
export function yunetInput(rgba, w, h, inW = 640, inH = 640) {
  const { padW, padH } = yunetPadSize(inW, inH);
  const plane = padW * padH;
  const data = new Float32Array(3 * plane);
  const s = Math.min(inW / w, inH / h, 1);
  const rw = s < 1 ? Math.max(1, Math.min(inW, Math.round(w * s))) : w;
  const rh = s < 1 ? Math.max(1, Math.min(inH, Math.round(h * s))) : h;
  if (rw === w && rh === h) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4, j = y * padW + x;
        data[j] = rgba[i + 2]; data[plane + j] = rgba[i + 1]; data[2 * plane + j] = rgba[i];
      }
    }
  } else {
    const xt = areaTaps(rw, w), yt = areaTaps(rh, h);
    const row = new Float64Array(rw * 3), acc = new Float64Array(rw * 3);
    for (let y = 0; y < rh; y++) {
      acc.fill(0);
      for (const [sy, wy] of yt[y]) {
        row.fill(0);
        const base = sy * w * 4;
        for (let x = 0; x < rw; x++) {
          for (const [sx, wx] of xt[x]) {
            const i = base + sx * 4;
            row[3 * x] += wx * rgba[i]; row[3 * x + 1] += wx * rgba[i + 1]; row[3 * x + 2] += wx * rgba[i + 2];
          }
        }
        for (let k = 0; k < rw * 3; k++) acc[k] += wy * row[k];
      }
      for (let x = 0; x < rw; x++) {
        const j = y * padW + x;
        data[2 * plane + j] = Math.min(255, Math.round(acc[3 * x]));
        data[plane + j] = Math.min(255, Math.round(acc[3 * x + 1]));
        data[j] = Math.min(255, Math.round(acc[3 * x + 2]));
      }
    }
  }
  return { data, width: padW, height: padH, dims: [1, 3, padH, padW], inW, inH, scaleX: rw / w, scaleY: rh / h };
}

// Box-filter taps for shrinking n source pixels to m (m <= n): for each destination pixel, the
// source pixels it covers and the share of its area each one contributes.
function areaTaps(m, n) {
  const k = n / m, taps = [];
  for (let d = 0; d < m; d++) {
    const a = d * k, b = Math.min(n, (d + 1) * k), list = [];
    for (let i = Math.floor(a); i < b && i < n; i++) {
      const cover = Math.min(b, i + 1) - Math.max(a, i);
      if (cover > 1e-9) list.push([i, cover / (b - a)]);
    }
    taps.push(list);
  }
  return taps;
}

// expf, as std::exp(float): exp in double, rounded to float32.
const expf = (x) => f32(Math.exp(x));

// Integer box area and intersection (cv::Rect_<int>::area, operator&).
const area = (r) => Math.imul(r[2], r[3]);
const isEmpty = (r) => r[2] <= 0 || r[3] <= 0;
function interArea(a, b) {
  if (isEmpty(a) || isEmpty(b)) return 0;
  const [xmin, xmax] = a[0] < b[0] ? [a, b] : [b, a];
  const [ymin, ymax] = a[1] < b[1] ? [a, b] : [b, a];
  if ((xmin[0] < 0 && xmin[0] + xmin[2] < xmax[0]) || (ymin[1] < 0 && ymin[1] + ymin[3] < ymax[1])) return 0;
  const iw = Math.min(xmin[2] - (xmax[0] - xmin[0]), xmax[2]);
  const ih = Math.min(ymin[3] - (ymax[1] - ymin[1]), ymax[3]);
  return iw <= 0 || ih <= 0 ? 0 : Math.imul(iw, ih);
}
// rectOverlap: 1.f - (float)jaccardDistance(a, b).
function overlap(a, b) {
  const aa = area(a), ab = area(b);
  if (((aa + ab) | 0) <= 0) return 1;
  const aab = interArea(a, b);
  return f32(1 - f32(1.0 - aab / (((aa + ab) | 0) - aab)));
}

// dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, keep, eta=1, topK): keeps scores
// strictly above the threshold, stable-sorts them high to low, cuts to topK, and drops any box
// whose overlap with an already kept box exceeds nmsThreshold. Returns kept indices in order.
export function nmsBoxes(boxes, scores, scoreThreshold, nmsThreshold, topK) {
  const st = f32(scoreThreshold), nt = f32(nmsThreshold);
  let order = [];
  for (let i = 0; i < scores.length; i++) if (scores[i] > st) order.push(i);
  order.sort((i, j) => (scores[i] > scores[j] ? -1 : scores[j] > scores[i] ? 1 : 0));
  if (topK > 0 && topK < order.length) order = order.slice(0, topK);
  const keep = [];
  for (const idx of order) {
    let ok = true;
    for (let k = 0; k < keep.length && ok; k++) ok = overlap(boxes[idx], boxes[keep[k]]) <= nt;
    if (ok) keep.push(idx);
  }
  return keep;
}

const tensorData = (t) => (t && t.data !== undefined && !ArrayBuffer.isView(t) ? t.data : t);

// FaceDetectorYN's post-processing for face_detection_yunet_2023mar.onnx. outputs: the 12 output
// tensors by name (an ONNX Runtime results object, or plain Float32Arrays), or an array in
// YUNET_OUTPUTS order. For an input size inW x inH the network ran at the padded size, and each
// stride s has (padH / s) * (padW / s) anchors in row-major order: cls_s and obj_s hold one value
// per anchor, bbox_s four (dx, dy, log w, log h), kps_s ten (five dx, dy pairs).
// Returns faces [x, y, w, h, 10 landmark coordinates, score] in input-size pixels, float32 values,
// highest score first (after NMS), as FaceDetectorYN::detect. The defaults are OpenCV's; the study
// used STUDY_DETECTOR.
export function yunetDecode(outputs, inW = 640, inH = 640, { scoreThreshold = 0.9, nmsThreshold = 0.3, topK = 5000 } = {}) {
  const { padW, padH } = yunetPadSize(inW, inH);
  const get = (k) => {
    const t = tensorData(Array.isArray(outputs) ? outputs[k] : outputs[YUNET_OUTPUTS[k]]);
    if (!t) throw new Error(`yunetDecode: missing output ${YUNET_OUTPUTS[k]}`);
    return t;
  };
  const thr = f32(scoreThreshold);
  const faces = [];
  YUNET_STRIDES.forEach((stride, i) => {
    const cols = Math.trunc(padW / stride), rows = Math.trunc(padH / stride);
    const cls = get(i), obj = get(i + 3), bbox = get(i + 6), kps = get(i + 9);
    const n = rows * cols;
    if (cls.length !== n || obj.length !== n || bbox.length !== 4 * n || kps.length !== 10 * n) {
      throw new Error(`yunetDecode: stride ${stride} outputs do not match a ${padW} x ${padH} input`);
    }
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        let cs = f32(cls[idx]), os = f32(obj[idx]);
        cs = cs > 1 ? 1 : cs; cs = cs < 0 ? 0 : cs;
        os = os > 1 ? 1 : os; os = os < 0 ? 0 : os;
        const score = f32(Math.sqrt(f32(cs * os)));
        if (score < thr) continue;
        const cx = f32(f32(c + f32(bbox[idx * 4])) * stride);
        const cy = f32(f32(r + f32(bbox[idx * 4 + 1])) * stride);
        const bw = f32(expf(bbox[idx * 4 + 2]) * stride);
        const bh = f32(expf(bbox[idx * 4 + 3]) * stride);
        const face = [f32(cx - f32(bw / 2)), f32(cy - f32(bh / 2)), bw, bh];
        for (let k = 0; k < 5; k++) {
          face.push(f32(f32(f32(kps[idx * 10 + 2 * k]) + c) * stride));
          face.push(f32(f32(f32(kps[idx * 10 + 2 * k + 1]) + r) * stride));
        }
        face.push(score);
        faces.push(face);
      }
    }
  });
  if (faces.length <= 1) return faces;
  const boxes = faces.map((f) => [Math.trunc(f[0]), Math.trunc(f[1]), Math.trunc(f[2]), Math.trunc(f[3])]);
  const keep = nmsBoxes(boxes, faces.map((f) => f[14]), scoreThreshold, nmsThreshold, topK);
  return keep.map((k) => faces[k]);
}

// The detector settings the study used (FaceDetectorYN.create(..., 0.7, 0.3, 5000)).
export const STUDY_DETECTOR = { scoreThreshold: 0.7, nmsThreshold: 0.3, topK: 5000 };

// The face the study aligned: the highest score, first on ties (numpy argmax over column 14).
export function bestFace(faces) {
  let best = null;
  for (const f of faces) if (best === null || f[14] > best[14]) best = f;
  return best;
}

// Maps faces from detector input pixels back to the original image (divides positions, sizes,
// and landmarks by the scale yunetInput used). info: yunetInput's result, or {scaleX, scaleY}.
export function toImageCoords(faces, info) {
  const sx = info.scaleX, sy = info.scaleY;
  return faces.map((f) => f.map((v, i) => (i === 14 ? v : i % 2 === 0 ? v / sx : v / sy)));
}
