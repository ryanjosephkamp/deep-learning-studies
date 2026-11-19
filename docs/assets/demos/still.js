// "Does it still know you?": SFace, one of the study's two pretrained models, runs on the reader's
// device. The reader keeps one clear, aligned picture of a face as the reference, then covers the face
// (with one of the study's occluders drawn on the aligned crop, or a hand) and watches the cosine score
// against the study's own cutoffs. Models load only on a tap; images never leave the device. The
// wording is in still-model.js, as functions of the numbers.
import { verdict, scoreLine, sourceNote, cutoffNote, passes, MB, farText, NO_FACE } from "./still-model.js?v=48670703c0";

const SVG = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, kids = []) => {
  const n = tag.startsWith("svg:") ? document.createElementNS(SVG, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === "text") n.textContent = v; else n.setAttribute(k, v); }
  for (const k of kids) n.append(k);
  return n;
};
const CROP = 112;
const LO = -0.2, HI = 1;  // the gauge's range of cosine scores; a lower score sits at the left end, marked as off the scale
const pos = (v) => (100 * (Math.min(HI, Math.max(LO, v)) - LO)) / (HI - LO);

function radios(label, items, onPick) {
  const g = el("div", { class: "st-chips", role: "radiogroup", "aria-label": label });
  const bs = items.map((it) => {
    const b = el("button", { type: "button", role: "radio", "data-id": it.id, text: it.label });
    b.addEventListener("click", () => onPick(it.id));
    g.append(b); return b;
  });
  g.addEventListener("keydown", (e) => {
    const s = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    const i = bs.indexOf(document.activeElement);
    if (!s || i < 0) return;
    e.preventDefault(); const n = bs[(i + s + bs.length) % bs.length]; n.click(); n.focus();
  });
  return { g, set(id) { for (const b of bs) { const on = b.dataset.id === id; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; } }, bs };
}

const put = (canvas, rgba) => canvas.getContext("2d").putImageData(new ImageData(rgba, CROP, CROP), 0, 0);
const wipe = (canvas) => canvas.getContext("2d").clearRect(0, 0, CROP, CROP);

export async function mount(root) {
  const data = await (await fetch(root.dataset.src)).json();
  const stage = root.querySelector(".plate-live-stage");
  const total = data.recognizer.bytes + data.detector.bytes + data.detector.runtime_bytes;
  const thr = data.thresholds;  // [{far: 0.01, cos}, {far: 0.001, cos}], loosest first
  const portraits = data.faces.filter((f) => f.id !== "drawn");

  // Before anything loads: what it does, what it costs, and that nothing leaves the device.
  const intro = el("div", { class: "st-intro" }, [
    el("p", { text: "This runs SFace itself, on your device. Keep one clear picture of a face as the reference, then cover the face and watch its score against the study's cutoffs." }),
    el("p", { class: "st-fine", text: `Loading fetches about ${MB(total)}: SFace (${MB(data.recognizer.bytes)}) and the YuNet face finder (${MB(data.detector.bytes)}) from OpenCV's model pages, and ONNX Runtime Web to run them from jsDelivr; every file is checked against a known SHA-256 before it runs. Your camera and photos stay on this device.` }),
  ]);
  const load = el("button", { type: "button", class: "st-load", text: `Load SFace (${MB(total)})` });
  const progress = el("p", { class: "st-progress", role: "status" });
  intro.append(load, progress);

  // The live part, shown once loaded.
  const live = el("div", { class: "st-live", hidden: "" });
  const camBtn = el("button", { type: "button", class: "st-src", text: "Use your camera" });
  const photoBtn = el("button", { type: "button", class: "st-src", text: "Use a photo" });
  const faceBtn = el("button", { type: "button", class: "st-src", text: portraits.length ? "Use a synthetic portrait" : "Use the drawn face" });
  const sources = el("div", { class: "st-sources" }, [camBtn, photoBtn, faceBtn]);
  const occR = radios("Occluder", [{ id: "none", label: "No occluder" }, { id: "square", label: "Centered square" }, { id: "eye", label: "Eye band" }, { id: "low", label: "Lower band" }],
    (id) => { state.occ = id; renderControls(); rescore(); });
  const squares = data.occluders.filter((c) => c.kind === "square").sort((a, b) => a.size - b.size);
  const range = el("input", { type: "range", min: "0", max: String(squares.length - 1), step: "1", value: String(squares.findIndex((c) => c.size === 20)), "aria-label": "Share of the aligned crop covered" });
  const rangeOut = el("output", { class: "st-size-out" });
  const sizeRow = el("label", { class: "st-size" }, [el("span", { text: "Square" }), range, rangeOut]);
  range.addEventListener("input", () => { renderControls(); rescore(); });
  const keep = el("button", { type: "button", class: "st-keep", text: "Keep this as the clear reference" });
  const hint = el("p", { class: "st-hint" });
  const file = el("input", { type: "file", accept: "image/*", hidden: "" });

  const refC = el("canvas", { class: "st-crop", width: CROP, height: CROP, "aria-hidden": "true" });
  const probeC = el("canvas", { class: "st-crop", width: CROP, height: CROP, "aria-hidden": "true" });
  const refCap = el("figcaption", { text: "Reference" }), probeCap = el("figcaption", { text: "Now" });
  const pics = el("div", { class: "st-pics" }, [el("figure", {}, [refC, refCap]), el("figure", {}, [probeC, probeCap])]);
  // The gauge: the score, its last few seconds, and the study's two cutoffs.
  const gauge = el("div", { class: "st-gauge", role: "img", "aria-label": "No score yet" });
  const track = el("div", { class: "st-track" });
  for (const t of thr) track.append(el("span", { class: "st-thr", style: `left:${pos(t.cos)}%` }, [el("span", { class: "st-thr-l", text: `${farText(t.far)} cutoff` })]));
  const trail = el("svg:svg", { class: "st-trail", viewBox: "0 0 100 20", preserveAspectRatio: "none", "aria-hidden": "true" });
  const trailPath = el("svg:polyline", { points: "" }); trail.append(trailPath);
  const dot = el("span", { class: "st-dot", hidden: "" });
  track.append(trail, dot);
  const ticks = el("div", { class: "st-ticks", "aria-hidden": "true" }, ["0", "0.25", "0.5", "0.75", "1"].map((v) => el("span", { style: `left:${pos(Number(v))}%`, text: v })));
  gauge.append(track, ticks);
  const read = el("p", { class: "st-read" });
  const say = el("p", { class: "st-say" });
  const announce = el("p", { class: "st-announce", role: "status" });  // verdict changes only, for screen readers
  const note = el("p", { class: "st-note" });
  const cut = el("p", { class: "st-cut", text: cutoffNote(thr) });
  live.append(el("div", { class: "st-body" }, [pics, el("div", { class: "st-meter" }, [gauge, read, say, cut])]),
    el("div", { class: "st-controls" }, [sources, el("div", { class: "st-row" }, [el("span", { class: "st-label", text: "Cover" }), occR.g]), sizeRow, keep, hint, file]), announce, note);
  stage.append(intro, live);

  const state = { token: 0, source: null, occ: "square", refEmb: null, lastCrop: null, lastFresh: false, history: [], stream: null, loop: 0, finder: null, sface: null, holdLm: null, holdAt: 0, portrait: -1, lastOk: null };

  function occluder() {
    if (state.occ === "none") return null;
    if (state.occ === "square") return squares[+range.value];
    return data.occluders.find((c) => c.kind === state.occ);
  }
  function renderControls() {
    occR.set(state.occ);
    const c = squares[+range.value];
    rangeOut.textContent = `${c.size}%`;
    range.setAttribute("aria-valuetext", `${c.size}% of the aligned crop`);
    sizeRow.hidden = state.occ !== "square";
    keep.hidden = state.source !== "camera";
    keep.disabled = !(state.source === "camera" && state.lastFresh);
    camBtn.textContent = state.stream ? "Stop the camera" : "Use your camera";
    camBtn.setAttribute("aria-pressed", String(Boolean(state.stream)));
    const o = occluder();
    probeCap.textContent = o ? `Now: ${o.label.charAt(0).toLowerCase() + o.label.slice(1)}` : "Now: no occluder drawn";
    hint.textContent = state.source === "camera" ? (state.refEmb ? "Try covering part of your face with a hand, or choose an occluder." : "Face the camera in good light, then keep a clear frame as the reference.") : "";
    note.textContent = state.source ? sourceNote(state.source) : "";
  }
  /** Back to an empty view: no reference, no score, nothing from an earlier source on screen. */
  function clearView() {
    wipe(refC); wipe(probeC);
    state.refEmb = null; state.lastCrop = null; state.lastFresh = false; state.history = []; state.holdLm = null; state.lastOk = null;
    dot.hidden = true; trailPath.setAttribute("points", "");
    read.textContent = ""; say.textContent = ""; announce.textContent = "";
    refCap.textContent = "Reference";
    gauge.setAttribute("aria-label", "No score yet");
  }

  async function scoreCrop(crop, token = state.token) {
    const { occlude } = await import("./face-runtime.js?v=ad42aeb51c");
    const probe = occlude(crop, occluder());
    if (token !== state.token) return;
    put(probeC, probe);
    if (!state.refEmb) return;
    const e = await state.sface.embed(probe);
    if (token !== state.token || !state.refEmb) return;
    show(state.sface.cosine(state.refEmb, e));
  }
  function show(s) {
    dot.hidden = false;
    dot.style.left = `${pos(s)}%`;
    dot.classList.toggle("is-off", s < LO);
    const now = performance.now();
    state.history.push([now, s]);
    while (state.history.length && now - state.history[0][0] > 8000) state.history.shift();
    trailPath.setAttribute("points", state.history.map(([t, v]) => `${pos(v)},${20 - (20 * (now - t)) / 8000}`).join(" "));
    read.textContent = scoreLine(s, thr);
    say.textContent = verdict(s, thr, state.source, occluder());
    gauge.setAttribute("aria-label", `Cosine score ${s.toFixed(3)}; the study's cutoffs are ${thr.map((t) => `${t.cos.toFixed(3)} for ${farText(t.far)} false accepts`).join(" and ")}`);
    const ok = passes(s, thr[0]);
    if (state.source !== "camera" || ok !== state.lastOk) announce.textContent = say.textContent;
    state.lastOk = ok;
  }
  async function setReference(crop, token = state.token) {
    const e = await state.sface.embed(crop);
    if (token !== state.token) return;
    put(refC, crop);
    state.refEmb = e; state.history = [];
    renderControls();
  }
  async function rescore() { if (state.lastCrop && state.source !== "camera") await scoreCrop(state.lastCrop); }

  function stopCamera() {
    clearTimeout(state.loop); state.loop = 0;
    if (state.stream) { for (const t of state.stream.getTracks()) t.stop(); state.stream = null; }
  }
  /** Start a new source: everything from the previous one stops and leaves the screen. */
  function begin(source) {
    stopCamera();
    const token = ++state.token;
    state.source = source;
    clearView(); renderControls();
    return token;
  }
  async function useStill(crop, token, label) {
    if (token !== state.token) return;
    state.lastCrop = crop;
    refCap.textContent = `Reference: ${label}, clear`;
    await setReference(crop, token);
    await scoreCrop(crop, token);
  }

  faceBtn.addEventListener("click", async () => {
    const token = begin(portraits.length ? "portrait" : "drawn");
    const list = portraits.length ? portraits : data.faces;
    state.portrait = (state.portrait + 1) % list.length;
    const f = list[state.portrait];
    if (portraits.length) faceBtn.textContent = `Next synthetic portrait (${state.portrait + 1} of ${list.length})`;
    const img = await createImageBitmap(await (await fetch(f.src)).blob());
    const c = new OffscreenCanvas(CROP, CROP); const g = c.getContext("2d"); g.imageSmoothingQuality = "high"; g.drawImage(img, 0, 0, CROP, CROP);
    img.close();
    await useStill(g.getImageData(0, 0, CROP, CROP).data, token, portraits.length ? "the portrait" : "the drawing");
  });
  photoBtn.addEventListener("click", () => file.click());
  file.addEventListener("change", async () => {
    const f = file.files && file.files[0]; file.value = "";
    if (!f) return;
    const token = begin("photo");
    try {
      const bmp = await createImageBitmap(f, { imageOrientation: "from-image" }).catch(() => createImageBitmap(f));
      const found = await state.finder.alignedFace(bmp);
      bmp.close();
      if (token !== state.token) return;
      if (!found) { read.textContent = "No face found in that photo. Try one where the face is upright and well lit."; return; }
      await useStill(found.crop, token, "your photo");
    } catch (e) { if (token === state.token) read.textContent = `That photo couldn't be read here (${e.message}).`; }
  });

  camBtn.addEventListener("click", async () => {
    if (state.stream) { begin(null); return; }
    const token = begin("camera");
    read.textContent = "Asking for the camera…";
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    } catch (e) {
      if (token === state.token) read.textContent = e && e.name === "NotAllowedError" ? "The camera wasn't allowed. You can use a photo instead." : `The camera couldn't start here (${e.message}).`;
      return;
    }
    // A request answered after the reader moved on is closed at once.
    if (token !== state.token) { for (const t of stream.getTracks()) t.stop(); return; }
    state.stream = stream;
    const video = el("video", { playsinline: "", muted: "" });
    video.muted = true; video.srcObject = stream;
    try { await video.play(); } catch (e) { begin(null); read.textContent = `The camera couldn't start here (${e.message}).`; return; }
    read.textContent = "";
    renderControls();
    // One frame at a time: the next starts when this one is scored; off screen or hidden, it waits.
    const step = async () => {
      if (token !== state.token || !state.stream) return;
      if (!onScreen || document.hidden || video.readyState < 2) { state.loop = setTimeout(step, 250); return; }
      try {
        const hold = state.holdLm && performance.now() - state.holdAt < 1500 ? state.holdLm : null;
        const f = await state.finder.alignedFace(video, { hold });
        if (token !== state.token) return;
        state.lastFresh = Boolean(f && f.fresh);
        if (!f) { wipe(probeC); read.textContent = "No score for this frame."; say.textContent = NO_FACE; renderControls(); }
        else {
          if (f.fresh) { state.holdLm = f.landmarks; state.holdAt = performance.now(); }
          state.lastCrop = f.crop;
          renderControls();
          if (!state.refEmb) put(probeC, f.crop);
          else {
            await scoreCrop(f.crop, token);
            if (!f.fresh && token === state.token) say.textContent += " No face was found in this frame, so it was aligned from your last clear one, as the study aligned each face before covering it.";
          }
        }
      } catch (e) {
        if (token !== state.token) return;
        begin(null);
        read.textContent = `The camera view stopped: ${e.message}.`;
        return;
      }
      if (token === state.token && state.stream) state.loop = setTimeout(step, 15);
    };
    step();
  });
  keep.addEventListener("click", async () => {
    if (state.source !== "camera" || !state.lastFresh || !state.lastCrop) return;
    refCap.textContent = "Reference: your clear frame";
    await setReference(state.lastCrop);
  });

  let onScreen = true;
  new IntersectionObserver((es) => { onScreen = es[0].isIntersecting; }, { threshold: 0.1 }).observe(root);
  addEventListener("pagehide", () => { begin(null); });

  load.addEventListener("click", async () => {
    load.disabled = true;
    const got = { d: 0, r: 0 };
    const tell = () => { progress.textContent = `Loading: ${MB(got.d + got.r)} of ${MB(data.detector.bytes + data.recognizer.bytes)} of model files…`; };
    try {
      const rt = await import("./face-runtime.js?v=ad42aeb51c");
      [state.finder, state.sface] = await Promise.all([
        rt.faceFinder(data.detector, (n) => { got.d = n; tell(); }),
        rt.recognizer(data.recognizer, (n) => { got.r = n; tell(); }),
      ]);
      live.hidden = false;
      faceBtn.click();
      progress.textContent = "Loaded, and every file matches its SHA-256.";
      intro.classList.add("is-done");
      faceBtn.focus();
    } catch (e) {
      load.disabled = false;
      progress.textContent = `It couldn't load here: ${e.message}. You can try again.`;
    }
  });
  renderControls();
}

document.querySelectorAll('[data-demo="sface"]').forEach((n) => mount(n));
