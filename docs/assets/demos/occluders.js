// The occluder explorer's view. It draws a face on the aligned crop with the chosen occluder over it,
// beside every model's measured rank-1, from data/faces.json. All values and sentences come from
// occluders-model.js. The face is a drawn face, a synthetic portrait, or the reader's own photo or camera,
// found and aligned on the device by face-runtime.js (loaded only when the reader asks for it).
import { KINDS, fitNote, conditionsOf, pick, caption } from "./occluders-model.js?v=ca4c8c620f";

const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const usDate = (iso) => { const [y, m, d] = iso.split("-").map(Number); return `${MONTHS[m - 1]} ${d}, ${y}`; };

// Arrow keys move the choice within a radio group, with one tab stop for the group.
function radioKeys(group) {
  group.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!step) return;
    const bs = [...group.querySelectorAll("button")];
    const i = bs.indexOf(document.activeElement);
    if (i < 0) return;
    e.preventDefault();
    const next = bs[(i + step + bs.length) % bs.length];
    next.click(); next.focus();
  });
}

const SVG = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, kids = []) => {
  const n = tag.startsWith("svg:") ? document.createElementNS(SVG, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
};

function noiseCells(rect, seed) {
  // A fixed example of uniform noise, in 4-pixel cells, so the drawing stays still.
  const [x, y, w, h] = rect;
  const g = el("svg:g");
  let s = seed;
  const rand = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let i = x; i < x + w; i += 4) {
    for (let j = y; j < y + h; j += 4) {
      const v = Math.round(rand() * 255);
      g.append(el("svg:rect", { x: i, y: j, width: Math.min(4, x + w - i), height: Math.min(4, y + h - j), fill: `rgb(${v},${v},${v})` }));
    }
  }
  return g;
}

// ---- the face under the occluder ----------------------------------------------------------------

const MB = (n) => `${(n / 1e6).toFixed(n < 1e6 ? 1 : 0)} MB`;

function loadImage(src) {
  return new Promise((ok, no) => { const i = new Image(); i.onload = () => ok(i); i.onerror = no; i.src = src; });
}

/** The face strip: the drawn face, any synthetic portraits, and the reader's own photo or camera. */
function mountFaces(root, data, canvas, redraw) {
  const ctx = canvas.getContext("2d");
  const portraits = data.faces;  // the synthetic portraits, or the drawn face when there are none
  const strip = el("div", { class: "ox-faces", role: "radiogroup", "aria-label": "Face" });
  const own = el("div", { class: "ox-own" });
  const status = el("p", { class: "ox-facestatus", role: "status" });
  const file = el("input", { type: "file", accept: "image/*", hidden: "" });
  let current = portraits[0], stream = null, loop = 0, runtime = null, lastLm = null, lastAt = 0, token = 0, onScreen = true;
  new IntersectionObserver((es) => { onScreen = es[0].isIntersecting; }, { threshold: 0.1 }).observe(root);

  function paint(src) {
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
  }
  function paintCrop(rgba) {
    const c = document.createElement("canvas"); c.width = c.height = data.crop;
    c.getContext("2d").putImageData(new ImageData(rgba, data.crop, data.crop), 0, 0);
    paint(c);
  }
  /** Stop whatever the reader's own face source was doing; a new choice gets a new token. */
  function stop() {
    token++;
    clearTimeout(loop); loop = 0;
    if (stream) { for (const tr of stream.getTracks()) tr.stop(); stream = null; }
    lastLm = null;
    camBtn.textContent = "Use your camera"; camBtn.setAttribute("aria-pressed", "false");
    return token;
  }
  async function getRuntime() {
    if (runtime) return runtime;
    status.textContent = "Loading the face finder…";
    const mod = await import("./face-runtime.js?v=ad42aeb51c");
    runtime = await mod.faceFinder(data.detector, (got, total) => { status.textContent = `Loading the face finder: ${MB(got)} of ${MB(total)}`; });
    return runtime;
  }

  const buttons = [];
  function choose(id) {
    for (const b of buttons) { const on = b.dataset.face === id; b.setAttribute("aria-checked", String(on)); b.tabIndex = on || (id === "own" && b === buttons[0]) ? 0 : -1; }
  }
  for (const f of portraits) {
    const b = el("button", { type: "button", role: "radio", class: "ox-facebtn", "data-face": f.id, "aria-label": f.name, title: f.name });
    b.append(el("img", { src: f.src, alt: "", width: "56", height: "56", loading: "lazy" }));
    b.addEventListener("click", async () => {
      stop(); current = f; choose(f.id);
      paint(await loadImage(f.src));
      status.textContent = f.credit; redraw();
    });
    buttons.push(b); strip.append(b);
  }
  strip.addEventListener("keydown", (e) => {
    const s = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    const i = buttons.indexOf(document.activeElement);
    if (!s || i < 0) return;
    e.preventDefault(); const n = buttons[(i + s + buttons.length) % buttons.length]; n.click(); n.focus();
  });

  const sizeNote = ` Finding a face loads the YuNet face finder and ONNX Runtime Web, about ${MB(data.detector.bytes + data.detector.runtime_bytes)}, the first time; every file is checked against a known SHA-256.`;
  const photoBtn = el("button", { type: "button", class: "ox-ownbtn", text: "Use a photo", title: `Choose a photo of a face. It stays on this device.${sizeNote}` });
  const camBtn = el("button", { type: "button", class: "ox-ownbtn", text: "Use your camera", "aria-pressed": "false", title: `Use your camera. Frames stay on this device.${sizeNote}` });
  own.append(photoBtn, camBtn, file);
  photoBtn.addEventListener("click", () => file.click());
  file.addEventListener("change", async () => {
    const f = file.files && file.files[0]; file.value = "";
    if (!f) return;
    const my = stop();
    try {
      const rt = await getRuntime();
      const bmp = await createImageBitmap(f, { imageOrientation: "from-image" }).catch(() => createImageBitmap(f));
      const found = await rt.alignedFace(bmp);
      bmp.close();
      if (my !== token) return;
      if (!found) { status.textContent = "No face found in that photo. Try one where the face is upright and well lit."; return; }
      paintCrop(found.crop);
      current = { id: "own", name: "your photo" }; choose("own"); redraw();
      status.textContent = "Your photo, found and aligned on this device as the study aligned its photos. Nothing was uploaded.";
    } catch (e) { if (my === token) status.textContent = `That photo couldn't be used here (${e.message}).`; }
  });
  camBtn.addEventListener("click", async () => {
    if (stream) { stop(); status.textContent = "The camera is off."; return; }
    const my = stop();
    try {
      const rt = await getRuntime();
      if (my !== token) return;
      status.textContent = "Asking for the camera…";
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      // A request answered after the reader moved on is closed at once.
      if (my !== token) { for (const tr of s.getTracks()) tr.stop(); return; }
      stream = s;
      const video = el("video", { playsinline: "", muted: "" });
      video.muted = true; video.srcObject = stream; await video.play();
      current = { id: "own", name: "your camera's picture" }; choose("own"); redraw();
      camBtn.textContent = "Stop the camera"; camBtn.setAttribute("aria-pressed", "true");
      status.textContent = "Your camera, aligned on this device as the study aligned its photos. Frames never leave it.";
      // One frame at a time: the next starts when this one is drawn; off screen or hidden, it waits.
      const step = async () => {
        if (my !== token) return;
        if (!onScreen || document.hidden || video.readyState < 2) { loop = setTimeout(step, 250); return; }
        try {
          const found = await rt.alignedFace(video, { hold: lastLm && performance.now() - lastAt < 1500 ? lastLm : null });
          if (my !== token) return;
          if (found) { paintCrop(found.crop); if (found.fresh) { lastLm = found.landmarks; lastAt = performance.now(); } }
          status.textContent = found ? (found.fresh ? "Your camera, aligned on this device as the study aligned its photos. Frames never leave it." : "No face found in this frame, so the last alignment is held, as the study aligned each face before covering it.") : "No face found. Face the camera in good light.";
        } catch (e) {
          if (my !== token) return;
          stop(); status.textContent = `The camera view stopped: ${e.message}.`; return;
        }
        if (my === token) loop = setTimeout(step, 30);
      };
      step();
    } catch (e) {
      if (my !== token) return;
      stop();
      status.textContent = e && e.name === "NotAllowedError" ? "The camera wasn't allowed. You can still use a photo." : `The camera couldn't start here (${e.message}).`;
    }
  });
  addEventListener("pagehide", () => { stop(); });

  choose(current.id);
  loadImage(current.src).then(paint);
  status.textContent = current.credit;
  return { strip, own, status, describe: () => (current.id === "drawn" ? "a drawn face" : current.id === "own" ? current.name : `a synthetic portrait of ${current.name}`) };
}

export async function mount(root) {
  const data = await (await fetch(root.dataset.src)).json();
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const state = { kind: "square", size: 20, test: "identity", playing: !reduce, touched: false };

  // Controls
  const kinds = el("div", { class: "ox-kinds", role: "radiogroup", "aria-label": "Occluder" });
  for (const k of KINDS) {
    const b = el("button", { type: "button", role: "radio", "data-kind": k.key, text: k.label });
    b.addEventListener("click", () => { touch(); state.kind = k.key; const list = conditionsOf(data, k.key); state.size = pick(data, k.key, state.size).size ?? list[0].size; render(); });
    kinds.append(b);
  }
  const range = el("input", { type: "range", min: "0", max: "9", step: "1", "aria-label": "Share of the crop covered" });
  const out = el("output", { class: "ox-size-out" });
  const sizeLab = el("label", { class: "ox-size" }, [el("span", { text: "Share of the crop" }), range, out]);
  range.addEventListener("input", () => { touch(); const list = conditionsOf(data, state.kind); state.size = list[+range.value].size; render(); });
  const tests = el("div", { class: "ox-test", role: "radiogroup", "aria-label": "Test" });
  for (const [key, label] of [["identity", "A new photo of the person"], ["same", "The same photo (the original way of scoring)"]]) {
    const b = el("button", { type: "button", role: "radio", "data-test": key, text: label });
    b.addEventListener("click", () => { touch(); state.test = key; render(); });
    tests.append(b);
  }
  radioKeys(kinds); radioKeys(tests);
  const play = el("button", { type: "button", class: "ox-play" });
  play.addEventListener("click", () => { state.playing = !state.playing; state.touched = true; render(); if (state.playing && !timer) timer = setTimeout(tick, 1600); });

  // The crop: the face drawn on a canvas, with the occluder and the template's landmarks over it
  const face = el("canvas", { class: "ox-face", width: 448, height: 448, "aria-hidden": "true" });
  const crop = el("svg:svg", { class: "ox-crop", viewBox: `0 0 ${data.crop} ${data.crop}`, role: "img" });
  const occ = el("svg:g", { class: "ox-occ" });
  const dots = el("svg:g", { class: "ox-dots" });
  for (const [x, y] of data.landmarks) dots.append(el("svg:circle", { cx: x, cy: y, r: 1.6 }));
  crop.append(occ, dots);
  const stack = el("div", { class: "ox-stack" }, [face, crop]);
  const faces = mountFaces(root, data, face, () => render());

  // The chart
  const detail = el("p", { class: "ox-detail", "aria-live": "polite", text: "Tap or focus a row to see its 95% interval." });
  let selected = null;
  function showDetail() {
    if (!selected) return;
    const cond = pick(data, state.kind, state.size);
    const m = data.models.find((x) => x.id === selected);
    const scores = state.test === "same" ? cond.same : cond.identity;
    const s = scores && scores[selected];
    detail.textContent = s ? `${cond.label}, ${m.name}: ${s.s}% (95% interval ${s.lo} to ${s.hi}%)` : `${cond.label}, ${m.name}: not run in this test`;
  }
  const chart = el("div", { class: "ox-chart", role: "list" });
  const rows = {};
  for (const m of data.models) {
    const bar = el("span", { class: "ox-ci" });
    const dot = el("span", { class: "ox-dot" });
    const ghost = el("span", { class: "ox-ghost", title: "No occluder" });
    const track = el("span", { class: "ox-track" }, [ghost, bar, dot]);
    const val = el("span", { class: "ox-val" });
    const row = el("div", { class: `ox-row ox-${m.group}`, role: "listitem", tabindex: "0" }, [el("span", { class: "ox-name", text: m.name }), track, val]);
    // Focus, hover, or a tap picks the row whose interval the line under the chart shows; render() keeps that line current.
    const show = () => { selected = m.id; showDetail(); };
    row.addEventListener("focus", show); row.addEventListener("mouseenter", show);
    row.addEventListener("click", () => { touch(); selected = m.id; render(); });
    rows[m.id] = { row, bar, dot, ghost, val };
    chart.append(row);
  }
  const legend = el("p", { class: "ox-legend", text: `Dot: rank-1, the share of probes whose closest gallery photo is the right person. Bar: its 95% interval, over test identities and, for the CNNs, their training seeds. Faint mark: the same model with no occluder. CNN rows: the project's network with attention, occlusion training, both, or neither, and one trained with a triplet loss, which failed to train, each the mean of ${WORDS[data.seeds] ?? data.seeds} training seeds.` });
  const axis = el("div", { class: "ox-axis", "aria-hidden": "true" }, [el("span"), el("span", { class: "ox-ticks" }, ["0", "25", "50", "75", "100%"].map((t) => el("span", { text: t }))), el("span")]);
  chart.append(axis);

  const cap = el("p", { class: "ox-caption", "aria-live": "polite" });
  const checked = el("p", { class: "ox-checked" });
  fetch(root.dataset.src.replace(/[^/]+\.json(\?.*)?$/, "checks.json"), { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (!c) return;
    const a = el("a", { href: data.checker_url, text: "the site's checker" });
    checked.replaceChildren(`Checked ${usDate(c.checked)}: ${c.passed.toLocaleString("en-US")} checks by `, a, ` of this page's values, and the numbers in its sentences, against the study's results; ${c.failed} failed.`);
  }).catch(() => {});
  const fit = el("p", { class: "ox-fit" });
  const chartHead = el("p", { class: "ox-charthead", text: `Each model's measured rank-1 for this occluder, on the study's test set` });
  const body = el("div", { class: "ox-body" }, [el("div", { class: "ox-cropwrap" }, [stack, faces.strip, faces.own, faces.status, fit]), el("div", { class: "ox-chartwrap" }, [chartHead, chart])]);
  const controls = el("div", { class: "ox-controls" }, [kinds, sizeLab, tests, play]);
  root.querySelector(".plate-live-stage").append(controls, body, detail, legend, cap, checked);

  const clean = data.conditions.find((c) => c.kind === "square" && c.size === 0);

  function render() {
    const cond = pick(data, state.kind, state.size);
    state.size = cond.size;
    const list = conditionsOf(data, state.kind);
    kinds.querySelectorAll("button").forEach((b) => { const on = b.dataset.kind === state.kind; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    tests.querySelectorAll("button").forEach((b) => { const on = b.dataset.test === state.test; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    range.max = String(list.length - 1);
    range.value = String(list.indexOf(cond));
    range.disabled = list.length < 2;
    out.textContent = `${cond.area}%`;
    range.setAttribute("aria-valuetext", `${cond.area}% of the crop`);
    play.textContent = state.playing ? "Pause" : "Play";
    play.setAttribute("aria-pressed", String(state.playing));

    occ.replaceChildren();
    if (cond.rect) {
      const [x, y, w, h] = cond.rect;
      if (cond.fill === "noise") occ.append(noiseCells(cond.rect, 7));
      else occ.append(el("svg:rect", { x, y, width: w, height: h, class: `ox-fill-${cond.fill}` }));
      if (cond.kind === "off") occ.append(el("svg:rect", { x, y, width: w, height: h, class: "ox-example" }));
    }
    crop.setAttribute("aria-label", `${cond.label}, drawn over ${faces.describe()} on the ${data.crop}-pixel aligned crop`);

    const scores = state.test === "same" ? cond.same : cond.identity;
    for (const m of data.models) {
      const r = rows[m.id];
      const s = scores && scores[m.id];
      r.row.classList.toggle("is-empty", !s);
      if (!s) { r.val.textContent = "not run"; r.row.setAttribute("aria-label", `${m.name}: not run in this test`); r.row.title = ""; continue; }
      r.dot.style.left = `${s.v}%`;
      r.bar.style.left = `${s.l}%`;
      r.bar.style.width = `${Math.max(0, s.h - s.l)}%`;
      const base = (state.test === "same" ? clean.same : clean.identity)[m.id];
      r.ghost.style.left = `${base.v}%`;
      r.val.textContent = `${s.s}%`;
      r.row.setAttribute("aria-label", `${m.name}: ${s.s}%, 95% interval ${s.lo} to ${s.hi}%`);
      r.row.title = `${m.name}: ${s.s}% (95% interval ${s.lo} to ${s.hi}%)`;
    }
    cap.setAttribute("aria-live", state.playing && !state.touched ? "off" : "polite");
    cap.textContent = caption(data, cond, state.test);
    fit.textContent = fitNote(data, cond);
    detail.setAttribute("aria-live", state.playing && !state.touched ? "off" : "polite");
    showDetail();
  }

  // Alive: when nobody has touched it, it steps through the square sizes on its own.
  function touch() { state.touched = true; state.playing = false; }
  let visible = false, timer = 0;
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { threshold: 0.25 }).observe(root);
  const sizes = conditionsOf(data, "square").map((c) => c.size);
  function tick() {
    if (state.playing && visible && !document.hidden) {
      if (state.kind !== "square") state.kind = "square";
      const i = sizes.indexOf(state.size);
      state.size = sizes[(i + 1) % sizes.length];
      render();
    }
    timer = setTimeout(tick, 1600);
  }
  root.addEventListener("keydown", (e) => {
    if (e.key === " " && e.target === root) { e.preventDefault(); state.playing = !state.playing; state.touched = true; render(); if (state.playing && !timer) timer = setTimeout(tick, 1600); }
  });
  render();
  if (!reduce) timer = setTimeout(tick, 8000);
}

document.querySelectorAll('[data-demo="faces"]').forEach((n) => mount(n));
