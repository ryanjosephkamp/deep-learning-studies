// The seed-draw plate's view. The reader takes one training seed for each of the four configurations
// (or lets the plate step through every way to do it) and sees the three primary contrasts for that
// draw against all 81 draws and the study's estimate with its interval, from data/seeds.json. All
// values and sentences come from seeds-model.js.
import {
  drawCount, drawAt, drawIndex, all81, weights, axis, place, lanes, title, readout, rowLabel,
  detail, caption, legend, tally, drawName,
} from "./seeds-model.js?v=220f667d7d";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const usDate = (iso) => { const [y, m, d] = iso.split("-").map(Number); return `${MONTHS[m - 1]} ${d}, ${y}`; };
const STEP = 1100;   // ms between draws while playing
const START = 8000;  // ms on screen, untouched, before the plate starts on its own

const el = (tag, attrs = {}, kids = []) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const k of kids) n.append(k);
  return n;
};

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

// A small seeded generator for the Draw button, so ?seed=N repeats a sequence of draws.
function generator(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// What each configuration has, in words, from its two flags.
function has(cf) {
  if (cf.attention && cf.augmentation) return "attention and occlusion training";
  if (cf.attention) return "attention only";
  if (cf.augmentation) return "occlusion training only";
  return "neither idea";
}

export async function mount(root) {
  const data = await (await fetch(root.dataset.src)).json();
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const n = drawCount(data);
  const keys = data.contrasts.map((c) => c.key);
  const values = Object.fromEntries(keys.map((k) => [k, all81(data, k)]));
  const ax = axis(data);
  const param = new URLSearchParams(location.search).get("seed");
  const rand = generator(param !== null && /^\d+$/.test(param) ? Number(param) : Math.floor(Math.random() * 2 ** 32));
  // seen: the draws visited so far; the pale strips always show all of them, motion or not
  const state = { index: 0, playing: false, touched: false, focus: null, pinned: null, seen: new Set([0]) };

  if (!root.hasAttribute("tabindex")) root.tabIndex = 0;
  if (!root.hasAttribute("aria-label")) {
    const h = root.querySelector("h3");
    root.setAttribute("aria-label", `Live plate: ${h ? h.textContent : "seed draws"}`);
  }

  // The seeds: one radio group of chips per configuration
  const ask = el("p", { class: "sd-ask", text: "Take one training seed for each configuration" });
  const seeds = el("div", { class: "sd-seeds", role: "group" });
  ask.id = `sd-ask-${Math.random().toString(36).slice(2, 8)}`;
  seeds.setAttribute("aria-labelledby", ask.id);
  seeds.append(el("div", { class: "sd-seedhead", "aria-hidden": "true" }, [el("span"), el("span"), el("span", { text: "seed" })]));
  const cfgRows = data.configs.map((cf, k) => {
    const sign = el("span", { class: "sd-sign", "aria-hidden": "true" });
    const name = el("span", { class: "sd-cname" }, [el("span", { text: cf.name }), el("small", { text: has(cf) })]);
    const chips = el("div", { class: "sd-chips", role: "radiogroup", "aria-label": `${cf.name}: training seed` });
    for (const s of data.seeds) {
      const b = el("button", { type: "button", role: "radio", class: "sd-chip", "data-seed": s, "aria-label": `seed ${s}`, text: s });
      b.addEventListener("click", () => {
        touch();
        const draw = drawAt(data, state.index);
        draw[k] = s;
        setIndex(drawIndex(data, draw));
      });
      chips.append(b);
    }
    radioKeys(chips);
    const row = el("div", { class: "sd-cfg" }, [sign, name, chips]);
    seeds.append(row);
    return { sign, chips };
  });

  // Draw, play, and the count of draws shown
  const drawBtn = el("button", { type: "button", class: "sd-btn sd-draw", text: "Draw" });
  drawBtn.addEventListener("click", () => {
    touch();
    let i = Math.floor(rand() * n);
    if (i === state.index) i = (i + 1 + Math.floor(rand() * (n - 1))) % n;  // a new draw each press
    setIndex(i);
  });
  const play = el("button", { type: "button", class: "sd-btn sd-play" });
  play.addEventListener("click", () => setPlaying(!state.playing));
  const count = el("span", { class: "sd-tally" });
  const actions = el("div", { class: "sd-actions" }, [drawBtn, play, count]);

  // The number lines
  const lines = el("div", { class: "sd-lines" });
  const rows = {};
  for (const c of data.contrasts) {
    const y = lanes(values[c.key]);
    const track = el("div", { class: "sd-track", "aria-hidden": "true" });
    for (const at of [...ax.ticks.map((t) => t.at).filter((at) => at !== 0), 0]) {
      const g = el("span", { class: at === 0 ? "sd-zero" : "sd-grid" });
      g.style.left = `${place(ax, at)}%`;
      track.append(g);
    }
    const ci = el("span", { class: "sd-ci" });
    ci.style.left = `${place(ax, 100 * c.study.ci95_low)}%`;
    ci.style.width = `${place(ax, 100 * c.study.ci95_high) - place(ax, 100 * c.study.ci95_low)}%`;
    const est = el("span", { class: "sd-est" });
    est.style.left = `${place(ax, 100 * c.study.estimate)}%`;
    const cloud = el("span", { class: "sd-cloud" });
    const dots = values[c.key].map((v, i) => {
      const d = el("i", { class: "sd-dot" });
      d.style.left = `${place(ax, 100 * v)}%`;
      d.style.top = `calc(${y[i]} * var(--sd-band))`;  // the strip's height is set in book.css
      cloud.append(d);
      return d;
    });
    const nowLine = el("span", { class: "sd-now-line" });
    const now = el("span", { class: "sd-now" });
    track.append(cloud, ci, est, nowLine, now);
    const read = el("p", { class: "sd-read" });
    const row = el("div", { class: "sd-line", tabindex: "0", role: "group", "data-key": c.key },
      [el("p", { class: "sd-name", text: title(data, c.key) }), track, read]);
    // Focus, hover, or a tap shows how the comparison is built; a tap keeps it until another row is
    // tapped, so it stays on a touch screen.
    const show = () => { state.focus = c.key; renderFocus(); };
    const hide = () => {
      if (state.focus !== c.key || row.contains(document.activeElement)) return;
      state.focus = state.pinned;
      renderFocus();
    };
    row.addEventListener("focus", show); row.addEventListener("mouseenter", show);
    row.addEventListener("blur", hide); row.addEventListener("mouseleave", hide);
    row.addEventListener("click", () => { state.pinned = c.key; show(); });
    rows[c.key] = { row, dots, y, now, nowLine, read };
    lines.append(row);
  }
  const ticks = el("div", { class: "sd-ticks", "aria-hidden": "true" });
  for (const t of ax.ticks) {
    const s = el("span", { class: t.major ? "is-major" : "is-minor", text: t.at < 0 ? `−${-t.at}` : String(t.at) });
    s.style.left = `${place(ax, t.at)}%`;
    ticks.append(s);
  }
  lines.append(el("div", { class: "sd-axis", "aria-hidden": "true" }, [ticks, el("span", { class: "sd-unit", text: "points" })]));

  const detailP = el("p", { class: "sd-detail", "aria-live": "polite", text: "Tap or focus a comparison to see how it is built from the four configurations." });
  const legendP = el("p", { class: "sd-legend", text: legend(data) });
  const cap = el("p", { class: "sd-caption" });
  const checked = el("p", { class: "sd-checked" });
  fetch(root.dataset.src.replace(/seeds\.json(\?.*)?$/, "checks.json"), { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).then((c) => {
    if (!c) return;
    const a = document.createElement("a"); a.href = data.checker_url; a.textContent = "the site's checker";
    checked.replaceChildren(`Checked ${usDate(c.checked)}: ${c.passed.toLocaleString("en-US")} checks by `, a, ` of this page's values, and the numbers in its sentences, against the study's results; ${c.failed} failed.`);
  }).catch(() => {});

  const pick = el("div", { class: "sd-pick" }, [ask, seeds, actions]);
  root.querySelector(".plate-live-stage").append(el("div", { class: "sd-body" }, [pick, lines]), detailP, legendP, cap, checked);

  function renderFocus() {
    const w = state.focus ? weights(data, state.focus) : null;
    data.configs.forEach((cf, k) => { cfgRows[k].sign.textContent = w ? (w[cf.id] > 0 ? "+" : w[cf.id] < 0 ? "−" : "") : ""; });
    for (const key of keys) rows[key].row.classList.toggle("is-focus", key === state.focus);
    if (state.focus) detailP.textContent = detail(data, state.focus);
  }

  function render() {
    const draw = drawAt(data, state.index);
    data.configs.forEach((cf, k) => {
      for (const b of cfgRows[k].chips.querySelectorAll("button")) {
        const on = b.dataset.seed === draw[k];
        b.setAttribute("aria-checked", String(on));
        b.tabIndex = on ? 0 : -1;
      }
    });
    for (const key of keys) {
      const r = rows[key];
      const v = values[key][state.index];
      r.now.style.left = r.nowLine.style.left = `${place(ax, 100 * v)}%`;
      r.now.style.top = `calc(${r.y[state.index]} * var(--sd-band))`;
      r.dots.forEach((d, i) => { d.classList.toggle("is-seen", state.seen.has(i)); });
      r.read.replaceChildren(...readout(data, key, draw).map((p, i) =>
        el("span", { class: i === 0 ? "sd-r sd-r-now" : "sd-r" }, [el("i", { text: p.label }), " ", el("b", { text: p.value })])));
      r.row.setAttribute("aria-label", rowLabel(data, key, draw));
    }
    count.textContent = tally(state.seen.size, n);
    play.textContent = state.playing ? "Pause" : "Play";
    play.setAttribute("aria-pressed", String(state.playing));
    play.setAttribute("aria-label", state.playing ? "Pause the draws" : `Step through all ${n} draws`);
    const quiet = state.playing && !state.touched;
    cap.setAttribute("aria-live", quiet ? "off" : "polite");
    cap.textContent = caption(data, draw);
    seeds.setAttribute("aria-description", drawName(data, draw));
  }

  function setIndex(i) {
    state.index = i;
    state.seen.add(i);
    render();
  }

  // Alive: once on screen and untouched for a while, it steps through every draw in order and keeps
  // going; it pauses off screen, on the Pause button, and on the space bar.
  let visible = false, startTimer = 0, stepTimer = 0;
  function touch() {
    state.touched = true;
    clearTimeout(startTimer); startTimer = 0;
    if (state.playing) setPlaying(false);
  }
  function setPlaying(on) {
    state.touched = true;
    clearTimeout(startTimer); startTimer = 0;
    state.playing = on;
    clearTimeout(stepTimer); stepTimer = 0;
    if (on) { setIndex((state.index + 1) % n); stepTimer = setTimeout(step, STEP); }
    else render();
  }
  function step() {
    stepTimer = 0;
    if (!state.playing) return;
    if (visible && !document.hidden) setIndex((state.index + 1) % n);
    stepTimer = setTimeout(step, STEP);
  }
  function autoStart() {
    startTimer = 0;
    if (state.touched || !visible || reduce) return;
    state.playing = true;
    render();
    stepTimer = setTimeout(step, STEP / 2);
  }
  new IntersectionObserver((es) => {
    visible = es[0].isIntersecting;
    if (visible && !state.touched && !state.playing && !reduce && !startTimer) startTimer = setTimeout(autoStart, START);
    if (!visible && startTimer) { clearTimeout(startTimer); startTimer = 0; }
  }, { threshold: 0.25 }).observe(root);
  root.addEventListener("keydown", (e) => {
    if (e.key === " " && e.target === root) { e.preventDefault(); setPlaying(!state.playing); }
  });

  render();
}

document.querySelectorAll('[data-demo="seeds"]').forEach((n) => mount(n));
