// The chapter hero's miniature: a centered black square grows over an aligned face through the sizes
// the study ran, and two bars show ArcFace's and SFace's measured rank-1 at each size. The page prints
// the first step, so it reads without this script. It runs on its own while on screen, pauses when off
// screen or hidden, has a Pause button, and with reduced motion holds one still frame.
const fig = document.querySelector(".hero-plate");
if (fig) {
  const data = JSON.parse(fig.querySelector(".hero-data").textContent);
  const sq = fig.querySelector(".hero-sq");
  const sizeOut = fig.querySelector(".hs-v");
  const pause = fig.querySelector(".hero-pause");
  const bars = [...fig.querySelectorAll(".hero-bar")].map((p) => ({ m: p.dataset.m, fill: p.querySelector(".hb-f"), val: p.querySelector(".hb-v") }));
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const still = data.steps.findIndex((s) => s.size === 20);

  function show(i) {
    const s = data.steps[i];
    const [x, y, w, h] = s.rect || [data.crop / 2, data.crop / 2, 0, 0];
    sq.setAttribute("x", x); sq.setAttribute("y", y); sq.setAttribute("width", w); sq.setAttribute("height", h);
    sizeOut.textContent = `${s.size}%`;
    for (const b of bars) {
      b.fill.style.width = `${s[b.m + "v"]}%`;
      b.val.textContent = `${s[b.m]}%`;
    }
    fig.dataset.step = String(i);
  }

  let i = 0, timer = 0, onScreen = false, paused = false;
  const hold = (k) => (data.steps[k].size === 0 ? 2200 : data.steps[k].size === data.steps[data.steps.length - 1].size ? 2000 : 1150);
  function tick() {
    timer = 0;
    if (!onScreen || document.hidden || paused) return;
    i = (i + 1) % data.steps.length;
    show(i);
    timer = setTimeout(tick, hold(i));
  }
  function wake() { if (!timer && onScreen && !document.hidden && !paused && !reduce) timer = setTimeout(tick, hold(i)); }

  if (reduce) {
    show(still < 0 ? 0 : still);
    pause.hidden = true;
  } else {
    show(0);
    new IntersectionObserver((es) => { onScreen = es[0].isIntersecting; if (onScreen) wake(); else { clearTimeout(timer); timer = 0; } }, { threshold: 0.2 }).observe(fig);
    document.addEventListener("visibilitychange", wake);
    pause.addEventListener("click", () => {
      paused = !paused;
      pause.textContent = paused ? "Play" : "Pause";
      pause.setAttribute("aria-pressed", String(paused));
      if (paused) { clearTimeout(timer); timer = 0; } else wake();
    });
  }
}
