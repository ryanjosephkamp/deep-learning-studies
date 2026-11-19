// The book's page script: the day and night printing, and lifting a plate off the page to look
// closer (pinch, scroll, or drag to zoom and pan; Escape or the close button to put it back), and
// which way the page turns between pages.
(function () {
  var root = document.documentElement;
  var KEY = "course-book-theme";
  var mem = null;  // the choice, kept here too, so the switch still works where storage is blocked
  function stored() { try { var v = localStorage.getItem(KEY); return v === null ? mem : v; } catch (e) { return mem; } }
  function store(v) { mem = v; try { if (v) localStorage.setItem(KEY, v); else localStorage.removeItem(KEY); } catch (e) {} }
  function apply(v) { if (v) root.setAttribute("data-theme", v); else root.removeAttribute("data-theme"); }
  apply(stored());

  // Turning the page: the page being left records its place in the book, and the page arriving turns
  // forward or back from it. A reload or a move within one page doesn't turn.
  var LEAF = "course-book-leaf";
  window.addEventListener("pageswap", function () {
    try { sessionStorage.setItem(LEAF, root.getAttribute("data-leaf")); } catch (e) {}
  });
  window.addEventListener("pagereveal", function (e) {
    if (!e.viewTransition) return;
    e.viewTransition.ready.catch(function () {});
    e.viewTransition.finished.catch(function () {});
    var from = null;
    try { from = sessionStorage.getItem(LEAF); sessionStorage.removeItem(LEAF); } catch (err) {}
    var to = root.getAttribute("data-leaf");
    if (from === null || to === null || from === to) { e.viewTransition.skipTransition(); return; }
    e.viewTransition.types.add(Number(to) > Number(from) ? "forward" : "back");
  });

  document.addEventListener("DOMContentLoaded", function () {
    var mode = document.querySelector(".mode");
    // A page restored from the back-forward cache takes up any choice made on another page meanwhile.
    window.addEventListener("pageshow", function (e) { if (e.persisted) { apply(stored()); if (mode && mode.relabel) mode.relabel(); } });
    if (mode) {
      var label = function () {
        var v = stored() || "auto";
        mode.textContent = v === "dark" ? "☾" : v === "light" ? "☀" : "◐";
        mode.setAttribute("aria-label", "Printing: " + (v === "auto" ? "follows the system" : v === "dark" ? "night" : "day") + ". Change it.");
      };
      label();
      mode.relabel = label;
      mode.addEventListener("click", function () {
        var v = stored() || "auto";
        var next = v === "auto" ? "light" : v === "light" ? "dark" : null;
        store(next); apply(next); label();
      });
    }

    // Lifting a plate
    var box = document.querySelector(".lifted");
    if (!box) return;
    var sheet = box.querySelector(".sheet"), img = document.createElement("img"), title = box.querySelector(".title");
    img.draggable = false;
    var foot = box.querySelector(".lift-cap"), close = box.querySelector(".close"), opener = null;
    var s = 1, x = 0, y = 0, pts = new Map(), last = null;
    function set() { sheet.style.transform = "translate(" + x + "px," + y + "px) scale(" + s + ")"; }
    function reset() { s = 1; x = 0; y = 0; set(); }
    function open(fig) {
      var src = Array.prototype.find.call(fig.querySelectorAll("button.lift img"), function (i) { return i.offsetParent !== null; }) || fig.querySelector("img");
      img.src = src.currentSrc || src.src; img.alt = src.alt;
      if (!img.parentNode) sheet.append(img);
      title.textContent = fig.querySelector("figcaption b").textContent;
      foot.textContent = fig.querySelector("figcaption .cap").textContent;
      opener = fig.querySelector("button.lift");
      box.hidden = false; reset(); close.focus();
      document.body.style.overflow = "hidden";
    }
    function shut() { box.hidden = true; document.body.style.overflow = ""; if (opener) opener.focus(); }
    document.querySelectorAll(".plate > button.lift").forEach(function (b) {
      b.addEventListener("click", function () { open(b.closest(".plate")); });
    });
    close.addEventListener("click", shut);
    // Keys: + and - zoom about the center, arrows pan once zoomed, 0 resets, Escape puts the plate back.
    function zoomBy(k) {
      var r = sheet.getBoundingClientRect(), cx = window.innerWidth / 2, cy = window.innerHeight / 2;
      var ns = Math.min(6, Math.max(1, s * k)), f = ns / s;
      x = cx - (cx - r.left) * f - (r.left - x); y = cy - (cy - r.top) * f - (r.top - y); s = ns;
      if (s === 1) { x = 0; y = 0; }
      set();
    }
    box.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { shut(); return; }
      if (e.key === "Tab") { e.preventDefault(); close.focus(); return; }
      if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomBy(1.25); }
      else if (e.key === "-" || e.key === "_") { e.preventDefault(); zoomBy(0.8); }
      else if (e.key === "0") { e.preventDefault(); reset(); }
      else if (s > 1 && e.key.indexOf("Arrow") === 0) {
        e.preventDefault();
        var step = 60;
        if (e.key === "ArrowLeft") x += step; else if (e.key === "ArrowRight") x -= step; else if (e.key === "ArrowUp") y += step; else y -= step;
        set();
      }
    });
    box.querySelector(".stage").addEventListener("click", function (e) { if (e.target === e.currentTarget && s === 1) shut(); });
    box.addEventListener("wheel", function (e) {
      e.preventDefault();
      var k = Math.exp(-e.deltaY * 0.0015), r = sheet.getBoundingClientRect();
      var ns = Math.min(6, Math.max(1, s * k)), f = ns / s;
      x = e.clientX - (e.clientX - r.left) * f - (r.left - x); y = e.clientY - (e.clientY - r.top) * f - (r.top - y); s = ns;
      if (s === 1) { x = 0; y = 0; }
      set();
    }, { passive: false });
    sheet.addEventListener("pointerdown", function (e) { sheet.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); last = null; });
    sheet.addEventListener("pointermove", function (e) {
      if (!pts.has(e.pointerId)) return;
      var prev = pts.get(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 1) { if (s > 1) { x += e.clientX - prev[0]; y += e.clientY - prev[1]; set(); } return; }
      var p = Array.from(pts.values()), d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
      var cx = (p[0][0] + p[1][0]) / 2, cy = (p[0][1] + p[1][1]) / 2;
      if (last) {
        var ns = Math.min(6, Math.max(1, s * d / last.d)), f = ns / s, r = sheet.getBoundingClientRect();
        x = cx - (cx - r.left) * f - (r.left - x) + (cx - last.cx); y = cy - (cy - r.top) * f - (r.top - y) + (cy - last.cy); s = ns;
        if (s === 1) { x = 0; y = 0; }
        set();
      }
      last = { d: d, cx: cx, cy: cy };
    });
    var up = function (e) { pts.delete(e.pointerId); last = null; };
    sheet.addEventListener("pointerup", up); sheet.addEventListener("pointercancel", up);
    sheet.addEventListener("dblclick", reset);
  });
})();
