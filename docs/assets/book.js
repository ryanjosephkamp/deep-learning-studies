// The book's page script: the day and night printing, the thumb index's flags (and, on a touch screen,
// scrubbing through them), lifting a plate off the page to look closer (pinch, scroll, or drag to zoom
// and pan; Escape or the close button to put it back), and which way the page turns between pages.
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

    // The thumb index: Contents, the chapter, and the chapter's sections. A flag names the tab under the
    // pointer or the keyboard's focus; on the chapter's page, the tab of the section being read stands out.
    var edge = document.querySelector("nav.edge");
    if (edge) {
      var flag = document.createElement("span");
      flag.className = "edge-flag"; flag.setAttribute("aria-hidden", "true"); flag.hidden = true;
      edge.appendChild(flag);
      var said = null;
      // the flag's words: the chapter's numeral or a section's fleuron in red, then the name (rebuilt only
      // when they change)
      var say = function (text) {
        if (text === said) return;
        said = text;
        var parts = text.split(" · ");
        flag.textContent = "";
        if (parts.length > 1) {
          var no = document.createElement("span");
          no.className = "flag-no"; no.textContent = parts[0];
          flag.appendChild(no);
          flag.appendChild(document.createTextNode(parts.slice(1).join(" · ")));
        } else flag.textContent = text;
      };
      var show = function (a) {
        if (!a || !a.dataset.flag) return;
        var r = a.getBoundingClientRect();
        say(a.dataset.flag); flag.hidden = false;
        flag.style.top = (r.top + r.height / 2) + "px";
      };
      // A new keyboard focus wins over a pointer parked on another tab; while neither changes, the pointer's
      // tab wins, then the focused tab's. The pointer can rest on the flag itself. Escape puts the flag away,
      // and it stays away until the pointer or the focus reaches another tab.
      var hovered = null, focused = null, latched = false;
      var update = function () {
        var a = hovered || focused;
        if (a && !latched) show(a); else flag.hidden = true;
      };
      var hover = function (a) { if (a === hovered) return; hovered = a; if (a) latched = false; update(); };
      edge.addEventListener("mouseover", function (e) { if (flag.contains(e.target)) return; hover(e.target.closest("a")); });
      edge.addEventListener("mouseleave", function () { hover(null); });
      edge.addEventListener("focusin", function (e) { var a = e.target.closest("a"); if (a !== focused) latched = false; focused = a; hovered = null; update(); });
      edge.addEventListener("focusout", function () { focused = null; update(); });
      document.addEventListener("keydown", function (e) {
        if (e.key === "Escape" && !flag.hidden && !scrub) { latched = true; flag.hidden = true; }
      });

      // A thumb on the edge: press a tab to see its name beside the thumb, slide along the edge to run through
      // the tabs, and let go to go to the one named. Slide away from the edge first to stay where you are. A
      // second finger calls the scrub off.
      var tabs = Array.prototype.slice.call(edge.querySelectorAll("a")), scrub = null, quiet = null;
      var touchy = matchMedia("(any-pointer: coarse)");
      var under = function (y) {
        for (var i = 0; i < tabs.length; i++) if (y < tabs[i].getBoundingClientRect().bottom) return tabs[i];
        return tabs[tabs.length - 1];
      };
      var pick = function (a, y) {
        if (a !== scrub.on) {
          // a tick as the thumb crosses a tab, where the phone allows it (only once the page has been touched)
          if (scrub.on) { scrub.on.classList.remove("is-on"); if (navigator.vibrate && navigator.userActivation && navigator.userActivation.hasBeenActive) { try { navigator.vibrate(3); } catch (x) {} } }
          scrub.on = a; a.classList.add("is-on");
        }
        say(scrub.off ? "Let go to stay here" : a.dataset.flag);
        flag.hidden = false;
        flag.style.top = Math.max(84, Math.min(innerHeight - 40, y - 56)) + "px";
      };
      // a tab within this page (the chapter's top or a section) always goes; another page's tab goes unless
      // it is this page
      var go = function (a) {
        var href = a.getAttribute("href");
        if (href.charAt(0) === "#") {
          // a second press on the tab of the place already in the address still takes the reader there
          var t = document.getElementById(href.slice(1));
          if (location.hash === href && t) t.scrollIntoView(); else location.hash = href;
        } else if (a.getAttribute("aria-current") !== "page") location.href = a.href;
      };
      // The click that the scrub's own touch makes as it lifts would only repeat the scrub's decision, so that
      // one click is dropped: a click from the same pointer (or, where the browser doesn't say, any pointer
      // click) within 700 ms of the release. A click from a key or a screen reader (detail 0), or from another
      // pointer, is the link's own.
      var end = function (go_, e) {
        if (!scrub) return;
        var a = scrub.on, off = scrub.off, id = scrub.id;
        scrub = null;
        if (a) a.classList.remove("is-on");
        flag.classList.remove("is-scrub", "is-off");
        flag.hidden = true; latched = false;
        quiet = { id: id, until: Date.now() + 700 };
        if (go_ && a && !off) go(a);
      };
      var mine = function (e) { return scrub && e.pointerId === scrub.id; };
      edge.addEventListener("pointerdown", function (e) {
        if (scrub) {
          // a second contact during a scrub calls it off, and its own click is dropped too
          if (e.pointerId !== scrub.id) { e.preventDefault(); end(false); quiet = { id: null, until: Date.now() + 700 }; }
          return;
        }
        // only a finger or a pen on a tab, and only where the strip is a touch strip (it doesn't scroll the page
        // there); a mouse, or a touch elsewhere, is a plain click
        if (e.pointerType === "mouse" || !e.isPrimary || !touchy.matches || !e.target.closest("a")) return;
        e.preventDefault();
        try { edge.setPointerCapture(e.pointerId); } catch (x) {}
        scrub = { id: e.pointerId, on: null, off: false };
        flag.classList.add("is-scrub");
        pick(e.target.closest("a"), e.clientY);
      });
      var track = function (e) {
        var off = e.clientX < innerWidth - 120;
        if (off !== scrub.off) { scrub.off = off; flag.classList.toggle("is-off", off); }
        pick(under(e.clientY), e.clientY);
      };
      // listened for on the window too, so a scrub whose capture failed still ends when the finger lifts
      window.addEventListener("pointermove", function (e) { if (mine(e)) track(e); });
      // the thumb's own release decides: where it lifts, not where it last moved
      window.addEventListener("pointerup", function (e) { if (!mine(e)) return; track(e); end(true, e); });
      window.addEventListener("pointercancel", function (e) { if (mine(e)) end(false, e); });
      edge.addEventListener("lostpointercapture", function (e) { if (mine(e)) end(false, e); });
      window.addEventListener("blur", function () { end(false); });
      window.addEventListener("pagehide", function () { end(false); });
      // a long press during a scrub doesn't open the link's menu; a mouse's right click still does
      edge.addEventListener("contextmenu", function (e) { if (scrub) e.preventDefault(); });
      edge.addEventListener("click", function (e) {
        if (!quiet || Date.now() > quiet.until || e.detail === 0) return;
        if (quiet.id !== null && typeof e.pointerId === "number" && e.pointerId > 0 && e.pointerId !== quiet.id) return;
        e.preventDefault(); quiet = null;
      });

      // On the chapter's page, the section being read: the last whose heading has passed the reading line, a
      // third of the way down the screen (never above where a jump to a heading leaves it), or the last
      // section once the page can scroll no further. Its tab stands out and is announced as the current location.
      var marks = tabs.filter(function (a) { return a.dataset.sec && a.getAttribute("href").charAt(0) === "#"; })
        .map(function (a) { return { a: a, h: document.getElementById(a.dataset.sec) }; })
        .filter(function (m) { return m.h; });
      if (marks.length) {
        var shown = null, queued = false;
        var spy = function () {
          queued = false;
          var line = Math.max(innerHeight / 3, 72), now = null;
          for (var i = 0; i < marks.length; i++) if (marks[i].h.getBoundingClientRect().top < line) now = marks[i];
          var last = marks[marks.length - 1], doc = document.documentElement;
          if (scrollY > 0 && scrollY + innerHeight >= doc.scrollHeight - 2 && last.h.getBoundingClientRect().top < innerHeight) now = last;
          if (now === shown) return;
          if (shown) shown.a.removeAttribute("aria-current");
          if (now) now.a.setAttribute("aria-current", "location");
          shown = now;
        };
        var later = function () { if (!queued) { queued = true; setTimeout(spy, 80); } };
        window.addEventListener("scroll", later, { passive: true });
        window.addEventListener("resize", later);
        window.addEventListener("hashchange", later);
        window.addEventListener("pageshow", later);
        // a demo that loads, or a font that arrives, can move the headings without a scroll
        if (window.ResizeObserver) new ResizeObserver(later).observe(document.body);
        spy();
      }
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
