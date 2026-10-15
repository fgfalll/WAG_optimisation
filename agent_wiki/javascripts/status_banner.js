/* ──────────────────────────────────────────────────────────────────────────
   Status banner — injected into Material's `announce` slot.

   WHY A SCRIPT AND NOT `theme.banner`:
   mkdocs.yml declares `theme.banner`, which Material only wires up from
   v9.6. The repository pins `mkdocs-material==9.5.*` (see
   .github/workflows/docs.yml), where the `announce` block in
   templates/base.html is empty. Rather than change the pinned toolchain,
   we inject into the slot that already exists:

       <div data-md-component="announce"> ... </div>

   TO UPGRADE LATER: bump the pin to `mkdocs-material==9.6.*` or `9.7.*`,
   then delete this file and the `theme.banner` key from mkdocs.yml.
   Keep the wording identical so the site does not shift.

   Absolute URLs are used deliberately — the banner renders on pages at many
   depths, so a relative link would resolve against each page's directory.
   ────────────────────────────────────────────────────────────────────────── */
(function () {
  "use strict";

  var BANNER_HTML =
    '<strong>Under heavy active development.</strong> ' +
    "A <strong>3D THMC reservoir simulator is being built from scratch in Rust</strong> " +
    '— <a href="https://fgfalll.github.io/WAG_optimisation/compositional/">no engine code exists yet</a>. ' +
    "Pages describing it are <strong>specifications and audit trails, not reports of running software</strong>. " +
    "The <strong>Python engine</strong> pages describe the shipped surrogate.";

  function inject() {
    var host = document.querySelector('[data-md-component="announce"]');
    if (!host) return false;
    // Idempotent: instant theme switching re-renders the slot.
    if (host.querySelector(".co2eor-status-banner")) return true;

    var aside = document.createElement("aside");
    aside.className = "md-banner co2eor-status-banner";
    aside.setAttribute("role", "status");

    var inner = document.createElement("div");
    inner.className = "md-banner__inner md-grid md-typeset";
    inner.innerHTML = "<p>" + BANNER_HTML + "</p>";

    aside.appendChild(inner);
    host.appendChild(aside);
    return true;
  }

  if (!inject()) {
    // Material swaps the announce slot on instant-navigation; observe it.
    var observer = new MutationObserver(function () {
      if (inject()) observer.disconnect();
    });
    document.addEventListener("DOMContentLoaded", function () {
      observer.observe(document.body, { childList: true, subtree: true });
    });
  }
})();