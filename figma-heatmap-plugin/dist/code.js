"use strict";
/* ------------------------------------------------------------------
   Figma ‑ Heat‑Layout prototype plugin (code.ts)
   ---------------------------------------------------------------
   ▸ Select exactly ONE frame then run the plugin.
   ▸ The script builds a 12×21 occupancy grid (“image_heat”),
     calls your backend (Python layout_predict API) and stores the
     six returned heat‑maps as plugin data on that frame.
   ▸ No UI – progress & errors reported with figma.notify / console.
------------------------------------------------------------------*/
// ───────────────────────────────────────────────────────── helpers ───
const GRID_ROWS = 12;
const GRID_COLS = 21;
const BACKEND_URL = "http://localhost:8000/predict"; // ← change if needed
/** Clamp n into [min,max]. */
function clamp(n, min = 0, max = 1) {
    return Math.min(Math.max(n, min), max);
}
/**
 * Build a 12×21 occupancy grid for all visible, non‑text fills in the frame.
 * Very naïve: marks a cell as 1 if ANY node’s bounding box intersects it.
 */
function buildImageHeat(frame) {
    const grid = Array.from({ length: GRID_ROWS }, () => new Array(GRID_COLS).fill(0));
    const fw = frame.width, fh = frame.height;
    frame.findAll(n => n.visible && "x" in n && "width" in n).forEach(node => {
        const bbox = node;
        const x0 = clamp((bbox.x - frame.x) / fw);
        const y0 = clamp((bbox.y - frame.y) / fh);
        const x1 = clamp((bbox.x + bbox.width - frame.x) / fw);
        const y1 = clamp((bbox.y + bbox.height - frame.y) / fh);
        const c0 = Math.floor(x0 * GRID_COLS);
        const r0 = Math.floor(y0 * GRID_ROWS);
        const c1 = Math.ceil(x1 * GRID_COLS);
        const r1 = Math.ceil(y1 * GRID_ROWS);
        for (let r = r0; r < r1; r++)
            for (let c = c0; c < c1; c++)
                if (r >= 0 && r < GRID_ROWS && c >= 0 && c < GRID_COLS)
                    grid[r][c] = 1;
    });
    return grid;
}
/** Flatten 2‑D grid → single space‑separated string */
function gridToString(g) {
    return g.flat().map(v => v.toFixed(1)).join(" ");
}
/** POST to backend and return JSON */
async function predictLayout(prompt) {
    const res = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
    });
    if (!res.ok)
        throw new Error(`Backend HTTP ${res.status}`);
    return res.json();
}
// ───────────────────────────────────────────────────────── main ─────
(async () => {
    // 1️⃣ Selection check -------------------------------------------------
    const sel = figma.currentPage.selection;
    if (sel.length !== 1 || sel[0].type !== "FRAME") {
        figma.notify("❌  Please select exactly one FRAME and run again.");
        figma.closePlugin();
        return;
    }
    const frame = sel[0];
    // 2️⃣ Build prompt ----------------------------------------------------
    const imageHeat = buildImageHeat(frame);
    const prompt = `FRAME_PCT ${frame.width} ${frame.height}\nimage_heat ${gridToString(imageHeat)}`;
    // 3️⃣ Call backend ----------------------------------------------------
    figma.notify("☎️  Calling layout_predict backend…", { timeout: 3000 });
    let data;
    try {
        data = await predictLayout(prompt);
    }
    catch (err) {
        console.error(err);
        figma.notify("❌  Backend error – see console.");
        figma.closePlugin();
        return;
    }
    // 4️⃣ Validate & store result ----------------------------------------
    const KEYS = [
        "title_heat", "location_heat", "time_heat",
        "host_organization_heat", "call_to_action_purpose_heat",
        "text_descriptions_details_heat"
    ];
    for (const k of KEYS) {
        if (!(k in data)) {
            figma.notify(`⚠️  Backend missing ${k}`);
            continue;
        }
        // attach as plugin‑data (string)
        frame.setPluginData(k, Array.isArray(data[k]) ? data[k].join(" ") : String(data[k]));
    }
    figma.notify("✅  Heat‑maps stored in plugin data!", { timeout: 4000 });
    figma.closePlugin();
})();
