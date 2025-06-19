#!/usr/bin/env python
# make_occ_heat.py  –  build occ_heat JSONL + visual for one poster
# ---------------------------------------------------------------
import json, numpy as np, matplotlib.pyplot as plt, pathlib, sys
from PIL import Image, ImageDraw, ImageFilter

# ── hard-coded locations ──────────────────────────────────────────
IMG_PATH = r"E:\SIA_works\PosterDatabase\PosterDataset\3D\3D-3.png"
ANN_PATH = r"E:\SIA_works\PosterDatabase\Label_Studio\annotations_split\annotation_306_153.json"

OUT_JSONL = pathlib.Path(IMG_PATH).with_suffix(".occ.jsonl")   # same folder, .occ.jsonl
OUT_PNG   = pathlib.Path(IMG_PATH).with_suffix(".occ.png")     # visual grids

HX, HY = 12, 21           # grid resolution
INK_THRESH = 245          # gray < 245 counts as “ink”
ALPHA = 0.6               # weight of ink grid
BETA  = 0.9               # weight of text-occupancy grid
FLOOR = 0.2               # minimum free-score (never zero)

# ── helper: down-sample an array to HX×HY by averaging ───────────
def downsample(arr):
    H, W = arr.shape
    g = np.zeros((HY, HX))
    for gy in range(HY):
        for gx in range(HX):
            y0,y1 = int(gy*H/HY), int((gy+1)*H/HY)
            x0,x1 = int(gx*W/HX), int((gx+1)*W/HX)
            g[gy, gx] = arr[y0:y1, x0:x1].mean()
    return g

# ── 1. ink ratio grid (graphics density) ─────────────────────────
img   = Image.open(IMG_PATH).convert("L")          # grayscale PIL Image
W, H  = img.size
# ── 1. ink ratio grid (graphics density) ─────────────────────────
# NEW: use edge density instead of gray-threshold
try:
    import cv2                             # OpenCV available?
    edges = cv2.Canny(np.array(img), 50, 120)      # 0/255
    ink_mask = edges.astype("float32") / 255.0
except ImportError:
    # fallback: PIL edge filter (coarser but avoids cv2 dependency)
    edges = img.filter(ImageFilter.FIND_EDGES)
    ink_mask = np.array(edges, dtype="float32") / 255.0

# soften so a few edges don’t dominate an entire cell
ink_blur = Image.fromarray((ink_mask*255).astype("uint8")).filter(
              ImageFilter.BoxBlur(3))
ink_g    = downsample(np.array(ink_blur)/255.0)        # 0–1 per cell


# ── 2. text-occupancy grid from Label-Studio rectangles ──────────
with open(ANN_PATH, encoding="utf-8") as f:
    data = json.load(f)

if "annotation" in data:                     # your file structure
    results = data["annotation"]["result"]
elif "result" in data:                       # single-task JSON
    results = data["result"]
else:                                        # bulk export
    results = data.get("annotations", [{}])[0].get("result", [])
text_mask = Image.new("1", (W, H), 0)
d = ImageDraw.Draw(text_mask)
for r in results:
    v   = r["value"]
    x0p,y0p = v["x"], v["y"]
    w_pct,h_pct = v["width"], v["height"]
    x0 = x0p/100*W;  y0 = y0p/100*H
    x1 = (x0p+w_pct)/100*W;  y1 = (y0p+h_pct)/100*H
    d.rectangle([x0,y0,x1,y1], fill=1)             # mark text area
txt_g = downsample(np.array(text_mask, dtype="float32"))

# ── 3. blend → occ_heat (1 = easiest) ─────────────────────────────
penalty = np.clip(ALPHA*ink_g + BETA*txt_g, 0, 1)
occ     = FLOOR + (1-FLOOR)*(1 - penalty)          # keep min floor

# ── 4. write JSONL line ───────────────────────────────────────────
flat   = " ".join(f"{v:.1f}" for v in occ.flatten())
with OUT_JSONL.open("w", encoding="utf-8") as f:
    f.write(json.dumps({"occ_heat": flat}) + "\n")
print(f"★ JSONL written → {OUT_JSONL}")

# ── 5. save visual grids ─────────────────────────────────────────
fig, axes = plt.subplots(1,3, figsize=(9,3), dpi=120)
for ax,mat,title,cmap in [
        (axes[0], ink_g, "Ink ratio", "Reds"),
        (axes[1], txt_g, "Text occupancy", "Blues"),
        (axes[2], occ,   "Combined occ_heat", "viridis")]:
    im = ax.imshow(mat, origin='upper', cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches='tight')
print(f"★ Visual grid saved → {OUT_PNG}")
