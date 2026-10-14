"""Objective style-adherence metrics.

This is what turns the demo from "squint at two images" into a number on a
slide. Three metrics, each mapping to one house-style rule:

  palette_score    fraction of pixels within tolerance of a palette colour
  flatness_score   fraction of pixels in low-gradient (flat fill) regions
  shadow_present   whether the single shadow colour appears at plausible mass

style_score is the mean of the three. Training images score ~1.0 by
construction, which validates the metric. The interesting numbers are for
generated images.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))

from house_style import PALETTE, PALETTE_LIST  # noqa: E402

PAL = np.array(PALETTE_LIST, dtype=np.int16)


def palette_score(img: Image.Image, tol: int = 28) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.int16).reshape(-1, 3)
    # Chunked nearest-colour distance to keep memory sane.
    best = np.full(a.shape[0], 1e9)
    for c in PAL:
        d = np.abs(a - c).sum(axis=1)
        best = np.minimum(best, d)
    return float((best <= tol * 3).mean())


def flatness_score(img: Image.Image, thresh: int = 10) -> float:
    g = np.asarray(img.convert("L"), dtype=np.int16)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    grad = gx + gy
    return float((grad <= thresh).mean())


def shadow_present(img: Image.Image, tol: int = 36) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    sc = np.array(PALETTE["shadow"], dtype=np.int16)
    mask = (np.abs(a - sc).sum(axis=2) <= tol * 3)
    frac = mask.mean()
    # Expect a visible but not dominant shadow: 0.5%-20% of the canvas.
    if frac < 0.005:
        return 0.0
    if frac > 0.20:
        return 0.5
    return 1.0


def score_image(path: Path) -> dict:
    img = Image.open(path)
    p = palette_score(img)
    f = flatness_score(img)
    s = shadow_present(img)
    return {
        "file": path.name,
        "palette": round(p, 4),
        "flatness": round(f, 4),
        "shadow": round(s, 4),
        "style_score": round((p + f + s) / 3, 4),
    }


def score_dir(d: Path) -> list:
    rows = [score_image(p) for p in sorted(d.glob("*.png"))]
    return rows


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parents[1] / "data" / "train"

    rows = score_dir(target)
    if not rows:
        print(f"No PNGs found in {target}")
        sys.exit(1)

    print(f"{'file':<40} {'palette':>8} {'flat':>8} {'shadow':>8} {'style':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r['file']:<40} {r['palette']:>8.3f} {r['flatness']:>8.3f} "
              f"{r['shadow']:>8.3f} {r['style_score']:>8.3f}")

    mean = sum(r["style_score"] for r in rows) / len(rows)
    print("-" * 76)
    print(f"{'MEAN':<40} {'':>8} {'':>8} {'':>8} {mean:>8.3f}")
