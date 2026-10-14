"""Build the v1-vs-v2-vs-ground-truth comparison grid.

This is the visual for the strongest part of the story: the evaluation
harness catching a regression. Three rows, same six holdout subjects:

  row 1  ground truth (the style bible)
  row 2  v1 (approved)
  row 3  v2 (rejected, mode collapsed)

Row 3 is visibly the same image six times, which lands mode collapse faster
than any number. Reads from saved PNGs, so no endpoint calls.
"""

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "09_evaluate_holdout.py")
ev = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ["x", "--endpoint", "d"]
spec.loader.exec_module(ev)
sys.argv = _argv

HOLDOUT = ROOT / "data" / "holdout"
RAW = ROOT / "outputs" / "raw"
OUT = ROOT / "outputs"

CELL = 320
LABEL_H = 46
ROW_HEADER_H = 40


def font(size):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def cell(img, caption, colour=(20, 20, 20)):
    img = img.resize((CELL, CELL), Image.LANCZOS)
    out = Image.new("RGB", (CELL, CELL + LABEL_H), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((8, CELL + 10), caption, fill=colour, font=font(20))
    return out


def main():
    cases = json.loads((HOLDOUT / "holdout.json").read_text(encoding="utf-8"))
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    per = {r["subject"]: r for r in results["version_comparison"]["per_subject"]}

    rows = []
    for label, loader, colour in [
        ("GROUND TRUTH  (style bible, never trained on)",
         lambda c: Image.open(HOLDOUT / c["file_name"]), (20, 90, 20)),
        ("v1  APPROVED  (LoRA on attention)",
         lambda c: Image.open(RAW / f"vercmp_v1_{c['subject'].replace(' ', '_')}.png"),
         (20, 20, 20)),
        ("v2  REJECTED  (LoRA on attention + conv/projection)",
         lambda c: Image.open(RAW / f"vercmp_v2_{c['subject'].replace(' ', '_')}.png"),
         (150, 30, 30)),
    ]:
        cells = []
        for c in cases:
            img = loader(c)
            subj = c["subject"]
            cov = ev.subject_coverage(img)
            if label.startswith("GROUND"):
                cap = f"{subj}  cov {cov:.2f}"
            elif label.startswith("v1"):
                cap = f"sty {per[subj]['v1_style_score']:.2f}  cov {cov:.2f}"
            else:
                cap = f"sty {per[subj]['v2_style_score']:.2f}  cov {cov:.2f}"
            cells.append(cell(img, cap, colour))
        rows.append((label, cells, colour))

    cols = len(cases)
    # Legend band: "cov" is target-matching, not higher-is-better, so without
    # this a viewer can read cov 0.83 as a good score.
    legend = ("sty = style score, 0-1, higher is better.   "
              "cov = subject coverage, fraction of canvas filled - "
              "TARGET is ~0.15, not higher.")
    legend_h = 32

    row_h = CELL + LABEL_H + ROW_HEADER_H
    sheet = Image.new("RGB", (CELL * cols, legend_h + row_h * len(rows)),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)

    d.rectangle([(0, 0), (sheet.width, legend_h - 6)], fill=(242, 242, 238))
    d.text((10, 6), legend, fill=(60, 60, 60), font=font(15))

    for i, (label, cells, colour) in enumerate(rows):
        y = legend_h + i * row_h
        d.text((8, y + 10), label, fill=colour, font=font(26))
        for j, c in enumerate(cells):
            sheet.paste(c, (j * CELL, y + ROW_HEADER_H))

    path = OUT / "05_version_comparison.png"
    sheet.save(path)
    print(f"-> {path}  ({sheet.size[0]}x{sheet.size[1]})")

    v2covs = [ev.subject_coverage(
        Image.open(RAW / f"vercmp_v2_{c['subject'].replace(' ', '_')}.png")) for c in cases]
    print(f"v2 coverage spread across 6 subjects: "
          f"{min(v2covs):.3f}-{max(v2covs):.3f} (range {max(v2covs)-min(v2covs):.3f})")
    print("-> row 3 is effectively the same image six times: mode collapse")


if __name__ == "__main__":
    main()
