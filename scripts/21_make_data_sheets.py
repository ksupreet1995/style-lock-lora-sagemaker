"""Build presentation sheets for the training data and its outputs.

Fills a real gap: the demo could show generated results but not the corpus
that produced them. For a studio audience the training set is the most
concrete artifact in the whole story, because it is the thing they would have
to supply. Showing it makes two abstract arguments physical:

  1. The captioning rule. Seeing "flatgem style, a sword" printed under an
     image lands faster than describing the convention.
  2. The holdout discipline. Seeing 6 images physically separated from 24
     explains why the evaluation is trustworthy.

Produces:
  06_training_data.png   all 24 v1 training images with their captions
  07_holdout_split.png   the 6 withheld references, labelled as never-trained
  08_input_to_output.png ground truth vs LoRA vs baseline, per subject
  09_training_data_v2.png the 44-image v2 corpus, for the rejection story
"""

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "style"))

spec = importlib.util.spec_from_file_location(
    "ev", ROOT / "scripts" / "09_evaluate_holdout.py")
ev = importlib.util.module_from_spec(spec)
_argv = sys.argv
sys.argv = ["x", "--endpoint", "d"]
spec.loader.exec_module(ev)
sys.argv = _argv
SM = ev._metrics_mod()

TRAIN = ROOT / "data" / "train"
TRAIN_V2 = ROOT / "data" / "train_v2"
HOLDOUT = ROOT / "data" / "holdout"
RAW = ROOT / "outputs" / "raw"
OUT = ROOT / "outputs"


def font(size, bold=False):
    for name in (["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def style_score(img):
    return (SM.palette_score(img) + SM.flatness_score(img)
            + SM.shadow_present(img)) / 3


def tile(img, lines, cell, label_h, colour=(25, 25, 25)):
    """One captioned cell. lines is a list of short strings."""
    out = Image.new("RGB", (cell, cell + label_h), (255, 255, 255))
    out.paste(img.resize((cell, cell), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(out)
    y = cell + 6
    for i, line in enumerate(lines):
        d.text((6, y), line, fill=colour if i == 0 else (90, 90, 90),
               font=font(15 if i == 0 else 13))
        y += 17
    return out


# Printed on every sheet that shows cov/sty numbers. Without it, a viewer can
# reasonably read "cov 0.99" as a good score, when the house style wants ~0.15.
# cov is target-matching, not higher-is-better.
LEGEND = ("sty = style score, 0-1, higher is better (palette + flat fills + shadow).   "
          "cov = subject coverage, fraction of canvas filled - TARGET is ~0.15, "
          "not higher.")


def sheet(tiles, cols, title, subtitle, path, header_h=74, legend=True):
    if not tiles:
        print(f"  skipped {path.name}: no tiles")
        return
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    legend_h = 30 if legend else 0
    img = Image.new("RGB", (tw * cols, header_h + legend_h + th * rows),
                    (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 12), title, fill=(15, 15, 15), font=font(30, bold=True))
    d.text((10, 48), subtitle, fill=(95, 95, 95), font=font(17))
    if legend:
        d.rectangle([(0, header_h), (img.width, header_h + legend_h - 6)],
                    fill=(242, 242, 238))
        d.text((10, header_h + 6), LEGEND, fill=(60, 60, 60), font=font(14))
    for i, t in enumerate(tiles):
        img.paste(t, ((i % cols) * tw,
                      header_h + legend_h + (i // cols) * th))
    img.save(path)
    print(f"  {path.name}  ({img.size[0]}x{img.size[1]}, {len(tiles)} tiles)")


def training_sheet():
    """All 24 v1 training images with the exact captions used."""
    meta = [json.loads(l) for l in
            open(TRAIN / "metadata.jsonl", encoding="utf-8") if l.strip()]
    tiles = []
    for row in sorted(meta, key=lambda r: r["file_name"]):
        img = Image.open(TRAIN / row["file_name"])
        tiles.append(tile(img, [f'"{row["text"]}"',
                                f'cov {ev.subject_coverage(img):.2f}  '
                                f'sty {style_score(img):.3f}'],
                          cell=190, label_h=42))
    sheet(tiles, 8,
          "v1 training corpus - 24 images, drawn procedurally in code",
          'Caption rule: name only what VARIES. No palette, no "flat", no '
          '"outline". Style binds to the trigger token "flatgem style".',
          OUT / "06_training_data.png")


def training_sheet_v2():
    meta = [json.loads(l) for l in
            open(TRAIN_V2 / "metadata.jsonl", encoding="utf-8") if l.strip()]
    tiles = []
    for row in sorted(meta, key=lambda r: r["file_name"]):
        img = Image.open(TRAIN_V2 / row["file_name"])
        tiles.append(tile(img, [f'"{row["text"]}"'], cell=150, label_h=26))
    sheet(tiles, 11,
          "v2 training corpus - 44 images, 10 subjects, scale varied 0.85-1.1",
          "Doubled the data and varied subject scale. It still failed: the "
          "problem was never the dataset.",
          OUT / "09_training_data_v2.png",
          legend=False)  # captions only, no cov/sty numbers


def holdout_sheet():
    cases = json.loads((HOLDOUT / "holdout.json").read_text(encoding="utf-8"))
    tiles = []
    for c in cases:
        img = Image.open(HOLDOUT / c["file_name"])
        tiles.append(tile(img, [c["subject"],
                                f'cov {ev.subject_coverage(img):.2f}  '
                                f'sty {style_score(img):.3f}'],
                          cell=250, label_h=42, colour=(20, 95, 20)))
    sheet(tiles, 6,
          "Held-out references - 6 images the model NEVER saw",
          "Generated by the same rules, excluded from training. This is the "
          "ground truth every honest number in the demo is measured against.",
          OUT / "07_holdout_split.png")


def input_output_sheet():
    """The money sheet: withheld reference next to what each arm produced."""
    cases = json.loads((HOLDOUT / "holdout.json").read_text(encoding="utf-8"))
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    per = {r["subject"]: r for r in results["holdout"]["per_subject"]}

    cell, label_h = 240, 44

    # Row-major: one ROW per arm, one COLUMN per subject. Same shape as
    # 05_version_comparison, which reads well on a 16:9 slide.
    rows_spec = [
        ("reference", lambda c: Image.open(HOLDOUT / c["file_name"]),
         "gt_coverage", "gt_style_score", (20, 95, 20)),
        ("LoRA", lambda c: Image.open(
            RAW / f"holdout_lora_{c['subject'].replace(' ', '_')}.png"),
         "lora_coverage", "lora_style_score", (25, 25, 25)),
        ("base + prompt", lambda c: Image.open(
            RAW / f"holdout_base_{c['subject'].replace(' ', '_')}.png"),
         "base_coverage", "base_style_score", (140, 60, 20)),
    ]

    tiles = []
    for arm, loader, cov_key, sty_key, colour in rows_spec:
        for c in cases:
            row = per[c["subject"]]
            head = f"{c['subject']}  [{arm}]" if arm == "reference" else arm
            tiles.append(tile(loader(c), [
                head,
                f"cov {row[cov_key]:.2f}  sty {row[sty_key]:.3f}",
            ], cell, label_h, colour=colour))

    sheet(tiles, 6,
          "Reference vs generated - 6 withheld subjects",
          "Row 1 reference (green), row 2 LoRA, row 3 base+prompt (orange). "
          "LoRA wins on style rules; both miss composition. Potion and sword "
          "(columns 1-2) are near-perfect on both.",
          OUT / "08_input_to_output.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print("building data sheets:")
    training_sheet()
    holdout_sheet()
    input_output_sheet()
    training_sheet_v2()
