"""Generate the v2 training set.

What changed from v1, and why:

v1 diagnosis (measured, not assumed): the v1 training set already had mean
subject coverage 0.1463 with std 0.047, identical to the holdout. The
preprocessing transform was a no-op (+0.002). So the composition failure was
NOT caused by the data, and my first explanation was wrong.

The real cause: base SDXL and the v1 LoRA fail almost identically (0.678 vs
0.667), so this is SDXL's own prior toward large frame-filling subjects.
Inference-side fixes could not shift it (best ablation 0.641 vs target
0.146), so it has to be addressed in training.

v2 changes:
  1. 10 subjects instead of 6, roughly doubling the dataset to 44 images.
     More signal for the same style relationship.
  2. Mild subject-scale variation (0.85-1.1). Not to widen the coverage
     range, but to repeat "small subject on a large field" at several sizes
     so the relationship is learned rather than one memorised size.
  3. Captions mention "icon" to give composition a lexical anchor the trigger
     can bind to. Still no palette/outline/shadow words.

The retrain itself also targets more UNet layers; see src/train_lora.py.

Holdout and evaluation subjects are deliberately excluded from training.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))

from draw_subjects import ALL_SUBJECTS, draw_icon_v2  # noqa: E402
from house_style import ACCENTS, TRIGGER  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
TRAIN_DIR = OUT / "train_v2"

# Keep the v1 holdout pairs out of training so the holdout stays valid.
HOLDOUT_PAIRS = [
    ("potion", "gem_violet"),
    ("sword", "gem_green"),
    ("gem", "gold"),
    ("key", "gem_blue"),
    ("shield", "gem_red"),
    ("coin stack", "gem_violet"),
]

SCALES = [0.85, 0.95, 1.0, 1.1]


def build():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for old in TRAIN_DIR.glob("*"):
        old.unlink()

    pairs = [(s, a) for s in ALL_SUBJECTS for a in ACCENTS
             if (s, a) not in HOLDOUT_PAIRS]

    rng = random.Random(11)
    rng.shuffle(pairs)

    # Cap at 44 images: enough to roughly double v1 without pushing the
    # training job much past ~100 min.
    pairs = pairs[:44]

    metadata = []
    for i, (subject, accent) in enumerate(pairs):
        scale = SCALES[i % len(SCALES)]
        img = draw_icon_v2(subject, accent, jitter=12, seed=2000 + i, scale=scale)
        fname = f"{i:03d}_{subject.replace(' ', '_')}_{accent}_s{int(scale*100)}.png"
        img.save(TRAIN_DIR / fname)

        # "icon" gives composition a lexical anchor. Still no style words.
        caption = f"{TRIGGER}, a {subject} icon"
        metadata.append({"file_name": fname, "text": caption})

    with open(TRAIN_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row) + "\n")

    print(f"v2 train images: {len(metadata)} -> {TRAIN_DIR}")
    print(f"subjects       : {len(ALL_SUBJECTS)}")
    print(f"scales         : {SCALES}")
    print("\nexample captions:")
    for row in metadata[:4]:
        print(f"  {row['file_name']}  ->  \"{row['text']}\"")


if __name__ == "__main__":
    build()
