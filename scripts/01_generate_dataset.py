"""Generate the synthetic house-style dataset.

Captioning discipline matters more than image count. The rule:

    Caption what VARIES. Never caption the style itself.

Every caption is "<trigger>, a <subject>" and nothing more. The style
attributes (palette, outline, shadow) are deliberately absent from the text,
so the model has nowhere to attach them except the trigger token. Describing
the style in captions is the most common way to get a weak LoRA: the style
smears across common words instead of binding to the trigger.

Also writes a held-out set drawn from the same rules but never trained on, so
the demo can compare generations against real ground truth.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "style"))

from draw_subjects import SUBJECTS, draw_icon  # noqa: E402
from house_style import ACCENTS, TRIGGER  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data"
TRAIN_DIR = OUT / "train"
HOLDOUT_DIR = OUT / "holdout"

# 6 subjects x 5 accents = 30 combinations. Train on 24, hold out 6.
HOLDOUT_PAIRS = [
    ("potion", "gem_violet"),
    ("sword", "gem_green"),
    ("gem", "gold"),
    ("key", "gem_blue"),
    ("shield", "gem_red"),
    ("coin stack", "gem_violet"),
]


def build():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = [(s, a) for s in SUBJECTS for a in ACCENTS]
    train_pairs = [p for p in all_pairs if p not in HOLDOUT_PAIRS]

    rng = random.Random(7)
    rng.shuffle(train_pairs)

    metadata = []
    for i, (subject, accent) in enumerate(train_pairs):
        img = draw_icon(subject, accent, jitter=10, seed=1000 + i)
        fname = f"{i:03d}_{subject.replace(' ', '_')}_{accent}.png"
        img.save(TRAIN_DIR / fname)

        # Caption: trigger + subject only. No style words.
        caption = f"{TRIGGER}, a {subject}"
        metadata.append({"file_name": fname, "text": caption})

    # HuggingFace datasets imagefolder convention.
    with open(TRAIN_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
        for row in metadata:
            f.write(json.dumps(row) + "\n")

    holdout_meta = []
    for i, (subject, accent) in enumerate(HOLDOUT_PAIRS):
        img = draw_icon(subject, accent, jitter=10, seed=9000 + i)
        fname = f"holdout_{subject.replace(' ', '_')}_{accent}.png"
        img.save(HOLDOUT_DIR / fname)
        holdout_meta.append({
            "file_name": fname,
            "subject": subject,
            "accent": accent,
            "prompt": f"{TRIGGER}, a {subject}",
        })

    with open(HOLDOUT_DIR / "holdout.json", "w", encoding="utf-8") as f:
        json.dump(holdout_meta, f, indent=2)

    print(f"Train images   : {len(metadata)}  -> {TRAIN_DIR}")
    print(f"Holdout images : {len(holdout_meta)}  -> {HOLDOUT_DIR}")
    print(f"\nExample captions:")
    for row in metadata[:4]:
        print(f"  {row['file_name']}  ->  \"{row['text']}\"")
    print("\nNote: no style words in any caption. Style binds to the trigger.")


if __name__ == "__main__":
    build()
