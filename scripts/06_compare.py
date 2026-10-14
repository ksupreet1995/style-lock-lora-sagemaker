"""Generate the four demo comparisons against a live endpoint.

  ab       fixed-seed base vs LoRA, side by side
  drift    5-pass iteration loop, with and without LoRA
  novel    a subject absent from training data
  dial     LoRA weight sweep 0.0 -> 1.2

Every output image is accompanied by style_score numbers so the claim is
measured, not asserted.
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import boto3
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "style"))
sys.path.insert(0, str(ROOT / "scripts"))

from house_style import TRIGGER, describe_style  # noqa: E402

OUT = ROOT / "outputs"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

REGION = config.region()


def invoke(endpoint, **kwargs):
    rt = boto3.client("sagemaker-runtime", region_name=REGION)
    resp = rt.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Body=json.dumps(kwargs),
    )
    payload = json.loads(resp["Body"].read())
    img = Image.open(io.BytesIO(base64.b64decode(payload["image_b64"])))
    return img, payload["meta"]


def save_raw(img, name):
    """Persist the unlabelled generation so it can be re-scored or re-laid-out
    later without paying for inference again."""
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    img.save(raw_dir / f"{name}.png")


def label(img, text, h=56):
    """Caption an image so the grid is self-explanatory on a slide."""
    out = Image.new("RGB", (img.width, img.height + h), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    d.text((14, img.height + 14), text, fill=(20, 20, 20), font=font)
    return out


def grid(images, cols=None):
    cols = cols or len(images)
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    sheet = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def _load_metrics():
    """Import 02_style_metrics by path; its name starts with a digit."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "style_metrics", ROOT / "scripts" / "02_style_metrics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_METRICS = None


def score(img):
    global _METRICS
    if _METRICS is None:
        _METRICS = _load_metrics()
    return (
        _METRICS.palette_score(img)
        + _METRICS.flatness_score(img)
        + _METRICS.shadow_present(img)
    ) / 3


def demo_ab(endpoint, subject="a lantern", seed=42):
    """Base + best-effort prompt vs LoRA. Same seed, same subject."""
    print(f"\n[ab] subject={subject} seed={seed}")
    base_prompt = f"{subject}, {describe_style()}"
    lora_prompt = f"{TRIGGER}, {subject}"

    img_base, _ = invoke(endpoint, prompt=base_prompt, use_lora=False, seed=seed)
    img_lora, _ = invoke(endpoint, prompt=lora_prompt, use_lora=True, lora_scale=1.0, seed=seed)

    save_raw(img_base, "ab_base")
    save_raw(img_lora, "ab_lora")

    sb, sl = score(img_base), score(img_lora)
    print(f"  base style_score: {sb:.3f}")
    print(f"  lora style_score: {sl:.3f}")

    sheet = grid([
        label(img_base, f"BASE + prompt engineering   style={sb:.3f}"),
        label(img_lora, f"BASE + LoRA   style={sl:.3f}"),
    ])
    path = OUT / "01_ab_comparison.png"
    sheet.save(path)
    print(f"  -> {path}")
    return {"base": sb, "lora": sl}


def demo_drift(endpoint, subject="a lantern", passes=5):
    """The customer's actual complaint: style decay over iterations.

    Simulates their loop by re-prompting with accumulating edit requests,
    which is what an artist iterating on an asset actually does.
    """
    print(f"\n[drift] {passes} passes")
    edits = [
        "", ", taller", ", taller with a wider base",
        ", taller with a wider base and a handle",
        ", taller with a wider base, a handle and decorative trim",
    ][:passes]

    rows, scores = [], {"base": [], "lora": []}
    for use_lora in (False, True):
        imgs = []
        for i, edit in enumerate(edits):
            if use_lora:
                prompt = f"{TRIGGER}, {subject}{edit}"
            else:
                prompt = f"{subject}{edit}, {describe_style()}"
            img, _ = invoke(endpoint, prompt=prompt, use_lora=use_lora,
                            lora_scale=1.0, seed=100 + i)
            save_raw(img, f"drift_{'lora' if use_lora else 'base'}_pass{i+1}")
            s = score(img)
            scores["lora" if use_lora else "base"].append(s)
            imgs.append(label(img.resize((512, 512)), f"pass {i+1}  style={s:.3f}"))
        rows.append(imgs)

    sheet = grid(rows[0] + rows[1], cols=passes)
    path = OUT / "02_drift_test.png"
    sheet.save(path)

    print(f"  base passes: {[round(x, 3) for x in scores['base']]}")
    print(f"  lora passes: {[round(x, 3) for x in scores['lora']]}")
    print(f"  base drift : {scores['base'][0] - scores['base'][-1]:+.3f}")
    print(f"  lora drift : {scores['lora'][0] - scores['lora'][-1]:+.3f}")
    print(f"  -> {path}  (top row base, bottom row LoRA)")
    return scores


def demo_novel(endpoint):
    """Subjects never seen in training. Proves style transfer, not memorisation."""
    print("\n[novel] unseen subjects")
    subjects = ["a spell book", "a treasure chest", "a compass", "a bell"]
    imgs, scores = [], []
    for i, s in enumerate(subjects):
        img, _ = invoke(endpoint, prompt=f"{TRIGGER}, {s}", use_lora=True,
                        lora_scale=1.0, seed=200 + i)
        save_raw(img, f"novel_{s.replace(' ', '_')}")
        sc = score(img)
        scores.append(sc)
        imgs.append(label(img.resize((512, 512)), f"{s}  style={sc:.3f}"))
        print(f"  {s}: {sc:.3f}")

    path = OUT / "03_novel_subjects.png"
    grid(imgs, cols=4).save(path)
    print(f"  -> {path}")
    return dict(zip(subjects, scores))


def demo_dial(endpoint, subject="a lantern", seed=42):
    """LoRA weight sweep. The control artists intuitively understand."""
    print("\n[dial] weight sweep")
    weights = [0.0, 0.3, 0.6, 0.9, 1.2]
    imgs, scores = [], []
    for w in weights:
        img, _ = invoke(endpoint, prompt=f"{TRIGGER}, {subject}", use_lora=w > 0,
                        lora_scale=w, seed=seed)
        save_raw(img, f"dial_w{str(w).replace('.', '_')}")
        sc = score(img)
        scores.append(sc)
        imgs.append(label(img.resize((512, 512)), f"weight {w}  style={sc:.3f}"))
        print(f"  weight {w}: {sc:.3f}")

    path = OUT / "04_style_dial.png"
    grid(imgs, cols=5).save(path)
    print(f"  -> {path}")
    return dict(zip([str(w) for w in weights], scores))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--only", choices=["ab", "drift", "novel", "dial"], default=None)
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    results_path = OUT / "results.json"

    # Merge into any existing results rather than overwriting. Running the
    # comparisons one at a time with --only is the normal rehearsal workflow,
    # and each run would otherwise clobber the previous run's scores.
    results = {}
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"warning: {results_path.name} unreadable, starting fresh")

    if args.only in (None, "ab"):
        results["ab"] = demo_ab(args.endpoint)
    if args.only in (None, "drift"):
        results["drift"] = demo_drift(args.endpoint)
    if args.only in (None, "novel"):
        results["novel"] = demo_novel(args.endpoint)
    if args.only in (None, "dial"):
        results["dial"] = demo_dial(args.endpoint)

    results["_meta"] = {
        "endpoint": args.endpoint,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nScores  -> {results_path}")
    print(f"Images  -> {OUT}")
    print(f"Sections present: {sorted(k for k in results if not k.startswith('_'))}")
