"""Procedural drawing of game inventory icons in the house style.

Each subject is drawn from primitives under the style rules in house_style.py.
Subject shape and colour vary; the *style* (palette, outline, shadow) never
does. That separation is exactly what we want a LoRA to learn.
"""

import random

from PIL import Image, ImageDraw

from house_style import CANVAS, OUTLINE_WIDTH, PALETTE, SHADOW_OFFSET

OL = PALETTE["outline"]
OW = OUTLINE_WIDTH


def _new_canvas() -> tuple:
    img = Image.new("RGB", (CANVAS, CANVAS), PALETTE["bg"])
    return img, ImageDraw.Draw(img)


def _shadow_layer() -> tuple:
    """Separate layer so the shadow sits behind the subject."""
    img = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _compose(base: Image.Image, shadow: Image.Image, subject: Image.Image) -> Image.Image:
    out = base.copy()
    out.paste(shadow, (0, 0), shadow)
    out.paste(subject, (0, 0), subject)
    return out


# --- Subject primitives ------------------------------------------------------
# Every draw_* function returns a list of (shape_kind, coords, fill) tuples so
# the same geometry can be rendered twice: once as a shadow, once as the real
# subject.


def _potion(cx, cy, accent):
    """Round-bottomed flask with a cork and a liquid line."""
    body_r = 86
    shapes = [
        # neck
        ("rect", (cx - 26, cy - 128, cx + 26, cy - 54), PALETTE["metal_light"]),
        # cork
        ("rect", (cx - 34, cy - 160, cx + 34, cy - 122), PALETTE["wood"]),
        # body
        ("ellipse", (cx - body_r, cy - 62, cx + body_r, cy + body_r + 10), PALETTE["metal_light"]),
        # liquid
        ("pieslice", (cx - body_r + 14, cy - 48, cx + body_r - 14, cy + body_r - 4), accent),
    ]
    return shapes


def _sword(cx, cy, accent):
    blade_half = 26
    return [
        ("polygon", [
            (cx, cy - 178),
            (cx + blade_half, cy - 130),
            (cx + blade_half, cy + 58),
            (cx - blade_half, cy + 58),
            (cx - blade_half, cy - 130),
        ], PALETTE["metal_light"]),
        # crossguard
        ("rect", (cx - 92, cy + 58, cx + 92, cy + 94), PALETTE["gold"]),
        # grip
        ("rect", (cx - 20, cy + 94, cx + 20, cy + 162), PALETTE["leather"]),
        # pommel
        ("ellipse", (cx - 32, cy + 150, cx + 32, cy + 196), accent),
    ]


def _gem(cx, cy, accent):
    r = 118
    return [
        ("polygon", [
            (cx, cy - r),
            (cx + r, cy - r // 3),
            (cx + int(r * 0.62), cy + r),
            (cx - int(r * 0.62), cy + r),
            (cx - r, cy - r // 3),
        ], accent),
        # top facet
        ("polygon", [
            (cx, cy - r),
            (cx + r, cy - r // 3),
            (cx, cy + r // 6),
            (cx - r, cy - r // 3),
        ], PALETTE["metal_light"]),
    ]


def _key(cx, cy, accent):
    return [
        ("ellipse", (cx - 76, cy - 150, cx + 16, cy - 58), accent),
        ("rect", (cx - 42, cy - 70, cx - 18, cy + 150), PALETTE["gold"]),
        ("rect", (cx - 42, cy + 70, cx + 52, cy + 98), PALETTE["gold"]),
        ("rect", (cx - 42, cy + 118, cx + 34, cy + 146), PALETTE["gold"]),
    ]


def _shield(cx, cy, accent):
    return [
        ("polygon", [
            (cx, cy - 150),
            (cx + 124, cy - 104),
            (cx + 96, cy + 86),
            (cx, cy + 158),
            (cx - 96, cy + 86),
            (cx - 124, cy - 104),
        ], PALETTE["metal_dark"]),
        ("polygon", [
            (cx, cy - 96),
            (cx + 70, cy - 66),
            (cx + 54, cy + 52),
            (cx, cy + 98),
            (cx - 54, cy + 52),
            (cx - 70, cy - 66),
        ], accent),
    ]


def _coin_stack(cx, cy, accent):
    shapes = []
    for i in range(3):
        y = cy + 72 - i * 58
        shapes.append(("ellipse", (cx - 96, y - 40, cx + 96, y + 40), PALETTE["gold"]))
        shapes.append(("ellipse", (cx - 44, y - 20, cx + 44, y + 20), accent))
    return shapes


SUBJECTS = {
    "potion": _potion,
    "sword": _sword,
    "gem": _gem,
    "key": _key,
    "shield": _shield,
    "coin stack": _coin_stack,
}


def _render(draw, shapes, override_fill=None, outline=True):
    for kind, coords, fill in shapes:
        f = override_fill if override_fill else fill
        ol = OL if (outline and not override_fill) else None
        w = OW if ol else 0
        if kind == "rect":
            draw.rectangle(coords, fill=f, outline=ol, width=w)
        elif kind == "ellipse":
            draw.ellipse(coords, fill=f, outline=ol, width=w)
        elif kind == "polygon":
            draw.polygon(coords, fill=f, outline=ol, width=w)
        elif kind == "pieslice":
            draw.pieslice(coords, start=0, end=180, fill=f, outline=ol, width=w)


def draw_icon(subject: str, accent_name: str, jitter: int = 0, seed: int = 0) -> Image.Image:
    """Render one icon in the house style.

    jitter shifts the subject slightly so the dataset is not pixel-identical,
    which would otherwise encourage the LoRA to memorise position.
    """
    rng = random.Random(seed)
    cx = CANVAS // 2 + rng.randint(-jitter, jitter)
    cy = CANVAS // 2 + rng.randint(-jitter, jitter)

    accent = PALETTE[accent_name]
    shapes = SUBJECTS[subject](cx, cy, accent)

    base, _ = _new_canvas()

    # Shadow: same geometry, offset, flat shadow colour, no outline.
    shadow_img, shadow_draw = _shadow_layer()
    offset_shapes = []
    for kind, coords, fill in shapes:
        if isinstance(coords, list):
            moved = [(x + SHADOW_OFFSET[0], y + SHADOW_OFFSET[1]) for x, y in coords]
        else:
            moved = (
                coords[0] + SHADOW_OFFSET[0], coords[1] + SHADOW_OFFSET[1],
                coords[2] + SHADOW_OFFSET[0], coords[3] + SHADOW_OFFSET[1],
            )
        offset_shapes.append((kind, moved, fill))
    _render(shadow_draw, offset_shapes, override_fill=PALETTE["shadow"], outline=False)

    subject_img, subject_draw = _shadow_layer()
    _render(subject_draw, shapes, outline=True)

    return _compose(base, shadow_img, subject_img)


# --- v2 additions -----------------------------------------------------------
# Four extra subjects to roughly double the dataset. Deliberately NOT lantern,
# spell book, treasure chest, compass or bell: those are held back for the
# A/B, drift, dial and novel-subject evaluations, and training on them would
# destroy those tests.


def _scroll(cx, cy, accent):
    return [
        ("rect", (cx - 104, cy - 70, cx + 104, cy + 70), PALETTE["metal_light"]),
        ("ellipse", (cx - 134, cy - 96, cx - 74, cy + 96), PALETTE["wood"]),
        ("ellipse", (cx + 74, cy - 96, cx + 134, cy + 96), PALETTE["wood"]),
        ("rect", (cx - 60, cy - 24, cx + 60, cy + 10), accent),
    ]


def _ring(cx, cy, accent):
    return [
        ("ellipse", (cx - 108, cy - 68, cx + 108, cy + 124), PALETTE["gold"]),
        # Hole punched with the background colour.
        ("ellipse", (cx - 62, cy - 16, cx + 62, cy + 80), PALETTE["bg"]),
        ("polygon", [
            (cx, cy - 152),
            (cx + 54, cy - 86),
            (cx, cy - 36),
            (cx - 54, cy - 86),
        ], accent),
    ]


def _axe(cx, cy, accent):
    return [
        ("rect", (cx - 18, cy - 150, cx + 18, cy + 160), PALETTE["wood"]),
        ("polygon", [
            (cx + 10, cy - 150),
            (cx + 128, cy - 112),
            (cx + 128, cy - 20),
            (cx + 10, cy - 10),
        ], PALETTE["metal_light"]),
        ("polygon", [
            (cx - 10, cy - 150),
            (cx - 128, cy - 112),
            (cx - 128, cy - 20),
            (cx - 10, cy - 10),
        ], accent),
    ]


def _helmet(cx, cy, accent):
    return [
        ("pieslice", (cx - 116, cy - 128, cx + 116, cy + 104), PALETTE["metal_dark"]),
        ("rect", (cx - 116, cy - 16, cx + 116, cy + 34), accent),
        ("rect", (cx - 116, cy + 34, cx + 116, cy + 78), PALETTE["metal_light"]),
        ("rect", (cx - 14, cy - 150, cx + 14, cy - 96), PALETTE["gold"]),
    ]


SUBJECTS_V2 = {
    "scroll": _scroll,
    "ring": _ring,
    "axe": _axe,
    "helmet": _helmet,
}


def _scale_shapes(shapes, cx, cy, s):
    """Scale shape geometry about (cx, cy).

    Outline width is deliberately NOT scaled: uniform outline weight is part
    of the style bible, so it stays constant across subject sizes.
    """
    out = []
    for kind, coords, fill in shapes:
        if isinstance(coords, list):
            moved = [(cx + (x - cx) * s, cy + (y - cy) * s) for x, y in coords]
        else:
            moved = (
                cx + (coords[0] - cx) * s, cy + (coords[1] - cy) * s,
                cx + (coords[2] - cx) * s, cy + (coords[3] - cy) * s,
            )
        out.append((kind, moved, fill))
    return out


ALL_SUBJECTS = {**SUBJECTS, **SUBJECTS_V2}


def draw_icon_v2(subject: str, accent_name: str, jitter: int = 0,
                 seed: int = 0, scale: float = 1.0) -> Image.Image:
    """Render one icon, with optional subject scaling.

    Mild scale variation (0.85-1.1) keeps coverage inside the house-style
    range while repeating the "small subject on a large field" relationship
    across different sizes, which is a stronger signal than one fixed size.
    """
    rng = random.Random(seed)
    cx = CANVAS // 2 + rng.randint(-jitter, jitter)
    cy = CANVAS // 2 + rng.randint(-jitter, jitter)

    accent = PALETTE[accent_name]
    shapes = ALL_SUBJECTS[subject](cx, cy, accent)
    if scale != 1.0:
        shapes = _scale_shapes(shapes, cx, cy, scale)

    base, _ = _new_canvas()

    shadow_img, shadow_draw = _shadow_layer()
    offset_shapes = []
    for kind, coords, fill in shapes:
        if isinstance(coords, list):
            moved = [(x + SHADOW_OFFSET[0], y + SHADOW_OFFSET[1]) for x, y in coords]
        else:
            moved = (
                coords[0] + SHADOW_OFFSET[0], coords[1] + SHADOW_OFFSET[1],
                coords[2] + SHADOW_OFFSET[0], coords[3] + SHADOW_OFFSET[1],
            )
        offset_shapes.append((kind, moved, fill))
    _render(shadow_draw, offset_shapes, override_fill=PALETTE["shadow"], outline=False)

    subject_img, subject_draw = _shadow_layer()
    _render(subject_draw, shapes, outline=True)

    return _compose(base, shadow_img, subject_img)
