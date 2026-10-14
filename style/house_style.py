"""The "house style" definition for the demo.

This is the single source of truth for the synthetic art style. It is
deliberately strict and machine-checkable, which is what makes the demo
measurable rather than subjective:

  - A locked 12-colour palette. Nothing else may appear in the image.
  - Uniform near-black outlines of fixed weight.
  - Flat fills, no gradients.
  - Hard offset drop shadow, bottom-right, single shadow colour.
  - Centred subject on a fixed neutral background.

SDXL cannot reproduce a locked palette from a text prompt alone. That is the
point: it gives the before/after comparison an objective basis.

Think of this as standing in for a real studio's style bible.
"""

STYLE_NAME = "flatgem"

# Trigger token used in every training caption. Deliberately a rare string so
# it carries no prior meaning for the text encoder.
TRIGGER = "flatgem style"

# --- Locked palette (12 colours) ---------------------------------------------
PALETTE = {
    "outline": (26, 23, 35),
    "shadow": (58, 52, 74),
    "bg": (232, 228, 217),
    "metal_light": (198, 205, 214),
    "metal_dark": (124, 136, 152),
    "wood": (140, 98, 62),
    "leather": (92, 62, 48),
    "gem_red": (206, 74, 74),
    "gem_blue": (74, 136, 206),
    "gem_green": (94, 176, 118),
    "gem_violet": (150, 100, 190),
    "gold": (224, 176, 74),
}

PALETTE_LIST = list(PALETTE.values())

# --- Geometry rules ----------------------------------------------------------
CANVAS = 512
OUTLINE_WIDTH = 7
SHADOW_OFFSET = (14, 14)
SUBJECT_BOX = (96, 96, 416, 416)  # subject must stay inside this

# Accent colours usable as a subject's primary fill.
ACCENTS = ["gem_red", "gem_blue", "gem_green", "gem_violet", "gold"]


def describe_style() -> str:
    """Human-readable style description.

    Used for the *baseline* prompt only. This is the best-effort prompt
    engineering that the LoRA is competing against, so it is written to be
    genuinely good rather than a straw man.
    """
    return (
        "flat vector game icon, bold uniform dark outline, flat fills with no "
        "gradients, hard offset drop shadow to the bottom right, limited muted "
        "palette, cream background, centred single object, clean vector art"
    )
