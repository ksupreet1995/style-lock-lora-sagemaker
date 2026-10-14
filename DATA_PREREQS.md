# Training data prerequisites

What your own art needs to look like to train a style LoRA with this repo.
Everything here is verified against the code in `src/train_lora.py` rather
than being general advice.

---

## The short version

| | |
|---|---|
| **How many images** | 25-50 to start. 20 is a workable floor, 100-200 is better. |
| **Plus a holdout** | 5-10 images you never train on, for honest evaluation. |
| **Format** | PNG or JPEG, RGB. |
| **Resolution** | 1024x1024 or larger on the short side. |
| **Aspect ratio** | **Square or near-square.** This is the one that bites. |
| **Captions** | One short line per image naming the subject. Auto-draftable. |
| **What you do NOT need** | No scoring, tagging, annotation, masks, or bounding boxes. |

**Total human effort for 50 images: roughly an hour**, most of it art
direction rather than data work.

---

## 1. Consistency matters more than quantity

The single biggest determinant of quality. Every image must be recognisably
the *same* style. Thirty tightly consistent images beat two hundred loosely
related ones, because a LoRA learns what is common across the set — and if
the only thing in common is "our studio made it", that is what it learns.

**Good:** one artist's icon set, one project's UI assets, one character
sheet's worth of work in a locked style.

**Bad:** a mixed portfolio, several projects spanning a style evolution, or
anything with a few outliers "to add variety". Outliers do not add variety,
they blur the target.

If your art spans multiple styles, that is multiple LoRAs, not one.

---

## 2. Subject variety within that one style

Style must be the constant; subject must vary. If every training image is a
sword, the model entangles "our style" with "sword" and struggles elsewhere.

Our 24-image demo set used 6 subjects x 5 colourways. The visible cost of
that narrowness: the weakest unseen subject (a bell) scored **0.783** against
~0.96 for the others. More subject variety broadens coverage.

**Rule of thumb:** at least 8-10 distinct subjects across the set.

---

## 3. Aspect ratio: the prerequisite people miss

`train_lora.py` resizes the **short** side to 1024 then centre-crops. Measured
effect on content retained:

| source | kept | lost |
|---|---|---|
| square (any size) | **100%** | 0% |
| 4:3 landscape | 75% | 25% |
| 16:9 landscape | **56%** | **44%** |
| 9:16 portrait | 56% | 44% |
| 3:1 banner | 33% | 67% |

A 16:9 frame loses nearly half its content, silently, including anything near
the left and right edges.

**Three options:**

1. **Supply square or near-square crops.** Simplest, and best if the art is
   icons, portraits or items.
2. **Pre-crop deliberately** so the subject is centred and the important
   content survives. An artist doing this by eye beats an automatic crop.
3. **Change the training code** to use bucketed aspect ratios, which trains
   on native shapes without cropping. Standard practice in the LoRA
   community, a modest change to the `IconDataset` class, and the right
   answer if your assets are genuinely wide.

Worth checking before a run rather than discovering it afterwards.

---

## 4. Resolution

Short side **1024px or more**. Smaller works but gets upscaled, and soft
inputs produce a soft LoRA. Larger is fine, it gets downscaled cleanly.

Avoid heavy JPEG compression: the model will faithfully learn the artefacts
as part of "the style".

---

## 5. Captions: the only manual step

One short line per image, naming only what **varies**:

```
flatgem style, a sword
flatgem style, a potion
flatgem style, a treasure chest
```

Format is `metadata.jsonl`, one JSON object per line:

```json
{"file_name": "001_sword.png", "text": "flatgem style, a sword"}
```

**The rule: caption the subject, never the style.** Do not write "flat cel
shading, bold outlines, muted palette" — those belong to the style, and
naming them spreads the style across generic vocabulary instead of binding it
to the trigger token. That is the most common cause of a weak LoRA.

**Trigger token:** pick a rare string, e.g. `acmestyle style`. It should
carry no prior meaning for the text encoder, so avoid real words like
"cartoon", and avoid a studio or brand name if it is likely to appear in web
training data.

A vision model can draft all captions; an artist then corrects a handful.

---

## 6. Holdout set: non-negotiable

5-10 images, same style, **never** used in training. Without them you cannot
tell generalisation from memorisation, and you cannot detect a regression.

This is what caught the v2 failure in our demo. Training loss looked
perfectly healthy throughout; only evaluation against withheld art revealed
the model had collapsed. That cost **$2.32** to discover instead of shipping.

Hold back a spread of subjects, not five variations of one thing.

---

## 7. Rights and consent

Not a technical prerequisite, but worth settling before training rather than
after.

- Does the organisation hold rights to the training images?
- Have contributing artists consented to their style being replicated?
- If the style is substantially one person's, that is a different
  conversation from replicating a house style built by a team.

Base model licensing is already handled: SDXL 1.0 is CreativeML Open
RAIL++-M, which permits commercial use. FLUX.1-dev is non-commercial and
disqualified for shippable assets unless separately licensed.

---

## What you do NOT need to provide

Worth stating explicitly, because a data request usually implies a labelling
project and that is where pilots stall.

- No style scores or quality ratings. Our `sty` and `cov` numbers are
  computed by script and are **not** training inputs.
- No tags, categories or keywords.
- No masks, segmentation or bounding boxes.
- No paired before/after examples.
- No annotation tooling or labelling vendor.

The training job reads the pixels and the caption. Nothing else.

---

## Effort estimate for a 50-image pilot

| task | who | time |
|---|---|---|
| Choose 50 representative images + 10 holdout | art director | 30-45 min |
| Crop to square if needed | artist | 15-30 min |
| Auto-draft captions | script | minutes |
| Review and correct captions | artist | 15-20 min |
| Upload to S3 | engineer | minutes |
| Train, evaluate, register | engineer | ~2 hrs, mostly GPU |

**About an hour of human time.** The genuinely manual part is deciding which
images represent the house style — that is an art direction judgement, not a
data-processing task.
