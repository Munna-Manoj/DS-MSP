# Documentation writing guide

The goal of every page in this repository: **anyone who lands here leaves having learned
something they can use.** Code that works but can't be understood is half-finished. This
guide is the standard all docs (`README`, `docs/`, `docs/learn/`, example docstrings) are
held to. If you write or edit docs here, follow it.

It is short on purpose. Read it once; use the [checklist](#the-checklist) every time.

---

## 1. First, know which kind of doc you're writing

Most bad documentation is bad because it tries to be four things at once. Following the
[Diátaxis](https://diataxis.fr/) framework, every section is exactly **one** of these:

| Type | Answers | Reader is… | Here |
| :-- | :-- | :-- | :-- |
| **Tutorial** | "Teach me, step by step." | learning | `docs/learn/` chapters, `examples/` |
| **How-to** | "How do I do X?" | working | README usage, the cookbook |
| **Reference** | "What exactly is X?" | looking up | API tables, Kalibr field orderings |
| **Explanation** | "Why does X work this way?" | understanding | the deep-dives |

Don't blend them in one section. A tutorial that stops to explain theory loses the beginner;
a reference table that tells a story is useless for lookup. If a section is doing two jobs,
split it.

---

## 2. The house rules (non-negotiable)

These are what make this repo's docs distinctive. Keep them everywhere.

1. **Prove it with a number.** Every claim and every demo ends in a measurable result —
   `0.18 px`, `1e-13`, `~28× faster`. "It works well" is not allowed; show the number.
2. **Snippets run.** A reader must be able to copy a snippet and have it work — either it is
   self-contained, or it explicitly continues a labeled setup block (see §3).
3. **Show the expected output.** If a snippet prints or returns something, show what, as a
   comment or an output block. The reader should know they succeeded.
4. **One idea per section.** Descriptive heading, one concept, then move on.
5. **Lead with the point.** First sentence says what the section is for. Don't warm up.

---

## 3. Code snippets — the part people get wrong

This is where most of our docs failed, so it gets its own rules.

**Every variable must be defined or imported. No free-floating names.**

```python
# BAD — what is seed_model? where did X_world_list come from? the reader is stuck.
result = calibrate(seed_model, X_world_list, keypoints_list, visibility_list)
```

```python
# GOOD — every name has an origin; it runs.
from ds_msp.models import KannalaBrandtModel
from ds_msp.calib import calibrate

seed = KannalaBrandtModel(fx=900, fy=900, cx=960, cy=540)   # initial guess
result = calibrate(seed, X_world_list, keypoints_list, visibility_list)
print(result["rms_px"])      # -> ~0.2 px
```

If `X_world_list` etc. genuinely come from earlier, **establish them once in a labeled setup
block and say so**, then reuse those exact names:

> The snippets below continue from this setup:
> ```python
> import numpy as np
> from ds_msp import DoubleSphereCamera
> cam = DoubleSphereCamera(711.57, 711.24, 949.18, 518.81, 0.183, 0.809)
> ```

**The rest of the snippet rules:**

- **Short — the minimum that teaches the point.** Cut anything incidental. A 6-line snippet
  that runs beats a 30-line one that's "realistic."
- **Annotate shapes and units** in comments: `pts = ...  # (N, 3) camera-frame points, metres`.
- **Show the result inline**: `uv, ok = cam.project(pts)   # uv: (N, 2) pixels`.
- **Import what you use, in the snippet** (or in the named setup block it continues).
- **Prefer real, runnable values** over `...` placeholders. If you must elide, make the
  elision obvious and never elide a name the snippet then uses.

---

## 4. Structure of a page

```
# Title — what it is, and who it's for
One sentence: the purpose of this page.

> Prerequisites / setup (if any), once, up top.

## Sections in reading order
   - small steps, each independently verifiable
   - a number or printed output per step

## Try it yourself / Next  (for tutorials)
```

- The **title** says *what* and, for tutorials, *who*.
- **Prerequisites go once, at the top** — not sprinkled through the body.
- **End tutorials with momentum**: a "change one thing and predict the result" exercise, and
  a link to the next step.

---

## 5. Make it visual

Walls of text don't teach. Use the right device for the job:

- **Tables** for comparisons and options (models, parameters, trade-offs).
- **[Mermaid diagrams](https://mermaid.js.org/)** for structure and flow — they render
  natively on GitHub, so prefer them over ASCII art for architecture and pipelines.
- **Callouts**, sparingly, for the one thing the reader must not miss:
  > **Note** / > **Warning**.
- **Whitespace and headings.** Break long passages; let the page breathe.

A figure or diagram should be *informative*, not decorative — if it doesn't help the reader
build a mental model, cut it.

**Generate visuals from real data, reproducibly.** GIFs and figures should come from a
checked-in script (e.g. `scripts/make_learn_gifs.py`), not a one-off screenshot — so they
can be regenerated, and so they show the *actual* output of the code the doc describes.

**Rich 3D / simulation renders** (WebGL/three.js, raylib, manim, …) are produced by
reproducible scripts whose heavy rendering dependencies stay out of this repo; only the final
asset is committed. Every render imports `ds_msp` and cross-checks its math against the library,
so a pretty picture can't drift from the real geometry.

---

## 6. Voice and word choice

- **Active voice, second person, concrete.** "Call `project()` to map points" — not "points
  can be mapped."
- **Cut filler.** Delete "simply", "just", "obviously", "of course" — they shame the reader
  who didn't find it simple.
- **Define jargon on first use.** "the *paraxial* focal (the slope `dr/dθ` at the axis)".
- **Short sentences.** One clause of meaning each.

---

## The checklist

Run this before committing any doc change:

- [ ] **Type** — each section is one Diátaxis kind (tutorial / how-to / reference / explanation).
- [ ] **Snippets run** — every variable is defined or imported, or continues a labeled setup
      block. No free-floating names.
- [ ] **Output shown** — the reader can tell they succeeded.
- [ ] **A number** proves each claim.
- [ ] **Headings** are descriptive; one idea per section.
- [ ] **Links resolve and assets exist** (`grep` the anchors; check the image paths).
- [ ] **Cold read** — a newcomer with no context could follow it and learn something.

> If you can't tick "cold read," the page isn't done yet.
