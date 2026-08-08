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
6. **No paragraph over ~40 words.** If a point needs more, split it into two short paragraphs,
   convert it to a bullet list, or move the secondary point into a `///` admonition. Formal
   derivation pages (e.g. `docs/explain/*_geometry.md`) may run longer where splitting would
   break a proof's logical continuity — tighten the worst offenders there, don't force every
   sentence apart.

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

**On the published MkDocs site, "it runs" is enforced mechanically, not just by convention.**
Every code sample on a `docs/learn/`, `docs/how-to/`, or `docs/explain/` page is a real file
under [`docs_src/`](../docs_src/README.md), pulled into the page with
`{* docs_src/<section>/<slug>/<name>.py hl[1,2] *}` — never hand-copied into the prose. Each
file has a mirrored test at `tests/docs_src/...` that asserts the exact values the page shows,
and `tools/check_docs_src_coverage.py` (CI governance) fails the build if a page references a
file that doesn't exist, or a file exists that no page includes or no test covers. See
`docs_src/README.md` for the exact convention (naming, the bundled-fixtures-only rule, how to
add a new example). `README.md` at the repo root doesn't get this — GitHub renders plain
markdown, not mkdocs macros — so its snippets stay hand-verified inline per the rules above.

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

**Equations get their own line — never run them together like words in a sentence.** A reader
scanning `$z > -w_2\,d_1$, where $d_1=\sqrt{x^2+y^2+z^2}$, so $\theta_{\max}=\arccos(-w_2)$` has
to parse three separate relationships stitched into one clause. Each gets its own display block
instead:

```
$$z > -w_2\, d_1, \qquad d_1 = \sqrt{x^2 + y^2 + z^2}$$

$$\theta_{\max} = \arccos(-w_2)$$
```

- **Inline `$...$` is for a single symbol or a short back-reference** to a display equation
  already shown ("...then $\theta$ from the equation above"), not for a relationship with its
  own operator (`=`, `>`, `\Longleftrightarrow`) — those get `$$...$$`.
- **One relationship per block.** A definition and its immediate substitution can share a block
  (`$$d_1 = \sqrt{x^2+y^2+z^2}$$` right under the equation that uses `d_1`) — a chain of three
  unrelated results cannot.
- **This applies inside tables too.** A comparison table with one short formula per cell is
  fine; a cell packing `$u=..., v=..., (\lambda,\psi)=...$` into one line needs either three
  columns or a footnote-style breakout below the table, not a comma-chain.

**Generate visuals from real data, reproducibly.** GIFs and figures should come from a
checked-in script (e.g. `scripts/make_learn_gifs.py`), not a one-off screenshot — so they
can be regenerated, and so they show the *actual* output of the code the doc describes.

**Rich 3D / simulation renders** (WebGL/three.js, raylib, manim, …) are produced by
reproducible scripts whose heavy rendering dependencies stay out of this repo; only the final
asset is committed. Every render imports `ds_msp` and cross-checks its math against the library,
so a pretty picture can't drift from the real geometry.

**A concept with a spatial or geometric relationship needs a figure by default** — not "if it
doesn't help, cut it" as the only test, but "if it's spatial, assume yes and justify skipping
it." Text alone rarely builds the mental model for a plane, a ray, a rotation, or a
manifold — the reader needs to *see* it. This is the 3Blue1Brown standard: geometric intuition
carried by a picture, not by a paragraph describing one.

### Callouts — use the right one, not just Note/Warning

MkDocs Material ships a full admonition vocabulary (`!!! type` or Typer-style `/// type`);
this project uses six with fixed meanings — pick the one that matches the *job*, not whichever
is closest to hand:

| Type | Renders as | Use for |
| :-- | :-- | :-- |
| `abstract` | 📄 Summary | A **definition** — the one-sentence meaning of a new term, stated precisely, before it's used. |
| `tip` | 💡 Tip | **Intuition** — the plain-language "why this makes sense" before or alongside the formal statement. |
| `success` | ✅ Key result | The **one takeaway** to remember from a section, restated outside the flow of prose. |
| `question` | ❓ Try it | A **predict-then-check** prompt: "what happens if...?" before the reader scrolls to the answer. |
| `warning` | ⚠️ Common mistake | A specific, real error readers make here — not a generic caution. |
| `quote` | 💬 Quote | An attributed line from a cited paper/source — never an unattributed aside. |

Sparingly — one or two per page, at the moment they earn their interruption, per the existing
"the one thing the reader must not miss" rule. A page with a callout every three lines has
stopped using them as callouts.

### Formulas get a walkthrough, not just a display block

Section 5's "equations get their own line" rule fixes *layout*. It does not, by itself, make
an equation *learnable*. Every equation that introduces a new relationship (not a trivial
substitution) needs its terms named in prose immediately after — the Thomas' Calculus
standard: no symbol reaches the page without the reader being told, in words, what it means
and why it's there.

```
$$\theta_{\max} = \arccos(-w_2)$$

Here $\theta_{\max}$ is the largest ray angle the model can represent (measured from the
optical axis), and $w_2$ is the model's *second shape parameter* — it controls how sharply
the projection surface curves away from the sensor. As $w_2 \to -1$, $\arccos(-w_2) \to 0$:
the model collapses to a pinhole with no valid wide-angle rays at all.
```

- **Name every symbol** that hasn't been named in the same section — not just the new ones,
  since a reader dropping in mid-page has no memory of a definition three screens up.
- **Say what happens at the edges** where that's illuminating (limits, degenerate cases,
  sign flips) — this is usually *where* the intuition lives, not an afterthought.
- **Use `///details | Full derivation`** (a collapsible block, closed by default) to hold the
  step-by-step algebra *behind* the walkthrough, not inline with it — Khan Academy's "show me
  the work" pattern. The main flow stays at the intuition/result level; rigor is one click
  away for the reader who wants it, never forced on the one who doesn't.

### Intuition before formalism

For any page introducing a genuinely new concept (not a mechanical how-to step), give the
geometric or intuitive picture *before* the equation that formalizes it — never derive first
and explain what it means afterward. A reader should be able to predict the shape of the
equation from the intuition, then see it confirmed.

### Define the vocabulary, then keep using it

**Bold a term the first time it's precisely defined on a page** (not every time it recurs) —
`**principal point** — the pixel where the optical axis meets the sensor`. If the term repeats
across the same page more than once and a reader could plausibly land there from a search
engine, wrap the first use in an `abbr:` inline tooltip too, so hovering recalls the
definition without breaking the reading flow. Consistent term choice site-wide is already a
`WRITING_GUIDE`/recall-stage rule (§6, EDIT stage P13) — this is the *first-use* half of that
same discipline.

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
- [ ] **Equations are display blocks** (`$$...$$`), one relationship per block — never two or
      more formulas chained into one sentence with inline `$...$`.
- [ ] **Every equation has a walkthrough** — each symbol named in prose right after it appears;
      full algebra (if any) lives in a collapsible `///details | Full derivation`, not inline.
- [ ] **Intuition precedes formalism** for any new concept — the picture/plain-language "why"
      comes before the equation that formalizes it, not after.
- [ ] **Callouts use the right type** (abstract/tip/success/question/warning/quote per the
      table above), sparingly — not just Note/Warning by default.
- [ ] **A spatial/geometric concept has a figure**, or the omission is a deliberate call, not
      an oversight.
- [ ] **Links resolve and assets exist** (`grep` the anchors; check the image paths).
- [ ] **Cold read** — a newcomer with no context could follow it and learn something.

> If you can't tick "cold read," the page isn't done yet.
