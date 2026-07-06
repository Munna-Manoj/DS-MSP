# `docs_src/` — real, tested code behind every doc-page example

Every code sample shown on the published docs site (`docs/learn/`, `docs/how-to/`,
`docs/explain/`) is a real, standalone file here, pulled into the page via mkdocs's
`{* docs_src/<path> hl[1,2] *}` macro (`mdx-include` + `markdown-include-variants`). No code is
hand-copied into a page's prose — if it's shown, it's a file that actually runs, with a mirrored
test at `tests/docs_src/<same path>/test_<name>.py`. `tools/check_docs_src_coverage.py` (CI
governance job) enforces both directions: every reference resolves to a real file, and every
file is referenced and tested.

This mirrors [Typer](https://github.com/fastapi/typer)'s `docs_src/` convention exactly. It is
**not** the same thing as [`examples/`](../examples) at the repo root:

| | `docs_src/` | `examples/` |
|---|---|---|
| **Purpose** | one small snippet per doc-page concept | a complete, standalone capstone pipeline |
| **Dependencies** | bundled fixtures only (`test_config.json`, `assets/test_image*.jpg`, `anns.json`) or none — **never** an external dataset | may need TUM-VI/EuRoC or other downloaded data |
| **Runs unconditionally?** | yes — every file must run in any fresh checkout, no setup | no — some need a dataset the reader downloads separately |
| **Included in a doc page?** | yes, via `{* ... *}` | no — pages link to it and show an excerpt instead |

## Layout

```
docs_src/<section>/<slug>/<snippet_name>.py
```

- `<section>` is `learn`, `how_to`, or `explain` — mirrors `docs/{learn,how-to,explain}` 1:1
  (`how_to` uses an underscore: it has to be a valid Python package name).
- `<slug>` is the doc page's filename stem with any leading `NN_` numeral stripped (page order
  comes from `mkdocs.yml`'s `nav:`, not the filename, so nothing is lost).
- `<snippet_name>` is a short, descriptive name for the concept the file demonstrates — not a
  sequential `tutorial001`, since these are usually distinct ideas within one page rather than
  a step-by-step build-up of one app.
- Every `<section>/<slug>/` directory has an `__init__.py` so tests can import it as
  `docs_src.<section>.<slug>.<snippet_name>`.

## Shape of a file

```python
def main() -> None:
    ...
    print(...)  # every value the doc page shows must be an explicit print, not a bare
                # trailing expression -- this is a real module, not a REPL session


if __name__ == "__main__":
    main()
```

Use plain relative paths for bundled fixtures (`open("test_config.json")`,
`cv2.imread("assets/test_image.jpg")`) — not `examples/*.py`'s `Path(__file__).resolve()`
dance. The mirrored test (`tests/docs_src/...`) runs from the repo root, which is what makes
this assumption safe here.

## Hard rule: no dataset dependence, no `skipif`

A `docs_src/` file may depend only on already-committed bundled fixtures or nothing external.
If an example needs TUM-VI/EuRoC data, it does **not** belong here — it stays a hand-authored,
clearly-excerpted fenced block on the doc page with a link to the real `examples/*.py` file. A
`skipif`-guarded docs_src file would defeat the whole point of the mechanism: an included file
must be unconditionally runnable, or the page is showing code that isn't actually exercised on
the common path.

## Adding a new example

1. Write `docs_src/<section>/<slug>/<name>.py` in the shape above.
2. Run it fresh from the repo root and capture its actual output — don't trust a value copied
   from prose; that's exactly the kind of drift this mechanism exists to prevent.
3. Write `tests/docs_src/<section>/<slug>/test_<name>.py`: import the module, call `.main()`
   under `capsys`, assert the exact values the doc page will show; add a second
   `subprocess.run([sys.executable, "-m", "docs_src.<section>.<slug>.<name>"], cwd=ROOT)` smoke
   check.
4. Reference it from the doc page: `{* docs_src/<section>/<slug>/<name>.py hl[1,2] *}`.
5. Run `python tools/check_docs_src_coverage.py`, `pytest tests/docs_src/... tests/docs -q`, and
   `mkdocs build --strict` before committing.
