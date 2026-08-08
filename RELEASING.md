# Releasing to PyPI

The package builds cleanly and the release is automated by
[`.github/workflows/release.yml`](.github/workflows/release.yml), which publishes to PyPI
via **Trusted Publishing (OIDC)** — no API token is stored anywhere.

Versioning is automated with **[release-please](https://github.com/googleapis/release-please)**,
which reads [Conventional Commits](https://www.conventionalcommits.org/) on `main` and keeps a
running **release PR**. You never hand-edit the version or the changelog — you review and merge a PR.
Publishing needs a **one-time setup** (only the project owner can do this); after that, every
release is just "merge the release PR".

## One-time setup (owner)

1. **Create the PyPI project via Trusted Publishing** (no token needed):
   - Log in to <https://pypi.org> → *Account* → *Publishing* → *Add a pending publisher*.
   - Fill in:
     - **PyPI project name:** `ds-msp`
     - **Owner:** `Munna-Manoj`
     - **Repository name:** `DS-MSP`
     - **Workflow name:** `release.yml`
     - **Environment name:** `pypi`
2. **Create the GitHub environment** named `pypi`:
   - GitHub repo → *Settings* → *Environments* → *New environment* → `pypi`.
   *(Optional but recommended: add required reviewers so a human approves each publish.)*

> Prefer an API token instead of OIDC? Add a `PYPI_API_TOKEN` repo secret and pass it to
> the publish step with `with: { password: ${{ secrets.PYPI_API_TOKEN }} }`. Trusted
> Publishing is recommended because there is no long-lived secret to leak.

## Cutting a release (the normal flow)

You don't tag by hand. The version comes from your commit messages:

| Commit type on `main`        | Effect on the next version (pre-1.0) |
| ---------------------------- | ------------------------------------ |
| `fix: …`                     | patch — `0.x.Y`                      |
| `feat: …`                    | minor — `0.X.0`                      |
| `feat!:` / `BREAKING CHANGE` | minor — `0.X.0` (would be major ≥1.0)|
| `docs/chore/test/refactor/ci`| no release on its own                |

1. **Merge feature work to `main`** with Conventional Commit messages (as usual).
2. **release-please opens/updates a "release PR"** titled `chore(main): release <version>`. It
   bumps `pyproject.toml` + `ds_msp/__init__` and writes the grouped `CHANGELOG.md` section. Leave
   it open until you're ready to ship; it keeps updating as more commits land.
3. **Review the release PR** — confirm the proposed version and curate the changelog wording if you
   want a more narrative entry. If it includes a requirement with `release_gated=yes`, run
   `python tools/check_traceability.py --release` and confirm every linked `realdata` test actually
   executed (did not skip) and passed on the provisioned dataset. A green all-skipped nightly run is
   not release evidence. (Optional local sanity check below.)
4. **Merge the release PR.** release-please creates the `vX.Y.Z` tag and a GitHub Release; the same
   workflow then builds, runs `twine check`, and publishes to PyPI via OIDC. Confirm at
   <https://pypi.org/project/ds-msp/>.
5. After the **first** successful publish, the `pip install ds-msp` line in the README goes live.

> Going to `1.0.0`: when you're ready to promise API stability, merge a commit with a
> `BREAKING CHANGE:` footer (or bump the manifest), and flip the classifier to
> `Development Status :: 5 - Production/Stable`.

## Local build sanity check

```bash
uv build && uv pip install --force-reinstall dist/ds_msp-*.whl
python -c "import ds_msp; print('ok')"
```
