# Releasing Reef

Reef's version is the git tag. Nothing in the tree carries a version string:
`setuptools-scm` derives it at build time (`[tool.setuptools_scm]` in
`pyproject.toml`), so a build of a tagged commit is `0.4.0` and a build of
the third commit after it is `0.4.1.dev3+g<sha>` — newer than the release,
older than the next one, and traceable to its commit. `reef --version` and
`reef.__version__` report it.

## Versioning

`vMAJOR.MINOR.PATCH`, on `0.x` while public interfaces still move.

- **Minor** (`0.4.0`): the regular release, cut from `main` about once a
  month. Carries new features and, on `0.x`, any breaking change — every one
  listed under *Breaking changes* in the notes.
- **Patch** (`0.4.1`): only from a release branch, only fixes (below). A
  patch never removes or renames anything.
- A commit between tags is a `.devN` pre-release. Nightly wheels, when they
  exist, carry this form and never go to PyPI.

The compatibility contract covers the `reef` top-level import surface, the
HTTP API, recipe YAML, and the CLI. `recipes/*` internals and
`reef.train.slime_backend` are not covered. Removing or renaming a covered
interface takes two minors: warn in one, remove in the next.

`reef-client` is released separately (its own repository and tag flow).
`pyproject.toml` pins the minimum client Reef speaks to; a wire change bumps
that pin and ships in both projects' notes.

## Cutting a minor release

1. Pick a `main` commit whose `ci` run is green.
2. Cut the branch and the tag from it:

   ```bash
   git switch -c release/v0.4 <sha>
   git push -u origin release/v0.4
   git tag -a v0.4.0 -m "reef 0.4.0"
   git push origin v0.4.0
   ```

3. The `release` workflow builds the sdist and wheel, checks the wheel
   version equals the tag, smoke-installs it, and creates a **draft** GitHub
   Release with both files attached and notes grouped by PR label
   (`.github/release.yml`).
4. Edit the draft: write a short *Highlights* paragraph at the top, check that
   every incompatible change appears under *Breaking changes* with what to do
   about it, and add a *Dependencies* line naming the `slimerl/slime` image
   tag and the Slime commit the image builds against. Publish.
5. Build and push the image from the same tag on a GPU host (CI cannot; the
   base image does not fit a hosted runner):

   ```bash
   git checkout v0.4.0
   docker build -f docker/Dockerfile.reef -t <registry>/reef:0.4.0 \
     --build-arg REEF_VERSION=0.4.0 .
   docker push <registry>/reef:0.4.0
   ```

6. When `REEF_PYPI_PUBLISH` is on (below), the workflow also uploads to PyPI.
   Check `pip install reef-infra==0.4.0` in a clean environment.

## Patch releases

A patch fixes something in a release without carrying `main` forward.

- Land the fix on `main` first. Cherry-pick it onto `release/v0.4` in a PR
  titled `[Cherry-pick to release/v0.4] <original title>`.
- Accepted: a regression against the previous release; a critical fix (crash,
  wrong result, data loss, security); a fix to a feature introduced in this
  minor; documentation; changes only the release branch needs.
- Not accepted: features, refactors, dependency bumps beyond what a fix
  requires, anything that changes a covered interface.
- Tag `v0.4.1` on the branch and follow steps 3–6.

Release branches are never merged back to `main`.

## PyPI

The distribution is **`reef-infra`** (`pip install reef-infra`); the import
package stays `reef`. The bare name `reef` on PyPI belongs to an unrelated,
abandoned package.

The `pypi` job in `.github/workflows/release.yml` uploads through
[trusted publishing](https://docs.pypi.org/trusted-publishers/) — no API
token is stored anywhere. One-time setup, in this order:

1. On pypi.org, signed in as the account that will own the project:
   *Your account → Publishing → Add a new pending publisher* with
   PyPI project name `reef-infra`, owner `Human-Agent-Society`, repository
   `reef`, workflow `release.yml`, environment `pypi`. A pending publisher
   reserves the name for that workflow; the project itself is created by the
   first upload.
2. In this repository, *Settings → Environments → New environment* `pypi`.
   Add the maintainers as required reviewers if a human should approve each
   upload (PyPI cannot delete or overwrite a version once it lands).
3. *Settings → Secrets and variables → Actions → Variables*:
   `REEF_PYPI_PUBLISH` = `true`.

The next tag then publishes. To rehearse without touching PyPI, leave the
variable unset: the tag still builds, checks, and drafts the GitHub Release.

## What is not part of a release

`RELEASE.md` deliberately has no release-candidate stage and no per-patch
branches. Both come back when a self-hosted GPU runner can gate a release
branch on an end-to-end training run; until then a release candidate would
only be a tag with nothing to verify it.
