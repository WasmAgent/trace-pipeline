# Zenodo DOI setup

> A 5-minute setup that gives every GitHub release of this repo a
> permanent DOI, suitable for academic citation alongside or in lieu of
> an arxiv ID.

## Why bother

Zenodo (CERN-hosted) gives software releases a permanent DOI without
requiring a journal. Citations look like:

```
telleroutlook (2026). trace-pipeline: v0.1.0.
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

Useful in three cases:

1. **Before arXiv ID is assigned** — the work is citable today.
2. **For software-specific cites** — some journals expect a software DOI
   distinct from the methodology paper's arxiv ID.
3. **As a long-term archive** — Zenodo is committed to long-term
   preservation; GitHub may not be.

## Setup (one-time, ~5 minutes)

### 1. Sign in to Zenodo via GitHub

Go to <https://zenodo.org/account/settings/github/> and sign in with
the same GitHub account that owns this repo (`telleroutlook`).

### 2. Toggle the repository

You'll see a list of your GitHub repos. Find `trace-pipeline` and
toggle the switch to **ON**. Zenodo now watches for new releases.

### 3. Re-publish the v0.1.0 release

Zenodo only picks up releases created **after** the toggle is on.
The `v0.1.0` release we already created (`gh release create v0.1.0`)
predates the Zenodo connection.

**Option A — easy, no version bump**: delete and re-create v0.1.0:

```bash
# Locally
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0

# Then re-create
git tag -a v0.1.0 -m "v0.1.0 — initial public release"
git push origin v0.1.0

gh release create v0.1.0 \
    --repo WasmAgent/trace-pipeline \
    --title "v0.1.0 — Initial public release" \
    --notes-file CHANGELOG.md
```

**Option B — cleaner, version bump**: tag a new `v0.1.1`:

```bash
git tag -a v0.1.1 -m "v0.1.1 — first DOI-bearing release"
git push origin v0.1.1
gh release create v0.1.1 ...
```

Either way, Zenodo will detect the GitHub release within ~1 minute
and create a DOI.

### 4. Get the DOI

Zenodo emails the new DOI when ready. You can also check at
<https://zenodo.org/me/uploads>.

The DOI looks like `10.5281/zenodo.XXXXXXX`.

### 5. Update repo files

Once you have the DOI, edit four places:

#### 5a. `README.md`

Add a Zenodo badge near the existing badges:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

The full markdown for the badge bar should now look like:

```markdown
[![CI](.../ci.yml/badge.svg)](.../ci.yml)
[![License: Apache 2.0](...)](LICENSE)
[![Python: 3.10+](...)](https://www.python.org)
[![Paper PDF](...)](papers/eval_trust/draft.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

#### 5b. `CITATION.cff`

Add a `doi:` line at top level and a `doi:` field in
`preferred-citation`:

```yaml
cff-version: 1.2.0
title: ...
doi: 10.5281/zenodo.XXXXXXX     # <-- new
preferred-citation:
  type: software
  doi: 10.5281/zenodo.XXXXXXX   # <-- new
  ...
```

#### 5c. Update the BibTeX in `README.md`

```bibtex
@misc{evaltrust2026,
  title  = {Silent Contamination in {LLM} Merging Evaluation: ...},
  author = {{telleroutlook}},
  year   = {2026},
  doi    = {10.5281/zenodo.XXXXXXX},
  url    = {https://doi.org/10.5281/zenodo.XXXXXXX},
}
```

#### 5d. (optional) `papers/eval_trust/draft.md`

If this software DOI should appear alongside the arXiv ID in the paper
itself, add it to Appendix B (Reproducibility):

```markdown
- **Code DOI:** 10.5281/zenodo.XXXXXXX (this release)
- **Paper:** arXiv:26XX.YYYYY (after endorsement)
```

Then rerun `prepare_arxiv.sh` to refresh `draft.pdf` + tar.

### 6. Commit + push

```bash
git add README.md CITATION.cff papers/eval_trust/
git commit -m "docs: add Zenodo DOI 10.5281/zenodo.XXXXXXX"
git push
```

## Concept DOI vs version DOI

Zenodo issues **two kinds** of DOI per repo:

- **Concept DOI** — points to the *latest* release. Use this in
  long-form references that should always resolve to the current
  version.
- **Version DOI** — points to a specific release (`v0.1.0` etc.).
  Use this when reproducibility is critical: a reviewer in 2030 should
  be able to fetch *exactly* what you cite.

Zenodo's badge URL gives the concept DOI by default. Version DOIs are
listed under <https://zenodo.org/me/uploads> for each release.

## Future releases

Once the toggle is on, every subsequent release tagged + pushed via
GitHub gets its own version DOI automatically. You don't need to do
anything beyond `gh release create`.

## Troubleshooting

- **Zenodo doesn't pick up my release**: check
  <https://zenodo.org/account/settings/github/> shows the repo as
  active. The first sync after enabling can take a few minutes.
- **DOI badge shows broken image**: the URL must be exact. Check that
  the number in `https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg`
  matches the number in `https://doi.org/10.5281/zenodo.XXXXXXX`.
- **The release is private**: Zenodo only archives public GitHub
  releases. Make sure the release is not in draft mode.
