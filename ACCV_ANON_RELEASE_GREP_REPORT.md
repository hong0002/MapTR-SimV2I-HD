# ACCV Anonymous Release Grep Report

- Repository path: `${MAPTR_ROOT}` for the audited SimV2I-HD MapTR working tree.
- Timestamp: 2026-07-04 21:00:40 KST
- Training/evaluation: not run.

## Scope

Release-bound text scan included Markdown, Python, shell, YAML, JSON, and TeX
files under the repository, with generated artifacts excluded.

Files scanned:

- Broad release-style file list after exclusions: 1002 files.
- Focused public-release paths: `README.md`, `INSTALL.md`, `DATA.md`,
  `REPRODUCE.md`, `docs/`, `scripts/`, `integrations/`, `tools/`, and
  `projects/`.

Excluded directories and artifacts:

- `.git/`
- `data/`
- `outputs/`
- `work_dirs/`
- `checkpoints/`
- `ckpts/`
- `pretrained/`
- `logs/`
- `wandb/`
- `tensorboard/`
- `__pycache__/`
- checkpoint, pickle, NumPy, image, and video artifacts

## Commands Used

The audit used `rg` with the exclusion globs above for these command families:

- Identity/private-path scan: private local roots, user identifiers, personal
  repository URLs, and deferred-work markers.
- Email scan: standard email-address regular expression.
- Geo-result leakage scan: geo AP/mAP wording, exact near-zero geo stress-test
  values, geo split counts, and ROI-free wording.
- Portable-path scan: local absolute path roots such as home, mount, Windows
  drive, and data-volume prefixes.
- Claim-safety scan: strong V2I/generalization claims, test-selected checkpoint
  wording, and misleading geographic split descriptions.

Scratch outputs were written under `/tmp/maptr_anon_*`.

## Findings

### MUST_FIX

- Release docs contained old V2I design-stub wording that could make the public
  snapshot look incomplete or outdated.
- The inactive V2I reference config contained deferred-work wording.
- Main public docs did not consistently name the primary 20k split as the
  scenario-disjoint controlled split.

### FIXED

- `README.md`
  - Added explicit scenario-disjoint controlled split wording for the main
    paper dataset.
- `DATA.md`
  - Added explicit scenario-disjoint controlled split wording for the main
    ACCV experiments.
- `REPRODUCE.md`
  - Added explicit scenario-disjoint controlled split wording for the public
    reproduction path.
- `docs/MAPTR_SERVER_SETUP.md`
  - Replaced outdated V2I design-stub references with current executable
    Dynamic Top-4 and Pose-gated Top-4 config references.
- `docs/MAPTR_V2I_PLAN.md`
  - Rewrote the note as current V2I design documentation.
  - Added safe wording that the geo subset is a diagnostic stress protocol, not
    the main benchmark.
- `integrations/maptr/configs/simv2i_maptr_v2i_placeholder.py`
  - Reworded it as an inactive reference config and removed deferred-work
    wording from the file contents.

### OK_WITH_CAVEAT

- Broad scan still finds deferred-work markers in legacy MapTR/MMDetection3D
  source comments under `tools/`, `projects/`, and `mmdetection3d/`.
  These are code comments in vendored or legacy integration code, not anonymous
  identity leaks. They were not modified because the request forbids changing
  model, dataset, evaluation, or training logic.
- Broad scan finds three email-address matches inside vendored
  `mmdetection3d/` metadata/source. These are upstream contact metadata, not
  SimV2I-HD author identities. They remain only as a caveat if the full vendor
  tree is included in an anonymous ZIP.
- Generated geo diagnostic reports under `outputs/` contain geo stress-test
  values, but `outputs/` is excluded from release-bound scanning and should not
  be included in the anonymous main-paper source.

### FALSE_POSITIVE

- `tools/create_geo_subset_split.py` contains ROI-overlap computation utilities.
  This is a diagnostic computation, not a claim that the geo split is
  ROI-overlap-free.
- Generic overlap terms in MapTR/MMDetection3D internals are not paper claims.

### REMAINING

- No private local path matches remain in focused public-release paths.
- No personal repository URL matches remain.
- No email matches remain outside vendored `mmdetection3d/`.
- No geo AP/mAP or exact geo stress-test values remain outside excluded
  `outputs/`.
- No misleading geographic-generalization claims were found in focused
  public-release paths.
- Deferred-work markers remain in legacy/vendored code comments only.

## Modified Files

- `README.md`
- `DATA.md`
- `REPRODUCE.md`
- `docs/MAPTR_SERVER_SETUP.md`
- `docs/MAPTR_V2I_PLAN.md`
- `integrations/maptr/configs/simv2i_maptr_v2i_placeholder.py`
- `ACCV_ANON_RELEASE_GREP_REPORT.md`

## Final Verdict

PASS_WITH_CAVEATS

The public docs/scripts/config wrappers no longer expose private paths,
personal identifiers, geo stress-test numbers, or risky main-paper geo-result
wording. Remaining broad-scan caveats are confined to legacy/vendored code
comments and upstream MMDetection3D contact metadata.
