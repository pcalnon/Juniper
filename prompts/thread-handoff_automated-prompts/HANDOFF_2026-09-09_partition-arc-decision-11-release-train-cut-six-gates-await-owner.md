# HANDOFF — decision 11's release train is cut; six PyPI gates await the owner, two trains remain

**Date**: 2026-09-09 · **Session**: <https://claude.ai/code/session_014FMjGiN9yK9ppfkUiBnao5>
**Worktree**: `/home/pcalnon/Development/python/Juniper/juniper-ml/.claude/worktrees/fancy-marinating-nova`
**Branch**: `worktree-fancy-marinating-nova` (at `origin/main`; no PR of its own — every change this
session shipped as an API-signed PR, see §0.1)
**Predecessor**: `prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-07_partition-arc-decision-11-shipped-and-the-release-train.md`
(its §2 invariant, §3 straggler table and §6 traps still hold; this document supersedes its §1 next
actions and §4 release table).

**Documents REFERENCED** (the ecosystem convention in
`/home/pcalnon/Development/python/Juniper/AGENTS.md` § Cross-Project Conventions requires the filename
on every citation, because more than one document is cited):

- `notes/JUNIPER_2026-08-29_JUNIPER-ECOSYSTEM_TRAIN-EVAL-TEST-PARTITION-DESIGN.md` — design of record
- `notes/JUNIPER_2026-08-30_JUNIPER-ECOSYSTEM_PARTITION-IMPLEMENTATION-PLAN.md` — rollout plan; **§9 is
  the live register**
- `notes/JUNIPER_2026-06-18_JUNIPER-ECOSYSTEM_PYPI-PUBLISH-PROCEDURE.md` — §11, the release ceremony
- `notes/JUNIPER_2026-02-23_JUNIPER-ML_THREAD-HANDOFF-PROCEDURE.md` — this document's template
- `util/release_train/registry.yaml`, `util/release_train/{detect,propose,ceremony}.py` — the instrument

**Documents CHANGED by this session** (juniper-ml): `juniper-model-core/juniper_model_core/crossval/splits.py`,
`juniper-model-core/CHANGELOG.md`, `util/ad-hoc/2026-09-08_push_signed_commit.py`,
`util/ad-hoc/2026-09-08_changelog_insert.py` (all in #1829); `util/ad-hoc/2026-09-08_bump_version_carriers.py`,
`util/ad-hoc/2026-09-08_insert_after_line.py` and this file (the handoff PR, §8);
`notes/releases/RELEASE_NOTES_{juniper-data-client_v0.5.0,juniper-data_v0.14.0,juniper-cascor_v0.11.0,juniper-canopy_v0.7.0,juniper-recurrence-model_v0.3.0,juniper-recurrence-client_v0.3.0}.md`
(the ceremony's exempt archive PRs #1838–#1842 and #1844). Sibling repos: §1's table.

---

## 0. PREFLIGHT

1. **Local `git commit` HANGS in this environment.** `commit.gpgsign=true` with a YubiKey-resident
   key; a `timeout 40 git commit --allow-empty` exits 124. Every commit this session went through the
   GitHub API (`createCommitOnBranch`, GitHub-signed): `util/open_signed_pr.py` for a new branch,
   `util/ad-hoc/2026-09-08_push_signed_commit.py` for a follow-up commit on an existing one. Do not
   run `git commit`, `git tag` or `gh release create --verify-tag` locally; do not `git push`.
2. **Six packages are on TestPyPI, cut as GitHub Releases, and parked at the `pypi` environment
   gate.** That gate is the owner's, never the session's (memory `feedback_deploy_approvals_paul_manages`).
   Until each is approved, `pip install` from PyPI still serves the pre-decision-11 versions and
   `POST /v1/crossval` in a PyPI-installed juniper-recurrence stays broken. The run URLs are in §2.
3. **`util/release_train/propose.py` and `ceremony.py` read the sibling checkouts on disk as inputs.**
   `git -C /home/pcalnon/Development/python/Juniper/<repo> pull --ff-only origin main` before EVERY
   run, or the whole-file API commit carries stale content. All five siblings were at `origin/main`
   at handoff (§8).
4. **The sub-agents this session spawned were killed by a session rate limit mid-task**, twice.
   They had already opened their PRs; their worktrees were removed after merge. If you spawn agents,
   give them private scratch paths — one overwrote my `scratchpad/commit_body.txt`
   (memory `reference_headless_commit_signing_hangs_use_api_commits`).
5. **Correction to the predecessor's §1: decision 5 IS implemented.** It said cascor's CLI has "zero
   `X_val`" because it grepped `src/main.py`; the plumbing is `src/spiral_problem/spiral_problem.py:1346-1440`
   (cascor#622 carves `train 0.8 / val 0.1 / test 0.1`, `_SPIRAL_PROBLEM_VAL_RATIO = 0.1`, and passes
   `x_val` to `fit()`). What remains unimplemented is **V-3, the measurement**, which cascor 0.11.0's
   changelog says out loud. Decision 12 is still unimplemented (`partition_provenance`: zero hits).
6. **Environment**: conda envs `JuniperCascor1` / `JuniperCanopy1` / `JuniperData`; canopy python needs
   `env -u LD_LIBRARY_PATH`; juniper-recurrence has no env. The worktree-isolated command guard refuses
   `sed` programs (`Nr file`, `a\`), loops that call `gh`/`git`, `$var` paths handed to `python`/`sed`,
   `env -u` inside loops, and here-docs — use the four `util/ad-hoc/2026-09-08_*.py` helpers instead.

---

## 1. Goal statement

Continue the decision-11 release train (design §9.5). **Wave 1 is cut: six of eight trains are at the
owner's PyPI gate.** Remaining: the owner's six approvals; the recurrence app's floor bump + 0.5.0;
juniper-ml's floors + 0.8.0; the documentation that records the released versions.

**Merged and verified this session** (each read back from `origin/main` or the GitHub API):

| repo | PR | what |
| --- | --- | --- |
| juniper-cascor | #631 | CHANGELOG entries for #614/#616/#620/#621/#622/#623/#625/#618/#629 — filed under `### Removed` so the renderer marks the release BREAKING |
| juniper-cascor | #635 | **0.11.0** bump (propose.py) |
| juniper-data-client | #193 | `[Unreleased]` and three docs said `"full"` stayed in `NPZ_SPLITS`; straggler S-3 |
| juniper-data-client | #194 | **0.5.0** bump + 16 carriers (`__init__`, twelve `Version:` headers, three docs headers) + the pip-audit CI fix (§6) |
| juniper-data | #386 | decision-11 `### Removed` entry (re-cut of #384, which #385 conflicted); straggler S-8 |
| juniper-data | #389 | **0.14.0** bump + `juniper_data/__init__.py` fallback literal |
| juniper-canopy | #602 | `juniper-data-client` ceiling `<0.6.0` (a hand-made follow-on: canopy's pin is in an extra, so the train opens none) |
| juniper-canopy | #604 | **straggler S-6 FIXED**: sequence installs were train-only since #369; `_whole_dataset` concatenates (1-D safe), restores entity-major order via `ticker_code_*`; S-4 doc |
| juniper-canopy | #606 | **0.7.0** bump + three fallback literals (`src/__init__.py`, `juniper_canopy/__init__.py`, `resolve_app_version`) |
| juniper-recurrence | #152 | stragglers S-5a/S-5b/S-5c/S-7: `DatasetRef.split` is `Literal["train","val","test","full"]` (422 at the edge), `--split` help names `val`, READMEs/docstring stop naming `_full`; model CHANGELOG gets its `derive_full_split` entry |
| juniper-recurrence | #156 | app `juniper-data-client` ceiling `<0.6.0` (train follow-on) |
| juniper-recurrence | #158 | **juniper-recurrence-model 0.3.0** bump |
| juniper-recurrence | #161 | **juniper-recurrence-client 0.3.0** bump + sub-package `AGENTS.md` header |
| juniper-ml | #1829 | straggler S-2 (`crossval/splits.py` docstring) + two release-train helpers |
| juniper-ml | #1838–#1842, #1844 | exempt notes-archive PRs, one per Release (auto-merge armed) |

**Open at handoff**: juniper-recurrence#159 — app `juniper-recurrence-model` ceilings `<0.4.0`
(dependencies, `[torch]`, `[bench-torch]`), auto-merge armed; it went BEHIND four times as this
contended lane moved and was refreshed with `update-branch` each time (§7 checks it). Closed unmerged:
juniper-data#384 (superseded by #386), juniper-recurrence#155 (anyio fix; another session's #154 landed
the identical filter six minutes earlier).

**Next actions, in order.** (1) Owner approves the six gates in §2. (2) §3 — the app's floor bump and
0.5.0, then juniper-ml's floors and 0.8.0. (3) §4 — documentation. (4) §5 — carried forward.

---

## 2. Six deployments pending owner approval

Each run is parked with `Publish to TestPyPI = success` and `Publish to PyPI = waiting`. TestPyPI serves
the version (HTTP 200 on `https://test.pypi.org/pypi/<pkg>/<ver>/json`); PyPI does not yet (404).

| package | Release | publish run (approve here) | archive PR |
| --- | --- | --- | --- |
| juniper-data-client 0.5.0 | `v0.5.0` | <https://github.com/pcalnon/juniper-data-client/actions/runs/34322900465> | juniper-ml#1838 |
| juniper-data 0.14.0 | `v0.14.0` | <https://github.com/pcalnon/juniper-data/actions/runs/34322906502> | juniper-ml#1839 |
| juniper-cascor 0.11.0 | `v0.11.0` | <https://github.com/pcalnon/juniper-cascor/actions/runs/34322913355> | juniper-ml#1840 |
| juniper-canopy 0.7.0 | `v0.7.0` | <https://github.com/pcalnon/juniper-canopy/actions/runs/34322919025> | juniper-ml#1841 |
| juniper-recurrence-model 0.3.0 | `juniper-recurrence-model-v0.3.0` | <https://github.com/pcalnon/juniper-recurrence/actions/runs/34323535743> | juniper-ml#1842 |
| juniper-recurrence-client 0.3.0 | `juniper-recurrence-client-v0.3.0` | <https://github.com/pcalnon/juniper-recurrence/actions/runs/34324270593> | juniper-ml#1844 |

The ceremony's `detect` reads "released" from PyPI, so every row reads `BUMPED_NOT_RELEASED` until the
gate is approved, then `UP_TO_DATE`. The gate also has a 5-minute wait timer; approval during the timer
proceeds when it expires. **Order of approval does not matter for these six** — no floor among them
points at another (the app's floor bump is §3a and comes after).

---

## 3. The two trains still to run, in dependency order

**3a. juniper-recurrence (app) 0.5.0** — needs #159 merged AND `juniper-recurrence-model 0.3.0` **on
PyPI** (the app lane runs `pip install -e ".[test]"`, which resolves the model from PyPI, so a floor at an
unpublished version is red CI). Then, as ONE PR: `juniper-recurrence/pyproject.toml` floors
`juniper-recurrence-model>=0.3.0,<0.4.0` in `dependencies`, `[torch]` and `[bench-torch]`, plus an app
CHANGELOG `### Changed` bullet ("requires juniper-recurrence-model 0.3.0 — `derive_full_split` is what keeps
`POST /v1/crossval` alive on post-#369 artifacts"). No lockfile in this repo. Merge, pull, then
`propose.py --package juniper-recurrence --execute --cross-repo` (bumps `_version.py`, the app CHANGELOG,
the root `AGENTS.md` **Version** header 0.4.0 → 0.5.0 and the app row; `scripts/check_version_drift.py`
checks all three), merge, pull, ceremony (tag `juniper-recurrence-v0.5.0`). juniper-ml's `<0.5.0` cap on
the app moves in 3b.

```bash
git -C /home/pcalnon/Development/python/Juniper/juniper-recurrence pull --ff-only origin main
python3 util/release_train/detect.py --repo-root . --ecosystem-root /home/pcalnon/Development/python/Juniper \
  --package juniper-recurrence --json > /tmp/m.json
python3 util/release_train/propose.py --manifest /tmp/m.json --package juniper-recurrence --repo-root . \
  --ecosystem-root /home/pcalnon/Development/python/Juniper --cross-repo --release-date $(date -u +%F)   # dry-run first; then --execute
```

**3b. juniper-ml 0.8.0** — after ALL seven are on PyPI (floors resolve nothing otherwise). One PR
carrying the pin and its lockstep artifacts — `tests/test_pyproject_extras.py` asserts exact strings and
`ExtrasDocsLockstepTest` asserts the four tables:

| file | lines | change |
| --- | --- | --- |
| `pyproject.toml` | 30, 47, 48, 49 | `juniper-data-client>=0.5.0`, `juniper-canopy>=0.7.0`, `juniper-cascor>=0.11.0`, `juniper-data>=0.14.0` |
| `pyproject.toml` | 67, 68, 69 | `juniper-recurrence-model>=0.3.0,<0.4.0`, `juniper-recurrence>=0.5.0,<0.6.0`, `juniper-recurrence-client>=0.3.0,<0.4.0` |
| `tests/test_pyproject_extras.py` | 108, 115–117, 131–133 | the same seven strings |
| `AGENTS.md` | 362, 364, 367 + `**Last Updated**` | extras table rows |
| `README.md` | 85, 87, 90 | extras table rows |
| `docs/QUICK_START.md` | 58, 60, 63 | extras table rows |
| `docs/REFERENCE.md` | 105, 108–110, 118–120; 168–172 | split-column rows; add a `0.8.x` compatibility-matrix row |
| `CHANGELOG.md` | `[Unreleased]` | one `### Changed` bullet naming the seven floors and why |

Then `propose.py --package juniper-ml --execute` (in-repo: it also folds the meta ceiling co-changes),
merge, ceremony (`publish.yml`, tag `v0.8.0`). Line numbers are as of `b26acd62`; re-grep, other
sessions edit `AGENTS.md` daily.

**Optional**: `juniper-model-core` 0.3.2 — the S-2 docstring is fixed on main but the published 0.3.1
wheel is stale. `detect` will offer it as a patch. Also UNRELEASED per `detect` but outside this arc:
`juniper-cascor-model` 0.2.0, `juniper-observability` 0.5.0.

---

## 4. Documentation owed once the wheels are on PyPI

- juniper-ml `docs/REFERENCE.md` ~`:5848` ("Decision 11 SHIPPED") — add the released versions; ~`:5900`
  — the model-core docstring is fixed on main (#1829); ~`:5901` — S-6 is FIXED (canopy#604); the version
  history table ~`:7161`.
- `docs/DEVELOPER_CHEATSHEET_JUNIPER-ML.md:317` and `:769` — same.
- `/home/pcalnon/Development/python/Juniper/AGENTS.md` § Data Contract — add "released as juniper-data
  0.14.0 / data-client 0.5.0 / cascor 0.11.0 / canopy 0.7.0 / recurrence-model 0.3.0 /
  recurrence-client 0.3.0 …" (unversioned file: edit in place, no PR).
- `notes/JUNIPER_2026-08-30_JUNIPER-ECOSYSTEM_PARTITION-IMPLEMENTATION-PLAN.md` §9 — release status row.
- The SemVer ruling's last step (memory `feedback_semver_beats_consumer_cap_2026-09-05`): verify each
  **published** wheel in a clean venv, not the checkout — e.g. `pip install juniper-data-client==0.5.0`
  and read `NPZ_SPLITS`; `pip install juniper-recurrence-model==0.3.0` and import `derive_full_split`.

---

## 5. Carried forward (unchanged from the predecessor unless noted)

| item | state |
| --- | --- |
| S-1 hf/kaggle stores (two-way cut, `*_full`, `generator_version="1.0.0"`) | documented in juniper-data 0.14.0's changelog as a known gap; product decision open |
| S-2 | fixed on main (#1829); published model-core wheel stale |
| S-3, S-4, S-5a/b/c, S-6, S-7, S-8 | **all fixed** (#193, #604, #152, #604, #152, #386) |
| plan R-2 / R-9 / S-5 / S-7 (canopy#559 OPEN), Chunk 5, Chunk 7 (re-baseline + snapshot provenance) | undisposed |
| Decision 12 (`partition_provenance`) | unimplemented; design §9.6.3 / §9.6.6 |
| Decision 5 | **implemented** (PREFLIGHT 5); V-2 / V-3 unmeasured, stated in cascor 0.11.0's notes |

---

## 6. Traps this session added (memory `reference_release_train_ceremony_traps_2026-09-09` has the long form)

- **data-client's Security Scans job fails on every version bump**: `pip freeze | grep` misses the PEP 660
  editable line, pip-audit audits the unpublished version. Fixed in #194 with
  `pip list --format=freeze --exclude juniper-data-client`. juniper-data's grep did not trip.
- **`propose.py` bumps `pyproject`/`_version.py`/CHANGELOG/root `AGENTS.md` only.** Carriers it misses:
  data-client `__init__.py` + twelve `Version:` headers (three spellings) + three docs headers; data and
  canopy `__init__.py` fallbacks; canopy `resolve_app_version()` and `juniper_canopy/__init__.py`;
  recurrence-client's sub-package `AGENTS.md`. `2026-09-08_bump_version_carriers.py` prepares them with
  an exact-count guard.
- **The ceremony gate reads the newest COMPLETED main run**; a concurrency-cancelled run halts it. Wait
  for the merge commit's own run. juniper-data's `ci.yml` never runs on push to main (gate passes on a
  weeks-old success).
- **The ceremony monitor died on a transient API timeout AFTER cutting juniper-data's Release** (exit 2).
  Verify `gh release view`, the archive PR and the publish run's jobs; never re-cut.
- **A proposal PR's CHANGELOG hunk sits at the top of `[Unreleased]`** and conflicts with any other PR
  adding an entry there; a superset follow-up commit does NOT resolve it (git sees two different inserts
  at one anchor) — re-cut from current main (data#384 → #386).
- **`gh pr merge --auto` on an already-green PR merges immediately** (data-client#193 did) — arm only
  after the review. A PR reading `CLEAN` with auto-merge armed merges within ~3 minutes on its own; a
  manual `gh pr merge` in that window races it ("Base branch was modified") harmlessly.
- **anyio 4.15 broke the recurrence app lane at collection** (`anyio.abc.BlockingPortal` deprecation under
  warnings-as-errors); fixed by #154 (another session) — re-run the dup-guard right before opening.
- **canopy main went red once on a timing flake** (`test_x7_loop_responsiveness` on the 3.13 job only,
  the other three matrix jobs green); the next main run cleared it. Check the failing test name before
  treating a red main as a defect.
- The predecessor's §6 traps (CodeQL threads, squash-first-commit, `Allow-Symbol-Loss`, `gh` 2.46
  `pr edit`) all still apply.

---

## 7. Verify the starting state

```bash
cd /home/pcalnon/Development/python/Juniper/juniper-ml/.claude/worktrees/fancy-marinating-nova
git fetch -q origin && git status --short && git log --oneline -1 origin/main
# The one recurrence PR this handoff could not wait for:
gh pr view 159 --repo pcalnon/juniper-recurrence --json state,mergedAt,mergeStateStatus
# Gate state per package (waiting = still the owner's; success = approved). Control: the run id must resolve.
gh run view 34322900465 --repo pcalnon/juniper-data-client --json jobs --jq '.jobs[] | "\(.name)\t\(.status)\t\(.conclusion)"'
gh run view 34324270593 --repo pcalnon/juniper-recurrence   --json jobs --jq '.jobs[] | "\(.name)\t\(.status)\t\(.conclusion)"'
# PyPI truth (200 once approved; TestPyPI is already 200 for all six):
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/juniper-recurrence-model/0.3.0/json
curl -s -o /dev/null -w '%{http_code}\n' https://test.pypi.org/pypi/juniper-recurrence-model/0.3.0/json
# Train view (exit 1 whenever anything is not UP_TO_DATE -- normal):
python3 util/release_train/detect.py --repo-root . --ecosystem-root /home/pcalnon/Development/python/Juniper \
  --package juniper-data-client --package juniper-data --package juniper-cascor --package juniper-canopy \
  --package juniper-recurrence-model --package juniper-recurrence-client --package juniper-recurrence
# Archive PRs (exempt; auto-merge armed):
gh pr list --repo pcalnon/juniper-ml --state all --search "release-notes: in:title" --limit 7
```

---

## 8. Git status at handoff

This worktree is at `origin/main` (`b26acd62` when written) with the handoff PR's three files
(this file, `util/ad-hoc/2026-09-08_bump_version_carriers.py`, `util/ad-hoc/2026-09-08_insert_after_line.py`)
uploaded through the API — the local copies are untracked, never committed locally. No local branch
carries work. The five sibling primary checkouts
(`/home/pcalnon/Development/python/Juniper/{juniper-data-client,juniper-data,juniper-cascor,juniper-canopy,juniper-recurrence}`)
are clean on `main` and were pulled to `origin/main` before the last propose/ceremony run; pull again
before trusting them. The four agent worktrees this session created under
`/home/pcalnon/Development/python/Juniper/worktrees/` were removed and pruned after their PRs merged;
the three `*--feature--drop-full-family--20260905-*` worktrees inherited from 2026-09-05 were left
alone, as before.

---

## 9. What this evidence cannot support

- **No wheel has been installed from PyPI**: the "released" claim is TestPyPI + a waiting PyPI job. The
  SemVer ruling's last step — verify the *published* wheel in a clean venv — is owed after approval (§4).
- **The recurrence app's `Literal` split narrowing (#152) had one pre-existing local failure**
  (`test_docs_require_auth_when_enabled`, service-core 0.5.0 installed locally vs `>=0.6.0` required); CI
  ran with a fresh install and was green. Not a defect of #152.
- **Wave 1 releases carry whatever else was on each `main` at bump time** (cascor#632-class fixes
  from other sessions that merged before the bump are in; those that merged after are not). The
  changelog is the record; nothing was hand-selected.
- **No consensus validation was run on this document** (the session limit terminated four sub-agents;
  spawning five more was not attempted). Treat §3's line numbers and §5's dispositions as claims to
  re-derive with §7.
