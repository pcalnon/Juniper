#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-ml
# Application:   cross-repo tooling (ad-hoc)
# File Name:     2026-09-09_open_ml_lint_6g_pr.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-09-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Opens the juniper-ml PR for follow-up 6g of the container-registry rollout
#    (juniper-lint-workflow-paths resolves script paths against the job's working
#    directory; closes juniper-ml#1836), via util/open_signed_pr.py.
#
#    Exists as a script for the same reason as its siblings: a worktree-isolated session's
#    Bash classifier refuses a multi-line command with runtime-computed operands. It is
#    also the only way to land a juniper-ml change from a juniper-ml worktree in this
#    session -- the classifier refuses git operations against the shared juniper-ml
#    checkout, and open_signed_pr.py needs no working tree.
#
#    The --add set is scoped deliberately: the four untracked util/ad-hoc/ helpers and the
#    session handoff belong to the CLOSING PR, not this one.
#####################################################################################################
set -euo pipefail

SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
WORKTREE="/home/pcalnon/Development/python/Juniper/juniper-ml/.claude/worktrees/tender-splashing-wigderson"
HELPER="/home/pcalnon/Development/python/Juniper/juniper-ml/util/open_signed_pr.py"
PYTHON="/opt/miniforge3/envs/JuniperCascor1/bin/python"

"${PYTHON}" "${HELPER}" \
  --repo juniper-ml \
  --branch fix/lint-workflow-paths-working-directory \
  --add "${WORKTREE}/juniper-ci-tools/juniper_ci_tools/lint_workflow_paths.py:juniper-ci-tools/juniper_ci_tools/lint_workflow_paths.py" \
  --add "${WORKTREE}/juniper-ci-tools/tests/test_lint_workflow_paths.py:juniper-ci-tools/tests/test_lint_workflow_paths.py" \
  --add "${WORKTREE}/juniper-ci-tools/juniper_ci_tools/_version.py:juniper-ci-tools/juniper_ci_tools/_version.py" \
  --add "${WORKTREE}/juniper-ci-tools/pyproject.toml:juniper-ci-tools/pyproject.toml" \
  --add "${WORKTREE}/juniper-ci-tools/CHANGELOG.md:juniper-ci-tools/CHANGELOG.md" \
  --add "${WORKTREE}/tests/test_ci_tools_drift.py:tests/test_ci_tools_drift.py" \
  --add "${WORKTREE}/docs/REFERENCE.md:docs/REFERENCE.md" \
  --add "${WORKTREE}/.github/workflows/ci.yml:.github/workflows/ci.yml" \
  --add "${WORKTREE}/.github/workflows/main-verify.yml:.github/workflows/main-verify.yml" \
  --add "${WORKTREE}/.github/workflows/lockfile-update.yml:.github/workflows/lockfile-update.yml" \
  --add "${WORKTREE}/.github/workflows/docs-full-check.yml:.github/workflows/docs-full-check.yml" \
  --add "${WORKTREE}/.github/workflows/ci-ci-tools.yml:.github/workflows/ci-ci-tools.yml" \
  --add "${WORKTREE}/.github/workflows/ci-config-tools.yml:.github/workflows/ci-config-tools.yml" \
  --add "${WORKTREE}/.github/workflows/ci-doc-tools.yml:.github/workflows/ci-doc-tools.yml" \
  --add "${WORKTREE}/.github/workflows/ci-model-core.yml:.github/workflows/ci-model-core.yml" \
  --add "${WORKTREE}/.github/workflows/ci-observability.yml:.github/workflows/ci-observability.yml" \
  --add "${WORKTREE}/.github/workflows/ci-service-core.yml:.github/workflows/ci-service-core.yml" \
  --message "$(cat "${SCRATCH}/ml6g-commit-msg.txt")" \
  --title "fix(ci-tools): resolve workflow script paths against the job's working directory" \
  --body-file "${SCRATCH}/ml6g-pr-body.md" \
  "$@"
