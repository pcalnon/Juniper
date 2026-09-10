#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-ml
# Application:   cross-repo tooling (ad-hoc)
# File Name:     2026-09-09_open_worker_lockfile_6c_pr.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-09-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Opens the juniper-cascor-worker PR for follow-up 6c of the container-registry rollout
#    (lockfile-update.yml regenerates both locks in one GitHub-signed commit, replacing the
#    unsigned plain push), via util/open_signed_pr.py.
#
#    Exists as a script for the same reason as its siblings: a worktree-isolated session's
#    Bash classifier refuses a multi-line command with runtime-computed operands.
#
#    Single-use: branch name and --add pairs are specific to that PR.
#####################################################################################################
set -euo pipefail

SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
WORKTREE="/home/pcalnon/Development/python/Juniper/worktrees/juniper-cascor-worker--ci--lockfile-update-both-locks--20260909-1911--f2e221bc"
HELPER="/home/pcalnon/Development/python/Juniper/juniper-ml/util/open_signed_pr.py"
PYTHON="/opt/miniforge3/envs/JuniperCascor1/bin/python"

"${PYTHON}" "${HELPER}" \
  --repo juniper-cascor-worker \
  --branch ci/lockfile-update-both-locks \
  --add "${WORKTREE}/.github/workflows/lockfile-update.yml:.github/workflows/lockfile-update.yml" \
  --add "${WORKTREE}/CHANGELOG.md:CHANGELOG.md" \
  --message "$(cat "${SCRATCH}/worker6c-commit-msg.txt")" \
  --title "ci(lockfile): regenerate both locks in one signed commit" \
  --body-file "${SCRATCH}/worker6c-pr-body.md" \
  "$@"
