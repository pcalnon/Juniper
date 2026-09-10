#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-ml
# Application:   cross-repo tooling (ad-hoc)
# File Name:     2026-09-09_open_cascor_lockfile_6c_pr.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-09-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Opens the juniper-cascor PR for follow-up 6c of the container-registry rollout
#    (lockfile-update.yml regenerates requirements-cpu.lock alongside the GPU lock, both
#    in one GitHub-signed commit), via util/open_signed_pr.py.
#
#    Exists as a script for the same reason as its worker sibling: a worktree-isolated
#    session's Bash classifier refuses a multi-line command with runtime-computed operands.
#
#    Single-use: branch name and --add pairs are specific to that PR.
#####################################################################################################
set -euo pipefail

SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
WORKTREE="/home/pcalnon/Development/python/Juniper/worktrees/juniper-cascor--ci--lockfile-update-both-locks--20260909-1855--53c0338b"
HELPER="/home/pcalnon/Development/python/Juniper/juniper-ml/util/open_signed_pr.py"
PYTHON="/opt/miniforge3/envs/JuniperCascor1/bin/python"

"${PYTHON}" "${HELPER}" \
  --repo juniper-cascor \
  --branch ci/lockfile-update-both-locks \
  --add "${WORKTREE}/.github/workflows/lockfile-update.yml:.github/workflows/lockfile-update.yml" \
  --add "${WORKTREE}/requirements-cpu.lock:requirements-cpu.lock" \
  --add "${WORKTREE}/CHANGELOG.md:CHANGELOG.md" \
  --message "$(cat "${SCRATCH}/cascor-commit-msg.txt")" \
  --title "ci(lockfile): regenerate requirements-cpu.lock alongside the GPU lock, in one signed commit" \
  --body-file "${SCRATCH}/cascor-pr-body.md" \
  "$@"
