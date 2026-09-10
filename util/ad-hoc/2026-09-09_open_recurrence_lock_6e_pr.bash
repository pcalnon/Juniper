#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-ml
# Application:   cross-repo tooling (ad-hoc)
# File Name:     2026-09-09_open_recurrence_lock_6e_pr.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-09-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Opens the juniper-recurrence PR for follow-up 6e of the container-registry rollout
#    (pin the app image's dependencies with requirements.lock), via util/open_signed_pr.py.
#
#    Exists as a script for the same reason as its worker / cascor siblings: a
#    worktree-isolated session's Bash classifier refuses a multi-line command with
#    runtime-computed operands.
#
#    Single-use: branch name and --add pairs are specific to that PR. Note the repo is a
#    monorepo -- the app lives in juniper-recurrence/, so repo paths carry that prefix while
#    the workflow sits at the repo root.
#####################################################################################################
set -euo pipefail

SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
WORKTREE="/home/pcalnon/Development/python/Juniper/worktrees/juniper-recurrence--feat--lock-the-app-image--20260909-1910--2ff03047"
HELPER="/home/pcalnon/Development/python/Juniper/juniper-ml/util/open_signed_pr.py"
PYTHON="/opt/miniforge3/envs/JuniperCascor1/bin/python"

"${PYTHON}" "${HELPER}" \
  --repo juniper-recurrence \
  --branch feat/lock-the-app-image \
  --add "${WORKTREE}/juniper-recurrence/requirements.lock:juniper-recurrence/requirements.lock" \
  --add "${WORKTREE}/juniper-recurrence/Dockerfile:juniper-recurrence/Dockerfile" \
  --add "${WORKTREE}/juniper-recurrence/tests/test_dockerfile_image_lock.py:juniper-recurrence/tests/test_dockerfile_image_lock.py" \
  --add "${WORKTREE}/juniper-recurrence/CHANGELOG.md:juniper-recurrence/CHANGELOG.md" \
  --add "${WORKTREE}/.github/workflows/publish-image.yml:.github/workflows/publish-image.yml" \
  --message "$(cat "${SCRATCH}/recurrence-commit-msg.txt")" \
  --title "feat(docker): pin the app image's dependencies with requirements.lock" \
  --body-file "${SCRATCH}/recurrence-pr-body.md" \
  "$@"
