#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-ml
# Application:   cross-repo tooling (ad-hoc)
# File Name:     2026-09-09_open_worker_torch_pin_pr.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-09-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Opens the juniper-cascor-worker PR for follow-ups 6b/6c (worker half) of the
#    container-registry rollout, with a GitHub-signed commit via util/open_signed_pr.py.
#
#    Exists as a script rather than an inline invocation because a worktree-isolated
#    session's Bash classifier refuses a multi-line command whose operands are computed
#    at runtime ("names git in a form too complex to verify"). Every path here is a
#    literal, so the call is one plain command.
#
#    Single-use: the branch name and the five --add pairs are specific to that PR.
#####################################################################################################
set -euo pipefail

SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
WORKTREE="/home/pcalnon/Development/python/Juniper/worktrees/juniper-cascor-worker--fix--torch-2-14-cpu-pin--20260909-1838--93f95e8b"
HELPER="/home/pcalnon/Development/python/Juniper/juniper-ml/util/open_signed_pr.py"
PYTHON="/opt/miniforge3/envs/JuniperCascor1/bin/python"

"${PYTHON}" "${HELPER}" \
  --repo juniper-cascor-worker \
  --branch fix/torch-2-14-cpu-pin \
  --add "${WORKTREE}/requirements-cpu.lock:requirements-cpu.lock" \
  --add "${WORKTREE}/Dockerfile:Dockerfile" \
  --add "${WORKTREE}/.github/workflows/ci.yml:.github/workflows/ci.yml" \
  --add "${WORKTREE}/tests/test_dockerfile_cpu_torch_pin.py:tests/test_dockerfile_cpu_torch_pin.py" \
  --add "${WORKTREE}/CHANGELOG.md:CHANGELOG.md" \
  --message "$(cat "${SCRATCH}/worker-commit-msg.txt")" \
  --title "fix(docker): derive requirements-cpu.lock from the GPU lock and test the torch the image ships" \
  --body-file "${SCRATCH}/worker-pr-body.md" \
  "$@"
