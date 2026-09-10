#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-ml
# Application:   cross-repo tooling (ad-hoc)
# File Name:     2026-09-09_fix_ml6g_reference_clobber.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-09-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Repairs the docs/REFERENCE.md whole-file clobber on juniper-ml#1869's branch: the
#    first commit uploaded a copy based on a worktree seven commits behind origin/main,
#    reverting #1857 / #1868's soak corrections. Re-uploads the file rebuilt from
#    origin/main with only this PR's one-line ceiling widen applied.
#
#    Single-use. The general lesson is recorded in the session handoff: open_signed_pr.py
#    and this helper both upload WHOLE files, so the worktree must be current with
#    origin/main for every file in the --add set, not just the ones being edited.
#####################################################################################################
set -euo pipefail

SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
WORKTREE="/home/pcalnon/Development/python/Juniper/juniper-ml/.claude/worktrees/tender-splashing-wigderson"
HELPER="/home/pcalnon/Development/python/Juniper/juniper-ml/util/ad-hoc/2026-08-26_commit_files_to_pr_branch.py"
PYTHON="/opt/miniforge3/envs/JuniperCascor1/bin/python"

"${PYTHON}" "${HELPER}" \
  --repo juniper-ml \
  --branch fix/lint-workflow-paths-working-directory \
  --add "${WORKTREE}/docs/REFERENCE.md:docs/REFERENCE.md" \
  --message "$(cat "${SCRATCH}/ml6g-fix-msg.txt")" \
  "$@"
