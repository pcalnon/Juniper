#!/usr/bin/env bash
#####################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-ml
# Application:   cross-repo tooling (ad-hoc)
# File Name:     2026-09-09_open_closing_handoff_pr.bash
# Author:        Paul Calnon
# Version:       0.1.0
#
# Date Created:  2026-09-09
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Opens the closing juniper-ml PR for the container-registry item-6 session: archives the
#    thread handoff and the util/ad-hoc/ helpers the session used.
#
#    Every --add here is a NEW file, so the whole-file-upload hazard that bit ml#1869
#    (open_signed_pr.py uploads whole files; the worktree was seven commits stale) does not
#    apply -- there is no prior content to revert. The worktree was fast-forwarded to
#    origin/main a51fe617 regardless.
#
#    Single-use.
#####################################################################################################
set -euo pipefail

SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
WORKTREE="/home/pcalnon/Development/python/Juniper/juniper-ml/.claude/worktrees/tender-splashing-wigderson"
HANDOFF="prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-09_container-registry-item-6-closed-and-a-whole-file-clobber-caught-by-ci.md"
HELPER="/home/pcalnon/Development/python/Juniper/juniper-ml/util/open_signed_pr.py"
PYTHON="/opt/miniforge3/envs/JuniperCascor1/bin/python"

"${PYTHON}" "${HELPER}" \
  --repo juniper-ml \
  --branch docs/handoff-2026-09-09-container-registry-item-6 \
  --add "${WORKTREE}/${HANDOFF}:${HANDOFF}" \
  --add "${WORKTREE}/util/ad-hoc/2026-09-09_open_worker_torch_pin_pr.bash:util/ad-hoc/2026-09-09_open_worker_torch_pin_pr.bash" \
  --add "${WORKTREE}/util/ad-hoc/2026-09-09_open_cascor_lockfile_6c_pr.bash:util/ad-hoc/2026-09-09_open_cascor_lockfile_6c_pr.bash" \
  --add "${WORKTREE}/util/ad-hoc/2026-09-09_open_worker_lockfile_6c_pr.bash:util/ad-hoc/2026-09-09_open_worker_lockfile_6c_pr.bash" \
  --add "${WORKTREE}/util/ad-hoc/2026-09-09_open_recurrence_lock_6e_pr.bash:util/ad-hoc/2026-09-09_open_recurrence_lock_6e_pr.bash" \
  --add "${WORKTREE}/util/ad-hoc/2026-09-09_open_ml_lint_6g_pr.bash:util/ad-hoc/2026-09-09_open_ml_lint_6g_pr.bash" \
  --add "${WORKTREE}/util/ad-hoc/2026-09-09_fix_ml6g_reference_clobber.bash:util/ad-hoc/2026-09-09_fix_ml6g_reference_clobber.bash" \
  --message "$(cat "${SCRATCH}/closing-commit-msg.txt")" \
  --title "docs(handoff): container-registry item 6 closed, and CI caught a whole-file clobber" \
  --body-file "${SCRATCH}/closing-pr-body.md" \
  "$@"
