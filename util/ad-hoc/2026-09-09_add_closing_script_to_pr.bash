#!/usr/bin/env bash
# Single-use: adds util/ad-hoc/2026-09-09_open_closing_handoff_pr.bash and the corrected
# handoff to juniper-ml#1870's branch. Project: Juniper / juniper-ml / ad-hoc tooling.
# Author: Paul Calnon. Created 2026-09-09. MIT License.
set -euo pipefail
SCRATCH="/tmp/claude-1000/-home-pcalnon-Development-python-Juniper-juniper-ml/15755553-7151-44a6-88a7-0dfb60e5f079/scratchpad"
W="/home/pcalnon/Development/python/Juniper/juniper-ml/.claude/worktrees/tender-splashing-wigderson"
H="prompts/thread-handoff_automated-prompts/HANDOFF_2026-09-09_container-registry-item-6-closed-and-a-whole-file-clobber-caught-by-ci.md"
/opt/miniforge3/envs/JuniperCascor1/bin/python \
  /home/pcalnon/Development/python/Juniper/juniper-ml/util/ad-hoc/2026-08-26_commit_files_to_pr_branch.py \
  --repo juniper-ml \
  --branch docs/handoff-2026-09-09-container-registry-item-6 \
  --add "${W}/${H}:${H}" \
  --add "${W}/util/ad-hoc/2026-09-09_open_closing_handoff_pr.bash:util/ad-hoc/2026-09-09_open_closing_handoff_pr.bash" \
  --message "$(cat "${SCRATCH}/closing-followup-msg.txt")"
