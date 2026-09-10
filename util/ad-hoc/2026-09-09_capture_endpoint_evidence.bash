#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Project     : Juniper
# Sub-Project : juniper-ml (ad-hoc)
# Application : canopy E2E validation arc
# Author      : Paul Calnon
# License     : MIT License
# Created     : 2026-09-09
# Status      : single-use (Phase 5 evidence capture)
# Retire when : the Phase-5 record is merged
# Related     : notes/JUNIPER_2026-08-09_JUNIPER-CANOPY_E2E-VALIDATION-EVIDENCE.md (Phase 5)
# ---------------------------------------------------------------------------
# WHY. Four Phase-5 claims were made from live `curl`s whose output was never
# written to a transcript, so the validation lanes scored them UNVERIFIABLE:
# the pool-history entries behind "F-CANOPY-036 is fixed at the server", the
# transport counters behind F-CASCOR-004's A/B, the equities availability behind
# the M-DATASET-17..26 re-attribution, and the identity of the cascor leg that
# served the third growth window. This writes all of them to one file.
#
# Usage: bash util/ad-hoc/2026-09-09_capture_endpoint_evidence.bash [OUTFILE]
set -uo pipefail

OUT="${1:-reports/e2e-canopy-2026-09-02/transcripts/2026-09-09_endpoint_evidence.txt}"
CANOPY="${JUNIPER_E2E_CANOPY_URL:-http://127.0.0.1:8052}"
CASCOR="${JUNIPER_E2E_CASCOR_URL:-http://127.0.0.1:8202}"
DATA="${JUNIPER_E2E_DATA_URL:-http://127.0.0.1:8101}"

grab() {
    printf '\n== %s ==\n' "$2"
    curl -s --max-time 20 "$1" || printf '<curl failed: exit %s>\n' "$?"
    printf '\n'
}

{
    printf '# canopy E2E arc — endpoint evidence captured %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '# Written by util/ad-hoc/2026-09-09_capture_endpoint_evidence.bash to archive four\n'
    printf '# Phase-5 claims that were originally made from un-transcripted live reads.\n'
    grab "$CANOPY/v1/health" "canopy $CANOPY /v1/health (the leg every browser instrument drove)"
    grab "$CASCOR/v1/health" "cascor $CASCOR /v1/health (git_sha is the WORKTREE BASE, not the fix — see the provenance note below)"
    grab "$CANOPY/api/v1/candidates/pool-history" "canopy /api/v1/candidates/pool-history (F-CANOPY-036: the entries accumulated in window 3)"
    grab "$CASCOR/v1/metrics/transport" "cascor /v1/metrics/transport (F-CASCOR-004: active_connections / send_failures)"
    grab "$CASCOR/v1/network" "cascor /v1/network (the fixture)"
    grab "$DATA/v1/generators" "data $DATA /v1/generators (M-DATASET-17..26: equities availability)"
} >"$OUT" 2>&1

printf 'wrote %s (%s lines)\n' "$OUT" "$(wc -l <"$OUT")"
