#!/usr/bin/env bash
#
# Project:     Juniper
# Sub-Project: juniper-ml
# Application: Performance lane -- PF-8 follow-up
# Author:      Paul Calnon
# License:     MIT License
#
# WHAT THIS IS
# ------------
# Brings up a cascor listener, drives ONE training run through it, and censuses the listener's
# own threads while its INITIAL output-layer pass runs -- the discriminating test left open by
# notes/JUNIPER_2026-09-10_JUNIPER-ECOSYSTEM_PERF-LANE-PF8-OCCUPANCY-PROBE.md section 4.2.
#
# It needs NO juniper-data: POST /v1/training/start materialises the in-process 'spiral'
# generator (api/routes/training.py, _generate_spiral_data), so this is a single-service probe.
#
# The census counts BURNING THREADS, not thread names: a validated OpenBLAS burn on this host
# runs 16 threads all named 'python', so width is the discriminator and names are not. See
# 2026-09-10_listener_thread_census.py.
#
# ARMS
#   unpinned  the service default -- configure_blas_threads() is a no-op unless
#             JUNIPER_CASCOR_BLAS_THREADS is set, so BLAS loads at the runtime default
#   pinned    OMP/MKL/OPENBLAS_NUM_THREADS=2 exported before the interpreter starts
#
# The variables are read once at library load, so the arm must be applied to the LISTENER's
# environment at launch. This script does that itself; do not export them into your shell.
#
# USAGE
#   bash util/ad-hoc/2026-09-10_listener_burst_probe.bash --arm unpinned --port 8217 \
#        --outdir /path/to/evidence
#
# It always tears the listener down, including on error.

set -euo pipefail

ARM="unpinned"
PORT="8217"
OUTDIR=""
CENSUS_SECONDS="90"
# Cell c000's values by default. Lowering output/candidate epochs lets a census window reach the
# SECOND and THIRD output passes, which is how you test whether the burst is confined to the
# INITIAL pass -- train_output_layer has no early exit (`for epoch in range(epochs)`, no break),
# so every pass runs its full budget and a short budget shortens every pass equally.
OUTPUT_EPOCHS="4000"
CANDIDATE_EPOCHS=""
MAX_ITERATIONS="10"
CASCOR_SRC="/home/pcalnon/Development/python/Juniper/juniper-cascor/src"
CASCOR_ENV="/opt/miniforge3/envs/JuniperCascor1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --census-seconds) CENSUS_SECONDS="$2"; shift 2 ;;
        --output-epochs) OUTPUT_EPOCHS="$2"; shift 2 ;;
        --candidate-epochs) CANDIDATE_EPOCHS="$2"; shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${OUTDIR}" ]]; then
    echo "REFUSED: --outdir is required" >&2
    exit 2
fi
case "${ARM}" in
    unpinned|pinned|openblas-only|omp-only|mkl-only) ;;
    *)
        echo "REFUSED: --arm must be one of: unpinned pinned openblas-only omp-only mkl-only" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTDIR}"
LOG_DIR="${OUTDIR}/cascor-logs-${ARM}"
mkdir -p "${LOG_DIR}"

# Port check. `ss` missing would make this read "free" for every port, so require it -- a
# fail-open port check is how a probe silently lands on someone else's service.
if ! command -v ss >/dev/null 2>&1; then
    echo "REFUSED: ss(8) not found; the port check would fail open" >&2
    exit 2
fi
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "REFUSED: port ${PORT} is already listening" >&2
    exit 2
fi

LISTENER_PID=""
CENSUS_PID=""
cleanup() {
    if [[ -n "${CENSUS_PID}" ]] && kill -0 "${CENSUS_PID}" 2>/dev/null; then
        kill "${CENSUS_PID}" 2>/dev/null || true
        wait "${CENSUS_PID}" 2>/dev/null || true
    fi
    if [[ -n "${LISTENER_PID}" ]] && kill -0 "${LISTENER_PID}" 2>/dev/null; then
        echo "tearing down listener ${LISTENER_PID}"
        kill "${LISTENER_PID}" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "${LISTENER_PID}" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "${LISTENER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "=== arm=${ARM} port=${PORT} ==="
echo "loadavg before: $(cat /proc/loadavg)"

# Launch. The arm is applied to the listener's own environment here, before the interpreter
# starts, because the three variables are read once at BLAS load time.
#
# The single-variable arms are the discrimination the probe note could not make: OpenBLAS and
# libgomp are BOTH 16 wide by default on this 16-core host, so a 16-thread burst does not name
# the library on width alone. Setting exactly one variable does name it -- whichever variable
# removes the burst owns the pool.
# NOTE: env(1) stops accepting options at the first NAME=VALUE, so every -u MUST precede every
# assignment. Getting this wrong makes env treat "-u" as the command and the listener dies at
# startup with "env: '-u': No such file or directory" -- which looks like a launch bug, not a
# mis-specified arm.
ARM_ENV=(-u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS)
case "${ARM}" in
    pinned)        ARM_ENV=(OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2) ;;
    openblas-only) ARM_ENV=(-u OMP_NUM_THREADS -u MKL_NUM_THREADS OPENBLAS_NUM_THREADS=2) ;;
    omp-only)      ARM_ENV=(-u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS OMP_NUM_THREADS=2) ;;
    mkl-only)      ARM_ENV=(-u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS MKL_NUM_THREADS=2) ;;
esac

env -C "${CASCOR_SRC}" \
    "${ARM_ENV[@]}" \
    LD_LIBRARY_PATH= \
    JUNIPER_CASCOR_METRICS_ENABLED=true \
    JUNIPER_CASCOR_AUTO_START=false \
    JUNIPER_CASCOR_AUTO_START_DATA_SERVICE=false \
    JUNIPER_CASCOR_LOG_LEVEL="${CASCOR_LOG_LEVEL:-INFO}" \
    JUNIPER_CASCOR_LOG_DIR="${LOG_DIR}" \
    "${CASCOR_ENV}/bin/uvicorn" api.app:create_app --factory --host 127.0.0.1 --port "${PORT}" \
    > "${OUTDIR}/listener-${ARM}.log" 2>&1 &
LISTENER_PID=$!
echo "listener pid ${LISTENER_PID}"

# Wait for health.
READY=0
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1; then READY=1; break; fi
    if ! kill -0 "${LISTENER_PID}" 2>/dev/null; then
        echo "listener died during startup; log tail:" >&2
        tail -30 "${OUTDIR}/listener-${ARM}.log" >&2
        exit 1
    fi
    sleep 1
done
if [[ "${READY}" != "1" ]]; then
    echo "listener never became healthy; log tail:" >&2
    tail -30 "${OUTDIR}/listener-${ARM}.log" >&2
    exit 1
fi
echo "listener healthy"

# Census the listener's own threads, starting BEFORE the run so the first output pass is inside
# the window from its first second.
"${CASCOR_ENV}/bin/python" "${HERE}/2026-09-10_listener_thread_census.py" \
    --pid "${LISTENER_PID}" \
    --duration "${CENSUS_SECONDS}" \
    --interval 0.5 \
    --out "${OUTDIR}/census-${ARM}.json" \
    > "${OUTDIR}/census-${ARM}.txt" 2>&1 &
CENSUS_PID=$!
sleep 1

# Drive one run. Params match pf8-occupancy-probe cell c000; radius 10.0 because the unit-radius
# spiral is degenerate for candidate training (cascor's own generator docstring).
PARAMS="\"max_epochs\": ${OUTPUT_EPOCHS}, \"output_epochs\": ${OUTPUT_EPOCHS}, \"max_iterations\": ${MAX_ITERATIONS}, \"max_hidden_units\": 10, \"candidate_pool_size\": 4, \"early_stopping\": true"
if [[ -n "${CANDIDATE_EPOCHS}" ]]; then
    PARAMS="${PARAMS}, \"candidate_epochs\": ${CANDIDATE_EPOCHS}"
fi
START_BODY="{
  \"dataset\": {\"generator\": \"spiral\", \"params\": {\"n_spirals\": 2, \"n_points_per_spiral\": 200, \"n_rotations\": 2.0, \"noise\": 0.05, \"radius\": 10.0, \"seed\": 20260807}},
  \"params\": {${PARAMS}}
}"
echo "starting training..."
curl -sf -X POST "http://127.0.0.1:${PORT}/v1/training/start" \
    -H 'Content-Type: application/json' \
    -d "${START_BODY}" \
    -o "${OUTDIR}/start-response-${ARM}.json" \
    || { echo "START FAILED; response:" >&2; cat "${OUTDIR}/start-response-${ARM}.json" 2>/dev/null >&2; tail -20 "${OUTDIR}/listener-${ARM}.log" >&2; exit 1; }
echo "start accepted"

# Let the census run to completion.
wait "${CENSUS_PID}" 2>/dev/null || true
CENSUS_PID=""

echo "loadavg after: $(cat /proc/loadavg)"
echo "--- census summary (${ARM}) ---"
cat "${OUTDIR}/census-${ARM}.txt"
echo "--- initial output pass, from the listener log ---"
grep -c "train_output_layer: Output Layer Training - Epoch" "${LOG_DIR}"/*.log 2>/dev/null || true
