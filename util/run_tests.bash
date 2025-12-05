#!/usr/bin/env bash
#####################################################################
# Script to run tests with proper PYTHONPATH
#####################################################################

# Get absolute path to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"

# Export PYTHONPATH
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

# Run pytest with all arguments passed through
cd "${SCRIPT_DIR}" || exit 1
/opt/miniforge3/envs/JuniperPython/bin/python -m pytest "$@"
