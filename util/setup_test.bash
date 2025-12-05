#!/usr/bin/env bash


COMMENT_REGEX="^[[:space:]]*#.*$"

USE_CONDA="1"
USE_MAMBA="$(echo "$(( ( A + 1 ) % 2 ))")"

CONDA_CMD="conda"
MAMBA_CMD="mamba"

CONDA_OFFSET="2"
MAMBA_OFFSET="3"


if [[ ${USE_CONDA} ]]; then
    CMD="${CONDA_CMD}"
    OFFSET="${CONDA_OFFSET}"
elif [[ ${USE_MAMBA} ]]; then
    CMD="${MAMBA_CMD}"
    OFFSET="${MAMBA_OFFSET}"
else
    echo "Borked"
    exit 1
fi


eval "${CMD} env list" | grep -v -e "${COMMENT_REGEX}" | tail -n +${OFFSET}
