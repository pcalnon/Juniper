#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     create_performance_profile.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date:          2025-10-11
# Last Modified: 2025-12-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     This script Profiles the Python class for optimization
#
#####################################################################################################################################################################################################
# Notes:
#
########################################################################################################)#############################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#     Moved defs to config file
#
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Source script config file
#####################################################################################################################################################################################################
set -o functrace
# shellcheck disable=SC2155
export PARENT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
source "conf/init.conf"; SUCCESS="$?"

# shellcheck disable=SC1091
[[ "${SUCCESS}" != "0" ]] && { source "conf/config_fail.conf"; log_error "${SUCCESS}" "${PARENT_PATH_PARAM}" "conf/init.conf" "${LINENO}" "${LOG_FILE}"; }
log_debug "Successfully Configured Current Script: $(basename "${PARENT_PATH_PARAM}"), by Sourcing the Init Config File: ${INIT_CONF}, Returned: \"${SUCCESS}\""


#######################################################################################################################################################################################
# Define Script Functions
#######################################################################################################################################################################################
function round_size() {
    SIZEF="${1}"
    SIZE="${SIZEF%.*}"
    DEC="0.${DIG}"
    if (( $(echo "${DEC} >= 0.5" | bc -l) )); then
        SIZE=$(( SIZE + 1 ))
    fi
    echo "${SIZE}"
}

function current_size() {
    CURRENT_SIZE="${1}"
    LABEL="${CURRENT_SIZE: -1}"
    SIZEF="${CURRENT_SIZE::-1}"
    for i in "${!SIZE_LABELS[@]}"; do
        if [[ "${SIZE_LABELS[${i}]}" == "${LABEL}" ]]; then
            break
        else
            #SIZE=$(( SIZE * SIZE_LABEL_MAG ))
            #SIZEF=$(( SIZEF * SIZE_LABEL_MAG ))
            SIZEF="$(echo "${SIZEF} * ${SIZE_LABEL_MAG}" | bc -l)"
        fi
    done
    SIZE="$(round_size "${SIZEF}")"
    echo "${SIZE}"
}

function readable_size() {
    CURRENT_SIZE="${1}"
    LABEL_INDEX=0
    export BYTES_LABEL=""
    while (( $(echo "${CURRENT_SIZE} >= ${SIZE_LABEL_MAG}" | bc -l) )); do
        CURRENT_SIZE="$(echo "${CURRENT_SIZE} / ${SIZE_LABEL_MAG}" | bc -l)"
        LABEL_INDEX=$(( LABEL_INDEX + 1 ))
    done
    SIZE="$(round_size "${CURRENT_SIZE}")"
    if (( LABEL_INDEX > 0 )); then
        BYTE_LABEL="${SIZE_LABELS[0]}"
    fi
    READABLE="${SIZE} ${SIZE_LABELS[${LABEL_INDEX}]}${BYTE_LABEL}"
    echo "${READABLE}"
}


#######################################################################################################################################################################################
# Generate Profile for performance tuning
# 	python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py)
#	python -m cProfile -o program.prof my_program.py
#######################################################################################################################################################################################
echo "Starting execution: ${DATE_STAMP}"
echo "time ${PYTHON} ${PARAM_LIST} ${PYTHON_FILE}"
time ${PYTHON} "${PARAM_LIST}" "${PYTHON_FILE}"
