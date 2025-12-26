#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     random_seed.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date:          2024-04-01
# Last Modified: 2025-12-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#   Call a python script to generate a new, cryptographically secure random value for use as seed in psuedo random functions
#
#####################################################################################################################################################################################################
# Notes:
#   /Users/pcalnon/opt/anaconda3/envs/pytorch_cuda/bin/python
#   /home/pcalnon/anaconda3/envs/pytorch_cuda/bin/python
#
########################################################################################################)#############################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################



#####################################################################################################################################################################################################
# Initialize script by sourcing the init_conf.bash config file
#####################################################################################################################################################################################################
set -o functrace
# shellcheck disable=SC2155
export PARENT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")" && INIT_CONF="conf/init.conf"
# shellcheck disable=SC2015
# shellcheck source=conf/init.conf
# shellcheck disable=SC1091
[[ -f "${INIT_CONF}" ]] && source "${INIT_CONF}" || { echo "Init Config File Not Found. Unable to Continue."; exit 1; }


####################################################################################################
# Get the OS name and find active python binary
####################################################################################################
OS_NAME=$(${OS_NAME_SCRIPT})

# Validate Local OS
if [[ "${OS_NAME}" == "${OS_LINUX}" ]]; then
    PYTHON_CMD="${HOME}/${CONDA_LINUX}/${PYTHON_LOC}"
elif [[ "${OS_NAME}" == "${OS_MACOS}" ]]; then
    PYTHON_CMD="${HOME}/${CONDA_MACOS}/${PYTHON_LOC}"
elif [[ "${OS_NAME}" == "${OS_WINDOWS}" ]]; then
    echo "Error: Why the hell are you running ${OS_WINDOWS}??"
    exit 1
else
    echo "Error: You are running an ${OS_UNKNOWN} OS. Cowardly not wading into this crazy."
    exit 2
fi

# shellcheck disable=SC2155
export PYTHON_VER="$(${PYTHON_CMD} --version)"
"${PYTHON_CMD}" "${RANDOM_SEED}"

RESULT="$?"
if [[ "${RESULT}" == "0" ]]; then
    echo -ne "\nSuccess!!  "
else
    echo -ne "\nFailure :(  "
fi
echo "Python Script: ${NEW_RANDOM_SEED_FILE_NAME}, returned: ${RESULT}"
