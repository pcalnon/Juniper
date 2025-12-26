#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       0.1.4 (0.7.3)
# File Name:     change_path.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date:          2025-10-11
# Last Modified: 2025-12-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
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


#####################################################################################################################################################################################################
#
#####################################################################################################################################################################################################
SLEEPY_TIME=3

OLD_PATH="Juniper\/src\/prototypes"
NEW_PATH="JuniperCanopy"

echo "Bash Version: $(/usr/bin/env bash --version)"
echo "Current Working Dir: $(pwd)"

echo "Changing Path:  Old: \"${OLD_PATH}\", New: \"${NEW_PATH}\""

echo "${SLEEPY_TIME} second Warning"
sleep "${SLEEPY_TIME}"

echo "Filenames:"

while read -r FILENAME; do
    # echo -ne "    grep -nI \"${OLD_PATH}\" \"${FILENAME}\"\n"
    # grep -nI "${OLD_PATH}" "${FILENAME}"

    echo -ne "    sed -i \"s/${OLD_PATH}/${NEW_PATH}/g\" ${FILENAME}\n\n"
    sed -i "s/${OLD_PATH}/${NEW_PATH}/g" "${FILENAME}"

done <<< "$(grep -rnI "${OLD_PATH}" ./* | awk -F ":" '{print $1;}' | sort -u)"
